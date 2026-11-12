"""Simple research loop — one self-contained LangGraph file.

Flow (modified for the IDS 706 Week 11 life-expectancy pipeline):
  START → warehouse_loader → data_scientist → feature_engineer → hypothesis → ml_engineer → judge
                                                                  ↑__________________________|
                                                                  |  (revise: memory → context → hypothesis)
                                                                  └── accept/stop → holdout_eval → error_analyst → END

New vs. the in-class template:
  * warehouse_loader  — reads the cleaned Airflow output straight from Postgres
                        (analytics.country_year_features), CSV fallback.
  * feature_engineer  — drops target-leakage columns (single-feature separability test)
                        and high-cardinality identifiers, log-transforms skewed features.
  * error_analyst     — after the sealed holdout: confusion matrix, precision/recall,
                        ROC AUC and per-slice (e.g. region) F1 for the best model.

LLM / train helpers live in loop_runtime.py (Cursor SDK + OpenRouter + heuristics).
"""

from __future__ import annotations

import json
import operator
import os
import sys
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
# Repo root locally; /app in the gateway container (loop_runtime.py is mounted there).
for _p in (REPO, Path("/app")):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from loop_runtime import (  # noqa: E402
    DEFAULT_MODEL,
    _build_model,
    FAMILIES,
    _SPEC_SCHEMA_HINT,
    _coerce_experiment_spec,
    _fit_score,
    _heuristic_plan,
    _heuristic_profile,
    _heuristic_spec,
    _heuristic_verdict,
    _repair_heuristic,
    _resolve_model,
    _validate_spec,
    extract_json_object,
    llm_configured,
    openrouter_chat,
)

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class SimpleLoopState(TypedDict, total=False):
    dataset_path: str
    target_column: str
    target_score: float
    max_cycles: int
    cycle: int
    seed: int
    holdout_frac: float
    model: str  # OpenRouter / Cursor model id from the UI / env
    profile: dict
    brief: dict
    hypothesis: dict  # conceptual plan from hypothesis_generator (not HPT numbers)
    experiment_spec: dict  # concrete HPT config from ml_engineer
    execution_result: dict
    verdict: dict
    memory: dict
    best_spec: dict
    best_val_f1: float
    holdout_metrics: dict
    source: dict  # where warehouse_loader got the data from
    feature_report: dict  # what feature_engineer dropped / created
    error_analysis: dict  # error_analyst output on the sealed holdout
    stop_reason: str
    steps: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def node_data_scientist(state: SimpleLoopState) -> dict:
    """Profile CSV via OpenRouter; heuristic fallback if key/parse fails."""
    path = Path(state["dataset_path"])
    target = state["target_column"]
    df = pd.read_csv(path)
    if target not in df.columns:
        return {
            "profile": {},
            "steps": [{"node": "data_scientist", "note": f"target {target!r} missing", "status": "error"}],
        }

    profile = _heuristic_profile(df, target)
    used_llm = False
    note = "heuristic"
    model = _resolve_model(state)

    if llm_configured(model):
        missing = {c: float(df[c].isna().mean()) for c in df.columns}
        nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
        sample = df.head(6).astype(str).replace("nan", "").to_dict(orient="records")
        system = (
            "You are the Data Scientist only — not the hypothesizer and not the ML engineer. "
            "Your job is dataset understanding: profile the CSV and flag risk. "
            "Do NOT propose model families, hyperparameters, or experiment configs. "
            "Return ONLY one JSON object with keys: "
            "n_rows (int), n_columns (int), target (string), target_rate (number|null), "
            "numeric_features (string[]), categorical_features (string[]), "
            "leakage_suspects (string[]), recommended_drops (string[]), "
            "notes (string[], short bullets about data risks — leakage, imbalance, sparsity). "
            "Flag PassengerId/Name/Ticket/Cabin-style identifiers and near-unique text as leakage. "
            "Do not invent columns that are not present."
        )
        user = (
            f"dataset={path.name}\ntarget={target}\nn_rows={len(df)}\n"
            f"columns={list(df.columns)}\n"
            f"dtypes={{{', '.join(f'{c}:{df[c].dtype}' for c in df.columns)}}}\n"
            f"missing_rate={missing}\nnunique={nunique}\n"
            f"sample_rows={sample}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=900, temperature=0.15)
        if content:
            parsed = extract_json_object(content)
            if parsed and (
                "recommended_drops" in parsed
                or "leakage_suspects" in parsed
                or "numeric_features" in parsed
            ):
                profile = parsed
                profile.setdefault("n_rows", int(len(df)))
                profile.setdefault("n_columns", int(len(df.columns)))
                profile.setdefault("target", target)
                profile.setdefault("recommended_drops", profile.get("leakage_suspects") or [])
                used_llm = True
                note = f"llm profile · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    return {
        "profile": profile,
        "steps": [
            {
                "node": "data_scientist",
                "note": note,
                "used_llm": used_llm,
                "model": model if used_llm else None,
                "output": profile,
            }
        ],
    }



WAREHOUSE_TABLE = "analytics.country_year_features"
WORK_DIR = Path(tempfile.gettempdir()) / "agent_loop"


def node_warehouse_loader(state: SimpleLoopState) -> dict:
    """Pull the Airflow-cleaned table from Postgres; fall back to the CSV export."""
    path = Path(state["dataset_path"])
    url = os.getenv("WAREHOUSE_URL")
    source = {"type": "csv", "path": str(path)}
    note = f"csv file {path.name}"
    if url and path.name.startswith("life_expectancy"):
        try:
            import psycopg2

            with psycopg2.connect(url) as conn, conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {WAREHOUSE_TABLE} ORDER BY iso3, year")
                cols = [c[0] for c in cur.description]
                rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=cols).drop(columns=["loaded_at"], errors="ignore")
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            out = WORK_DIR / "warehouse_snapshot.csv"
            df.to_csv(out, index=False)
            source = {"type": "postgres", "table": WAREHOUSE_TABLE, "rows": int(len(df)), "snapshot": str(out)}
            note = f"postgres {WAREHOUSE_TABLE} · {len(df)} rows"
            return {
                "dataset_path": str(out),
                "source": source,
                "steps": [{"node": "warehouse_loader", "note": note, "output": source}],
            }
        except Exception as exc:  # noqa: BLE001
            note = f"postgres unavailable ({exc.__class__.__name__}); using csv {path.name}"
            source["warehouse_error"] = str(exc)[:200]
    return {
        "source": source,
        "steps": [{"node": "warehouse_loader", "note": note, "output": source}],
    }


def _single_feature_accuracy(x: pd.Series, y: pd.Series) -> float:
    """Train accuracy of a depth-1 tree on one column — ~1.0 means the column encodes the target."""
    from sklearn.tree import DecisionTreeClassifier

    mask = x.notna() & y.notna()
    if mask.sum() < 20 or y[mask].nunique() < 2:
        return 0.0
    stump = DecisionTreeClassifier(max_depth=1).fit(x[mask].to_frame(), y[mask])
    return float(stump.score(x[mask].to_frame(), y[mask]))


def node_feature_engineer(state: SimpleLoopState) -> dict:
    """Deterministic feature step between profiling and hypothesis generation."""
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    profile = dict(state.get("profile") or {})
    y = df[target]

    leakage, identifiers, logged = [], [], []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if _single_feature_accuracy(df[c], y) >= 0.98:
                leakage.append(c)
        elif df[c].nunique(dropna=True) > 50:
            identifiers.append(c)
    drops = [c for c in dict.fromkeys(leakage + identifiers + list(profile.get("recommended_drops") or [])) if c in df.columns and c != target]
    out_df = df.drop(columns=drops)

    for c in list(out_df.columns):
        if c == target or not pd.api.types.is_numeric_dtype(out_df[c]):
            continue
        col = out_df[c].dropna()
        if len(col) and (col > 0).all() and float(col.skew()) > 2.0:
            out_df[f"log10_{c}"] = np.log10(out_df[c])
            out_df = out_df.drop(columns=[c])
            logged.append(c)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out = WORK_DIR / f"engineered_{Path(state['dataset_path']).stem}.csv"
    out_df.to_csv(out, index=False)
    report = {
        "dropped_leakage": leakage,
        "dropped_identifiers": identifiers,
        "dropped_total": drops,
        "log_transformed": logged,
        "n_features": int(out_df.shape[1] - 1),
        "engineered_path": str(out),
    }
    # Columns are physically gone, so downstream agents no longer need to drop them.
    profile["recommended_drops"] = []
    profile["leakage_suspects"] = []
    profile.setdefault("notes", [])
    profile["notes"] = list(profile["notes"]) + [
        f"feature_engineer dropped {drops} and log-transformed {logged}"
    ]
    note = f"dropped {len(drops)} cols (leakage: {leakage or 'none'}); log: {logged or 'none'}"
    return {
        "dataset_path": str(out),
        "profile": profile,
        "feature_report": report,
        "steps": [{"node": "feature_engineer", "note": note, "output": report}],
    }


def node_hypothesis(state: SimpleLoopState) -> dict:
    """Conceptual next experiment from DS profile + Judge feedback — no HPT numbers."""
    profile = state.get("profile") or {}
    brief = state.get("brief") or {}
    verdict = state.get("verdict") or {}
    history = list(brief.get("history") or (state.get("memory") or {}).get("history") or [])
    model = _resolve_model(state)

    plan = _heuristic_plan(profile, history, verdict)
    used_llm = False
    note = "heuristic plan"

    if llm_configured(model):
        system = (
            "You are the Hypothesis Generator only — not the Data Scientist and not the ML Engineer. "
            "Read the data-scientist profile and the latest judge verdict / lessons. "
            "Propose ONE conceptual research direction for the next trial. "
            "Do NOT invent concrete hyperparameter values, C/learning_rate/max_depth numbers, "
            "or a full ExperimentSpec — the ML Engineer owns HPT config. "
            "Return ONLY one JSON object with keys: "
            "direction (1-2 sentences: what should change conceptually), "
            "why (cite profile signals and/or judge observation), "
            "focus (one of: baseline_then_iterate|capacity_or_regularization|model_family_switch|feature_drops), "
            "respect_drops (string[] of columns to keep dropped — copy from profile when relevant), "
            "avoid_repeats (string[] of model families or ideas already tried). "
            "Stay conceptual: e.g. 'try a tree ensemble with less regularization' — never 'C=0.5'."
        )
        user = (
            f"profile={json.dumps(profile, default=str)}\n"
            f"latest_verdict={json.dumps(verdict, default=str)}\n"
            f"brief={json.dumps(brief, default=str)}\n"
            f"recent_history={json.dumps(history[-5:], default=str)}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=500, temperature=0.4)
        if content:
            parsed = extract_json_object(content)
            if parsed and parsed.get("direction"):
                plan = {
                    "direction": parsed.get("direction"),
                    "why": parsed.get("why") or plan.get("why"),
                    "focus": parsed.get("focus") or plan.get("focus"),
                    "respect_drops": parsed.get("respect_drops")
                    if isinstance(parsed.get("respect_drops"), list)
                    else plan.get("respect_drops"),
                    "avoid_repeats": parsed.get("avoid_repeats")
                    if isinstance(parsed.get("avoid_repeats"), list)
                    else plan.get("avoid_repeats"),
                }
                used_llm = True
                note = f"live conceptual hypothesis · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    return {
        "hypothesis": plan,
        "steps": [
            {
                "node": "hypothesis_generator",
                "note": note,
                "hypothesis": plan,
                "output": plan,
                "used_llm": used_llm,
                "model": model if used_llm else None,
            }
        ],
    }


def node_ml_engineer(state: SimpleLoopState) -> dict:
    """Turn conceptual hypothesis into concrete HPT ExperimentSpec, then train/val."""
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    profile = state.get("profile") or {}
    plan = dict(state.get("hypothesis") or {})
    brief = state.get("brief") or {}
    history = list(brief.get("history") or (state.get("memory") or {}).get("history") or [])
    prev_spec = dict(state.get("experiment_spec") or state.get("best_spec") or {})
    seed = int(state.get("seed") or 42)
    holdout_frac = float(state.get("holdout_frac") or 0.2)
    model_id = _resolve_model(state)

    spec = _heuristic_spec(profile, history, plan)
    used_llm = False
    repaired = False
    note = "heuristic HPT"

    default_drops = (
        plan.get("respect_drops")
        or profile.get("recommended_drops")
        or profile.get("leakage_suspects")
        or []
    )
    next_eid = f"exp-{len(history) + 1}"
    raw_llm: str | None = None

    if llm_configured(model_id):
        system = (
            "You are the ML Engineer only — not the Hypothesis Generator and not the Judge. "
            "The hypothesis agent already decided the conceptual direction. "
            "Your job: translate that plan into a concrete hyperparameter / training config. "
            "Return ONLY one JSON object matching this schema exactly: "
            f"{_SPEC_SCHEMA_HINT} "
            "model_family MUST be one of logistic_regression|random_forest|hist_gradient_boosting. "
            "hyperparameters MUST be a JSON object with concrete numbers for that family "
            "(C/max_iter OR n_estimators/max_depth OR learning_rate/max_iter). "
            "drop_columns is a string array. rationale is one sentence. "
            "No markdown fences, no commentary."
        )
        user = (
            f"hypothesis_plan={json.dumps(plan, default=str)}\n"
            f"profile={json.dumps({k: profile.get(k) for k in ('recommended_drops', 'leakage_suspects', 'n_rows', 'target_rate')}, default=str)}\n"
            f"prior_spec={json.dumps(prev_spec, default=str)}\n"
            f"recent_history={json.dumps(history[-3:], default=str)}\n"
            f"allowed_families={list(FAMILIES)}\n"
            f"next_experiment_id={next_eid}\n"
        )
        content, err = openrouter_chat(
            system, user, model=model_id, max_tokens=700, temperature=0.2, json_object=True
        )
        raw_llm = content
        coerced = _coerce_experiment_spec(
            extract_json_object(content or ""),
            experiment_id=next_eid,
            default_drops=default_drops,
        )
        if coerced is None and content and llm_configured(model_id):
            # One strict repair pass — common with Cursor prose wrapping JSON.
            fix_system = (
                "Convert the text into ONE valid ExperimentSpec JSON object. "
                f"Schema example: {_SPEC_SCHEMA_HINT} "
                f"model_family must be one of {list(FAMILIES)}. "
                "Output JSON only."
            )
            fix_user = (
                f"raw_model_output={content[:3500]}\n"
                f"plan={json.dumps(plan, default=str)}\n"
                f"next_experiment_id={next_eid}\n"
            )
            content2, err2 = openrouter_chat(
                fix_system,
                fix_user,
                model=model_id,
                max_tokens=500,
                temperature=0.0,
                json_object=True,
            )
            raw_llm = content2 or content
            coerced = _coerce_experiment_spec(
                extract_json_object(content2 or ""),
                experiment_id=next_eid,
                default_drops=default_drops,
            )
            err = err2 or err
        if coerced:
            spec = coerced
            used_llm = True
            note = f"live HPT config · {model_id}"
        else:
            note = f"llm unparsable HPT; heuristic ({err or 'ok'})"

    err = _validate_spec(spec)
    if err and llm_configured(model_id):
        system = (
            "You are the ML Engineer repairing an invalid HPT ExperimentSpec. "
            f"Return ONLY JSON matching: {_SPEC_SCHEMA_HINT} "
            f"model_family must be one of {list(FAMILIES)}."
        )
        user = (
            f"error={err}\n"
            f"spec={json.dumps(spec, default=str)}\n"
            f"plan={json.dumps(plan, default=str)}\n"
            f"raw_model_output={(raw_llm or '')[:2000]}\n"
        )
        content, _llm_err = openrouter_chat(
            system, user, model=model_id, max_tokens=500, temperature=0.0, json_object=True
        )
        coerced = _coerce_experiment_spec(
            extract_json_object(content or ""),
            experiment_id=next_eid,
            default_drops=default_drops,
        )
        if coerced:
            spec = coerced
            used_llm = True
            repaired = True
            note = f"repaired HPT after: {err}"
            err = _validate_spec(spec)
    if err:
        spec = _repair_heuristic(spec, err)
        repaired = True
        note = f"heuristic repair after: {err}"
        err = _validate_spec(spec)

    cycle = int(state.get("cycle") or 0) + 1
    if err:
        result = {"status": "failed", "error": err, "validation_metrics": {}}
        return {
            "execution_result": result,
            "experiment_spec": spec,
            "cycle": cycle,
            "steps": [
                {
                    "node": "ml_engineer",
                    "note": note,
                    "status": "failed",
                    "repaired": repaired,
                    "used_llm": used_llm,
                    "experiment_id": spec.get("experiment_id"),
                    "model_family": spec.get("model_family"),
                    "spec": spec,
                    "output": {"experiment_spec": spec, "error": err},
                }
            ],
        }

    try:
        result = _fit_score(df, target, spec, seed, holdout_frac, on_holdout=False)
    except Exception as exc:  # noqa: BLE001
        if not repaired:
            spec = _repair_heuristic(spec, str(exc))
            repaired = True
            note = f"heuristic repair after run error: {exc}"
            try:
                result = _fit_score(df, target, spec, seed, holdout_frac, on_holdout=False)
            except Exception as exc2:  # noqa: BLE001
                result = {"status": "failed", "error": str(exc2), "validation_metrics": {}}
        else:
            result = {"status": "failed", "error": str(exc), "validation_metrics": {}}

    f1 = float((result.get("validation_metrics") or {}).get("f1") or 0)
    patch: dict = {
        "execution_result": result,
        "experiment_spec": spec,
        "cycle": cycle,
        "steps": [
            {
                "node": "ml_engineer",
                "note": note if result.get("status") != "succeeded" else f"{note} · f1={f1:.3f}",
                "f1": f1 if result.get("status") == "succeeded" else None,
                "status": result.get("status"),
                "repaired": repaired,
                "used_llm": used_llm,
                "experiment_id": spec.get("experiment_id"),
                "model_family": spec.get("model_family"),
                "validation_metrics": result.get("validation_metrics") or {},
                "duration_seconds": result.get("duration_seconds"),
                "spec": spec,
                "output": {
                    "experiment_spec": spec,
                    "execution_result": {
                        "status": result.get("status"),
                        "validation_metrics": result.get("validation_metrics"),
                        "error": result.get("error"),
                    },
                },
            }
        ],
    }
    if result.get("status") == "succeeded" and f1 > float(state.get("best_val_f1") or -1):
        patch["best_val_f1"] = f1
        patch["best_spec"] = spec
    return patch


def node_judge(state: SimpleLoopState) -> dict:
    """Verdict via OpenRouter; heuristic fallback."""
    result = state.get("execution_result") or {}
    spec = state.get("experiment_spec") or {}
    plan = state.get("hypothesis") or {}
    brief = state.get("brief") or {}
    target_score = float(state.get("target_score") or 0.8)
    model = _resolve_model(state)

    verdict = _heuristic_verdict(result, target_score)
    used_llm = False
    note = "heuristic"

    if llm_configured(model):
        system = (
            "You are the Judge only — not the Hypothesis Generator and not the ML Engineer. "
            "Decide from VALIDATION evidence only — never invent holdout scores, "
            "never rewrite hyperparameters. "
            "Return ONLY one JSON object with keys: "
            "action (accept|revise|reject), observation (1-2 sentences with the val F1), "
            "next_action (short conceptual instruction for the hypothesis agent — no HPT numbers). "
            "accept only if validation F1 meets or exceeds target_score; "
            "revise when under target but recoverable; reject for hopeless/broken runs."
        )
        user = (
            f"target_score={target_score}\n"
            f"hypothesis_plan={json.dumps(plan, default=str)}\n"
            f"spec={json.dumps(spec, default=str)}\n"
            f"result={json.dumps(result, default=str)}\n"
            f"brief={json.dumps(brief, default=str)}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=500, temperature=0.1)
        if content:
            parsed = extract_json_object(content)
            if parsed and parsed.get("action") in {"accept", "revise", "reject"}:
                verdict = {
                    "action": parsed["action"],
                    "observation": parsed.get("observation") or verdict.get("observation"),
                    "next_action": parsed.get("next_action") or verdict.get("next_action"),
                }
                used_llm = True
                note = f"llm verdict · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    return {
        "verdict": verdict,
        "steps": [
            {
                "node": "judge",
                "note": note,
                "verdict": verdict,
                "used_llm": used_llm,
                "model": model if used_llm else None,
            }
        ],
    }


def node_memory_writer(state: SimpleLoopState) -> dict:
    """Append this cycle's lesson so the next hypothesis can see history."""
    mem = dict(state.get("memory") or {"history": [], "findings": [], "lessons": []})
    hist = list(mem.get("history") or [])
    hist.append(
        {
            "cycle": state.get("cycle"),
            "hypothesis": state.get("hypothesis"),
            "spec": state.get("experiment_spec"),
            "model_family": (state.get("experiment_spec") or {}).get("model_family"),
            "f1": (state.get("execution_result") or {}).get("validation_metrics", {}).get("f1"),
            "verdict": state.get("verdict"),
        }
    )
    mem["history"] = hist
    lessons = list(mem.get("lessons") or [])
    lessons.append(str((state.get("verdict") or {}).get("observation") or ""))
    mem["lessons"] = lessons
    return {"memory": mem, "steps": [{"node": "memory_writer", "note": "filed lesson"}]}


def node_context_builder(state: SimpleLoopState) -> dict:
    """Build a short brief for the next hypothesis from recent memory."""
    mem = state.get("memory") or {}
    brief = {
        "history": mem.get("history") or [],
        "lessons": (mem.get("lessons") or [])[-3:],
    }
    return {"brief": brief, "steps": [{"node": "context_builder", "note": "next brief"}]}


def node_holdout(state: SimpleLoopState) -> dict:
    """Score the best spec ONCE on the sealed holdout (same seed as ml_engineer)."""
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    spec = state.get("best_spec") or state.get("experiment_spec") or {}
    seed = int(state.get("seed") or 42)
    holdout_frac = float(state.get("holdout_frac") or 0.2)
    if _validate_spec(spec):
        spec = _repair_heuristic(dict(spec), "invalid best_spec")
    metrics = _fit_score(df, target, spec, seed, holdout_frac, on_holdout=True)
    return {
        "holdout_metrics": metrics,
        "stop_reason": "finalize",
        "steps": [{"node": "holdout_eval", "note": "sealed holdout", "metrics": metrics}],
    }


def node_error_analyst(state: SimpleLoopState) -> dict:
    """Re-fit the best spec on train+val and break down errors on the sealed holdout."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    spec = state.get("best_spec") or state.get("experiment_spec") or {}
    seed = int(state.get("seed") or 42)
    holdout_frac = float(state.get("holdout_frac") or 0.2)
    y = df[target]
    drops = [c for c in (spec.get("drop_columns") or []) if c in df.columns and c != target]
    X = df.drop(columns=[target, *drops], errors="ignore")
    # Same split as loop_runtime._fit_score(on_holdout=True).
    X_tmp, X_hold, y_tmp, y_hold = train_test_split(
        X, y, test_size=holdout_frac, random_state=seed, stratify=y if y.nunique() > 1 else None
    )
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]
    )
    pipe = Pipeline([("pre", pre), ("clf", _build_model(str(spec.get("model_family") or "logistic_regression"), dict(spec.get("hyperparameters") or {})))])
    pipe.fit(X_tmp, y_tmp)
    pred = pipe.predict(X_hold)
    analysis: dict = {
        "model_family": spec.get("model_family"),
        "n_holdout": int(len(y_hold)),
        "accuracy": round(float(accuracy_score(y_hold, pred)), 4),
        "precision": round(float(precision_score(y_hold, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_hold, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_hold, pred, zero_division=0)), 4),
        "confusion_matrix": {"labels": [0, 1], "matrix": confusion_matrix(y_hold, pred, labels=[0, 1]).tolist()},
    }
    if hasattr(pipe, "predict_proba") and y_hold.nunique() == 2:
        analysis["roc_auc"] = round(float(roc_auc_score(y_hold, pipe.predict_proba(X_hold)[:, 1])), 4)

    slice_cols = [c for c in cat if 2 <= X[c].nunique() <= 12]
    if slice_cols:
        sc = "region" if "region" in slice_cols else slice_cols[0]
        rows = []
        hold = X_hold.assign(_y=y_hold.values, _p=pred)
        for key, g in hold.groupby(sc):
            rows.append(
                {
                    sc: key,
                    "n": int(len(g)),
                    "accuracy": round(float(accuracy_score(g["_y"], g["_p"])), 4),
                    "f1": round(float(f1_score(g["_y"], g["_p"], zero_division=0)), 4),
                    "positive_rate": round(float(g["_y"].mean()), 4),
                }
            )
        analysis["slice_column"] = sc
        analysis["slices"] = sorted(rows, key=lambda r: r["accuracy"])
        worst = analysis["slices"][0]
        note = f"holdout f1={analysis['f1']:.3f}; weakest {sc}: {worst[sc]} (acc {worst['accuracy']:.2f})"
    else:
        note = f"holdout f1={analysis['f1']:.3f}"
    return {
        "error_analysis": analysis,
        "steps": [{"node": "error_analyst", "note": note, "output": analysis}],
    }


def route_after_judge(state: SimpleLoopState) -> str:
    """Router = plain code (not an LLM)."""
    action = str((state.get("verdict") or {}).get("action") or "stop")
    cycle = int(state.get("cycle") or 0)
    max_cycles = int(state.get("max_cycles") or 3)
    if action == "accept":
        return "holdout"
    if action == "revise" and cycle < max_cycles:
        return "memory"
    return "holdout"


# ---------------------------------------------------------------------------
# GRAPH WIRING
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(SimpleLoopState)

    g.add_node("warehouse_loader", node_warehouse_loader)
    g.add_node("data_scientist", node_data_scientist)
    g.add_node("feature_engineer", node_feature_engineer)
    g.add_node("hypothesis_generator", node_hypothesis)
    g.add_node("ml_engineer", node_ml_engineer)
    g.add_node("judge", node_judge)
    g.add_node("memory_writer", node_memory_writer)
    g.add_node("context_builder", node_context_builder)
    g.add_node("holdout_eval", node_holdout)
    g.add_node("error_analyst", node_error_analyst)

    g.add_edge(START, "warehouse_loader")
    g.add_edge("warehouse_loader", "data_scientist")
    g.add_edge("data_scientist", "feature_engineer")
    g.add_edge("feature_engineer", "hypothesis_generator")
    g.add_edge("hypothesis_generator", "ml_engineer")
    g.add_edge("ml_engineer", "judge")
    g.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "memory": "memory_writer",
            "holdout": "holdout_eval",
        },
    )
    g.add_edge("memory_writer", "context_builder")
    g.add_edge("context_builder", "hypothesis_generator")  # loop back
    g.add_edge("holdout_eval", "error_analyst")
    g.add_edge("error_analyst", END)
    return g.compile()


REQUIRED_NODES = frozenset(
    {
        "warehouse_loader",
        "data_scientist",
        "feature_engineer",
        "hypothesis_generator",
        "ml_engineer",
        "judge",
        "memory_writer",
        "context_builder",
        "holdout_eval",
        "error_analyst",
    }
)


def assess_graph_ready(app) -> dict:
    """Return readiness dict from a compiled LangGraph app."""
    raw = app.get_graph().nodes
    present = {n for n in raw if n not in {START, END, "__start__", "__end__"}}
    missing = sorted(REQUIRED_NODES - present)
    return {
        "ready": not missing,
        "nodes": sorted(present),
        "missing_nodes": missing,
        "error": None if not missing else f"missing nodes: {', '.join(missing)}",
    }


def get_status() -> dict:
    """Import-safe status for the gateway / UI."""
    try:
        app = build_graph()
    except NotImplementedError as exc:
        return {
            "ready": False,
            "nodes": [],
            "missing_nodes": sorted(REQUIRED_NODES),
            "error": str(exc) or "build_graph not implemented",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ready": False,
            "nodes": [],
            "missing_nodes": sorted(REQUIRED_NODES),
            "error": str(exc),
        }
    return assess_graph_ready(app)


def _initial_state(
    *,
    dataset_path: str,
    target_column: str,
    target_score: float,
    max_cycles: int,
    seed: int,
    holdout_frac: float,
    model: str | None = None,
) -> SimpleLoopState:
    state: SimpleLoopState = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "target_score": target_score,
        "max_cycles": max_cycles,
        "cycle": 0,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "best_val_f1": -1.0,
        "steps": [],
        "memory": {"history": [], "findings": [], "lessons": []},
        "brief": {},
    }
    chosen = (model or "").strip() or DEFAULT_MODEL
    state["model"] = chosen
    return state


def _finalize_payload(final: dict) -> dict:
    return {
        "dataset_path": final.get("dataset_path"),
        "target_column": final.get("target_column"),
        "cycles": final.get("cycle"),
        "best_val_f1": final.get("best_val_f1"),
        "best_spec": final.get("best_spec"),
        "holdout_metrics": final.get("holdout_metrics"),
        "source": final.get("source"),
        "feature_report": final.get("feature_report"),
        "error_analysis": final.get("error_analysis"),
        "verdict": final.get("verdict"),
        "stop_reason": final.get("stop_reason"),
        "profile": final.get("profile"),
        "memory": final.get("memory"),
        "model": final.get("model"),
        "steps": final.get("steps") or [],
    }


def run_solution_loop(
    *,
    dataset_path: str,
    target_column: str = "high_life_expectancy",
    target_score: float = 0.9,
    max_cycles: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    model: str | None = None,
) -> dict:
    """Run the whole graph to completion; return a summary dict."""
    app = build_graph()
    initial = _initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        target_score=target_score,
        max_cycles=max_cycles,
        seed=seed,
        holdout_frac=holdout_frac,
        model=model,
    )
    final = app.invoke(initial, config={"recursion_limit": 12 + max_cycles * 6})
    return _finalize_payload(final)


def stream_solution_loop(
    *,
    dataset_path: str,
    target_column: str = "high_life_expectancy",
    target_score: float = 0.9,
    max_cycles: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    model: str | None = None,
):
    """Yield UI events as each node starts/finishes (used by the gateway SSE)."""
    app = build_graph()
    initial = _initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        target_score=target_score,
        max_cycles=max_cycles,
        seed=seed,
        holdout_frac=holdout_frac,
        model=model,
    )
    yield {
        "type": "run_started",
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "model": initial.get("model"),
    }

    merged: dict = dict(initial)
    last_node: str | None = None
    try:
        for update in app.stream(
            initial, stream_mode="updates", config={"recursion_limit": 12 + max_cycles * 6}
        ):
            if not isinstance(update, dict):
                continue
            for node_name, patch in update.items():
                if last_node == "judge" and node_name in {"memory_writer", "holdout_eval"}:
                    action = str((merged.get("verdict") or {}).get("action") or "stop")
                    yield {
                        "type": "router",
                        "node": "router",
                        "status": "completed",
                        "action": action,
                        "route": "memory" if node_name == "memory_writer" else "holdout",
                    }
                    if node_name == "memory_writer":
                        yield {
                            "type": "node_completed",
                            "node": "research_memory",
                            "status": "completed",
                            "note": "history + lessons filed",
                        }

                yield {"type": "node_started", "node": node_name, "status": "running"}
                if isinstance(patch, dict):
                    steps = list(patch.get("steps") or [])
                    # Mirror Annotated[list, operator.add] — stream patches omit prior steps.
                    merged.update({k: v for k, v in patch.items() if k != "steps"})
                    if steps:
                        merged["steps"] = list(merged.get("steps") or []) + steps
                    note = steps[-1].get("note") if steps else None
                    yield {
                        "type": "node_completed",
                        "node": node_name,
                        "status": "completed",
                        "note": note,
                        "step": steps[-1] if steps else None,
                        "cycle": merged.get("cycle"),
                        "best_val_f1": merged.get("best_val_f1"),
                        "verdict": merged.get("verdict"),
                    }
                else:
                    yield {"type": "node_completed", "node": node_name, "status": "completed"}
                last_node = node_name
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": str(exc)}
        return

    yield {"type": "run_completed", "result": _finalize_payload(merged)}


if __name__ == "__main__":
    ds = REPO / "datasets" / "life_expectancy_clean.csv"
    result = run_solution_loop(dataset_path=str(ds), target_column="high_life_expectancy", max_cycles=3)
    print(json.dumps(result, indent=2, default=str))
