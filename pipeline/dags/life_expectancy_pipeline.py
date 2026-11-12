"""Life-expectancy pipeline: OWID indicators + World Bank countries -> Postgres -> analysis.

Graph
-----
create_schema ──────────────────────────────────────────────┐
ingest_owid   [5 indicator downloads in parallel] ─┐        │
                                                    ├─ transform_owid  [5 cleans in parallel] → combine
ingest_worldbank [country metadata] ───────────────┘   transform_countries [clean]
                         merge_and_validate (join + target + quality gate)
                         load_to_postgres (idempotent replace + row-count reconcile)
analysis [parallel]:     visualize · train_baseline · export_for_agents
                         publish_summary

Schedule: weekly (Mondays 06:00 UTC), catchup off, 2 retries per task.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task, task_group
from airflow.providers.postgres.hooks.postgres import PostgresHook

REPO = Path(os.getenv("PIPELINE_REPO", "/repo"))
DATA_ROOT = REPO / "data"
CACHE_DIR = REPO / "pipeline" / "cache"
REPORT_DIR = REPO / "reports" / "pipeline"
AGENT_DATASET = REPO / "datasets" / "life_expectancy_clean.csv"
CONN_ID = "warehouse_db"
TABLE = "analytics.country_year_features"

OWID_SLUGS = [
    "life-expectancy",
    "gdp-per-capita-worldbank",
    "share-of-the-population-with-access-to-electricity",
    "share-of-population-urban",
    "children-per-woman-un",
]


def _run_dir(run_id: str) -> Path:
    safe = run_id.replace(":", "_").replace("+", "_")
    d = DATA_ROOT / "runs" / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_table():
    import pandas as pd

    hook = PostgresHook(postgres_conn_id=CONN_ID)
    with hook.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE}")
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    for c in ("gdp_per_capita", "electricity_access_pct", "urban_population_pct", "fertility_rate", "life_expectancy"):
        df[c] = pd.to_numeric(df[c])
    return df


@dag(
    dag_id="life_expectancy_pipeline",
    description="OWID indicators + World Bank country metadata -> Postgres -> analysis",
    schedule="0 6 * * 1",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "ids706", "retries": 2, "retry_delay": timedelta(seconds=30)},
    tags=["ids706", "week11", "etl"],
)
def life_expectancy_pipeline():
    @task
    def create_schema() -> str:
        hook = PostgresHook(postgres_conn_id=CONN_ID)
        hook.run(
            f"""
            CREATE SCHEMA IF NOT EXISTS analytics;
            CREATE TABLE IF NOT EXISTS {TABLE} (
                iso3 CHAR(3) NOT NULL,
                country TEXT NOT NULL,
                year INT NOT NULL,
                region TEXT,
                income_group TEXT,
                gdp_per_capita DOUBLE PRECISION,
                electricity_access_pct DOUBLE PRECISION,
                urban_population_pct DOUBLE PRECISION,
                fertility_rate DOUBLE PRECISION,
                life_expectancy DOUBLE PRECISION NOT NULL,
                high_life_expectancy SMALLINT NOT NULL,
                loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (iso3, year)
            );
            CREATE TABLE IF NOT EXISTS analytics.pipeline_runs (
                run_id TEXT PRIMARY KEY,
                rows_loaded INT,
                countries INT,
                target_rate DOUBLE PRECISION,
                finished_at TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        return TABLE

    # ---------------- INGEST (both sources in parallel) ----------------
    @task_group(group_id="ingest_owid")
    def ingest_owid():
        @task(task_id="download_indicator")
        def download(slug: str) -> dict:
            from pipeline import etl

            from airflow.operators.python import get_current_context

            rid = get_current_context()["run_id"]
            return etl.ingest_owid_indicator(slug, _run_dir(rid) / "raw", CACHE_DIR)

        # Dynamic task mapping: one parallel task instance per indicator.
        return download.expand(slug=OWID_SLUGS)

    @task_group(group_id="ingest_worldbank")
    def ingest_worldbank():
        @task
        def download_countries() -> dict:
            from pipeline import etl

            from airflow.operators.python import get_current_context

            rid = get_current_context()["run_id"]
            return etl.ingest_worldbank_countries(_run_dir(rid) / "raw", CACHE_DIR)

        return download_countries()

    # ---------------- TRANSFORM ----------------
    @task_group(group_id="transform_owid")
    def transform_owid(raw_infos):
        @task(task_id="clean_indicator")
        def clean(info: dict) -> dict:
            from pipeline import etl

            from airflow.operators.python import get_current_context

            rid = get_current_context()["run_id"]
            return etl.clean_owid_indicator(info["slug"], Path(info["path"]), _run_dir(rid) / "clean")

        @task
        def combine(cleaned: list[dict]) -> dict:
            from pipeline import etl

            from airflow.operators.python import get_current_context

            rid = get_current_context()["run_id"]
            return etl.combine_indicators([c["path"] for c in cleaned], _run_dir(rid) / "clean")

        return combine(clean.expand(info=raw_infos))

    @task_group(group_id="transform_countries")
    def transform_countries(raw_info):
        @task
        def clean_countries(info: dict) -> dict:
            from pipeline import etl

            from airflow.operators.python import get_current_context

            rid = get_current_context()["run_id"]
            return etl.clean_countries(Path(info["path"]), _run_dir(rid) / "clean")

        return clean_countries(raw_info)

    @task
    def merge_and_validate(ind: dict, cty: dict) -> dict:
        import pandas as pd

        from pipeline import etl

        from airflow.operators.python import get_current_context

        rid = get_current_context()["run_id"]
        info = etl.merge_datasets(Path(ind["path"]), Path(cty["path"]), _run_dir(rid) / "final")
        info["quality"] = etl.quality_checks(pd.read_csv(info["path"]))
        return info

    @task
    def load_to_postgres(merged: dict, _table: str) -> dict:
        import pandas as pd
        from psycopg2.extras import execute_values

        from airflow.operators.python import get_current_context

        df = pd.read_csv(merged["path"])
        df = df.astype(object).where(pd.notna(df), None)
        cols = list(df.columns)
        hook = PostgresHook(postgres_conn_id=CONN_ID)
        with hook.get_conn() as conn, conn.cursor() as cur:
            # Idempotent full refresh inside one transaction.
            cur.execute(f"TRUNCATE {TABLE}")
            execute_values(
                cur,
                f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES %s",
                [tuple(r) for r in df.itertuples(index=False)],
                page_size=1000,
            )
            cur.execute(f"SELECT count(*) FROM {TABLE}")
            loaded = cur.fetchone()[0]
            if loaded != len(df):
                raise ValueError(f"reconcile failed: sent {len(df)} rows, table has {loaded}")
            cur.execute(
                "INSERT INTO analytics.pipeline_runs (run_id, rows_loaded, countries, target_rate) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (run_id) DO UPDATE SET rows_loaded = EXCLUDED.rows_loaded",
                (get_current_context()["run_id"], loaded, merged["countries"], merged["target_rate"]),
            )
            conn.commit()
        return {"rows_loaded": int(loaded), "table": TABLE}

    # ---------------- ANALYSIS (parallel, reads from Postgres) ----------------
    @task_group(group_id="analysis")
    def analysis(load_info):
        @task
        def visualize(_load: dict) -> list[str]:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            df = _read_table()
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            paths = []

            fig, ax = plt.subplots(figsize=(8, 5))
            for region, g in df.groupby("region"):
                ax.scatter(np.log10(g["gdp_per_capita"]), g["life_expectancy"], s=6, alpha=0.5, label=region)
            ax.axhline(75, color="black", linestyle="--", linewidth=1, label="75-year threshold")
            ax.set_xlabel("log10 GDP per capita (PPP, constant $)")
            ax.set_ylabel("Life expectancy at birth (years)")
            ax.set_title("Income vs life expectancy, 2000-2023")
            ax.legend(fontsize=7, markerscale=2)
            fig.tight_layout()
            p = REPORT_DIR / "gdp_vs_life_expectancy.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths.append(str(p))

            trend = df.groupby(["year", "region"])["life_expectancy"].mean().unstack()
            fig, ax = plt.subplots(figsize=(8, 5))
            trend.plot(ax=ax)
            ax.set_ylabel("Mean life expectancy (years)")
            ax.set_title("Life expectancy by World Bank region")
            ax.legend(fontsize=7)
            fig.tight_layout()
            p = REPORT_DIR / "life_expectancy_by_region.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths.append(str(p))
            return paths

        @task
        def train_baseline(_load: dict) -> dict:
            import numpy as np
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            from sklearn.model_selection import GroupShuffleSplit
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            df = _read_table()
            num = ["gdp_per_capita", "electricity_access_pct", "urban_population_pct", "fertility_rate"]
            cat = ["region", "income_group"]
            X, y = df[num + cat], df["high_life_expectancy"]
            # Split by country so the test set has unseen countries (no country leakage).
            tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=42).split(X, y, df["iso3"]))
            pre = ColumnTransformer(
                [
                    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
                    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
                ]
            )
            X = X.assign(gdp_per_capita=np.log10(X["gdp_per_capita"].clip(lower=1)))
            model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
            model.fit(X.iloc[tr], y.iloc[tr])
            proba = model.predict_proba(X.iloc[te])[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = {
                "model": "logistic_regression (log GDP, scaled numerics, one-hot region/income)",
                "split": "GroupShuffleSplit by country, 25% countries held out",
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "accuracy": round(float(accuracy_score(y.iloc[te], pred)), 4),
                "f1": round(float(f1_score(y.iloc[te], pred)), 4),
                "roc_auc": round(float(roc_auc_score(y.iloc[te], proba)), 4),
            }
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
            return metrics

        @task
        def export_for_agents(_load: dict) -> dict:
            df = _read_table().drop(columns=["loaded_at"])
            AGENT_DATASET.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(AGENT_DATASET, index=False)
            return {"path": str(AGENT_DATASET), "rows": int(len(df))}

        return [visualize(load_info), train_baseline(load_info), export_for_agents(load_info)]

    @task
    def publish_summary(merged: dict, load_info: dict, outputs: list) -> dict:
        from airflow.operators.python import get_current_context

        summary = {
            "run_id": get_current_context()["run_id"],
            "rows_loaded": load_info["rows_loaded"],
            "countries": merged["countries"],
            "target_rate": merged["target_rate"],
            "quality": {k: v for k, v in merged["quality"].items() if k != "null_rates"},
            "figures": outputs[0],
            "baseline_metrics": outputs[1],
            "agent_dataset": outputs[2],
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "last_run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    table = create_schema()
    owid_raw = ingest_owid()
    wb_raw = ingest_worldbank()
    indicators = transform_owid(owid_raw)
    countries = transform_countries(wb_raw)
    merged = merge_and_validate(indicators, countries)
    loaded = load_to_postgres(merged, table)
    outputs = analysis(loaded)
    publish_summary(merged, loaded, outputs)


life_expectancy_pipeline()
