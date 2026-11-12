"""Pure ETL helpers for the life-expectancy pipeline.

Kept free of Airflow imports so every step can be unit-tested locally and
called from thin Airflow tasks in ``pipeline/dags/life_expectancy_pipeline.py``.

Problem statement
-----------------
Which socioeconomic conditions (income, electricity access, urbanisation,
fertility, region, income group) separate country-years where people can
expect to live 75+ years from those where they cannot?

Sources
-------
1. Our World in Data (OWID) grapher CSVs — one table per indicator,
   keyed by (country, ISO3 code, year).
2. World Bank country API — country metadata (region, income group),
   keyed by ISO3 code.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

YEAR_MIN = 2000
YEAR_MAX = 2023
LIFE_EXPECTANCY_THRESHOLD = 75.0
TARGET = "high_life_expectancy"

# OWID slug -> (raw value column, clean column name, valid range)
OWID_INDICATORS: dict[str, tuple[str, str, tuple[float, float]]] = {
    "life-expectancy": ("life_expectancy_0", "life_expectancy", (20.0, 95.0)),
    "gdp-per-capita-worldbank": ("ny_gdp_pcap_pp_kd", "gdp_per_capita", (100.0, 250_000.0)),
    "share-of-the-population-with-access-to-electricity": (
        "eg_elc_accs_zs",
        "electricity_access_pct",
        (0.0, 100.0),
    ),
    "share-of-population-urban": (
        "share__area_type_urban__data_type_estimates",
        "urban_population_pct",
        (0.0, 100.0),
    ),
    "children-per-woman-un": (
        "fertility_rate__sex_all__age_all__variant_estimates",
        "fertility_rate",
        (0.5, 10.0),
    ),
}

OWID_URL = "https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
WB_COUNTRY_URL = "https://api.worldbank.org/v2/country?format=json&per_page=400"
HTTP_HEADERS = {"User-Agent": "ids706-orchestration-assignment/1.0"}


# ---------------------------------------------------------------------------
# INGEST
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path, *, attempts: int = 3, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
            resp.raise_for_status()
            if not resp.content:
                raise ValueError(f"empty response from {url}")
            dest.write_bytes(resp.content)
            return dest
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"download failed for {url}: {last_exc}")


def ingest_owid_indicator(slug: str, raw_dir: Path, cache_dir: Path | None = None) -> dict:
    """Download one OWID indicator CSV; fall back to the committed snapshot."""
    dest = raw_dir / f"owid_{slug}.csv"
    source = "live"
    try:
        _download(OWID_URL.format(slug=slug), dest)
    except Exception:
        if cache_dir is None or not (cache_dir / f"owid_{slug}.csv").exists():
            raise
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((cache_dir / f"owid_{slug}.csv").read_bytes())
        source = "cache"
    raw_col = OWID_INDICATORS[slug][0]
    header = pd.read_csv(dest, nrows=0).columns.tolist()
    if raw_col not in header:
        raise ValueError(f"{slug}: expected column {raw_col!r}, got {header}")
    n_rows = sum(1 for _ in dest.open()) - 1
    return {"slug": slug, "path": str(dest), "rows": n_rows, "source": source}


def ingest_worldbank_countries(raw_dir: Path, cache_dir: Path | None = None) -> dict:
    """Download World Bank country metadata (JSON) and flatten it to CSV."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / "worldbank_countries.json"
    source = "live"
    try:
        _download(WB_COUNTRY_URL, json_path)
        payload = json.loads(json_path.read_text())
    except Exception:
        if cache_dir is None or not (cache_dir / "worldbank_countries.json").exists():
            raise
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_bytes((cache_dir / "worldbank_countries.json").read_bytes())
        payload = json.loads(json_path.read_text())
        source = "cache"
    records = payload[1]
    rows = [
        {
            "iso3": r.get("id"),
            "country_name": r.get("name"),
            "region_id": (r.get("region") or {}).get("id"),
            "region": (r.get("region") or {}).get("value"),
            "income_group_id": (r.get("incomeLevel") or {}).get("id"),
            "income_group": (r.get("incomeLevel") or {}).get("value"),
            "lending_type": (r.get("lendingType") or {}).get("value"),
            "capital_city": r.get("capitalCity"),
            "longitude": r.get("longitude"),
            "latitude": r.get("latitude"),
        }
        for r in records
    ]
    csv_path = raw_dir / "worldbank_countries.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"path": str(csv_path), "rows": len(rows), "source": source}


# ---------------------------------------------------------------------------
# TRANSFORM
# ---------------------------------------------------------------------------
def clean_owid_indicator(slug: str, raw_path: Path, out_dir: Path) -> dict:
    """Keep real countries (ISO3), years in range, valid values; one row per key."""
    raw_col, clean_col, (lo, hi) = OWID_INDICATORS[slug]
    df = pd.read_csv(raw_path)
    n_in = len(df)
    df = df.rename(columns={"entity": "country", "code": "iso3", raw_col: clean_col})
    df = df[["iso3", "year", clean_col]]
    # OWID aggregates (World, continents, income groups) have no code or OWID_* codes.
    df = df[df["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}")]
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    df[clean_col] = pd.to_numeric(df[clean_col], errors="coerce")
    n_invalid = int(((df[clean_col] < lo) | (df[clean_col] > hi)).sum())
    df = df[df[clean_col].between(lo, hi)]
    n_dupes = int(df.duplicated(["iso3", "year"]).sum())
    df = df.drop_duplicates(["iso3", "year"], keep="last")
    df["year"] = df["year"].astype(int)
    out = out_dir / f"clean_{clean_col}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return {
        "slug": slug,
        "column": clean_col,
        "path": str(out),
        "rows_in": n_in,
        "rows_out": int(len(df)),
        "invalid_values_dropped": n_invalid,
        "duplicates_dropped": n_dupes,
    }


def combine_indicators(clean_paths: Iterable[str], out_dir: Path) -> dict:
    """Outer-join every clean indicator on (iso3, year)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(p) for p in clean_paths]
    combined = frames[0]
    for f in frames[1:]:
        combined = combined.merge(f, on=["iso3", "year"], how="outer")
    out = out_dir / "indicators_combined.csv"
    combined.to_csv(out, index=False)
    return {"path": str(out), "rows": int(len(combined)), "columns": list(combined.columns)}


def clean_countries(raw_csv: Path, out_dir: Path) -> dict:
    """Drop World Bank aggregates (region id 'NA') and normalise text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_csv, keep_default_na=False, na_values=[""])
    n_in = len(df)
    df = df[df["region_id"].notna() & (df["region_id"] != "NA")]
    df = df[df["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}")]
    for c in ("country_name", "region", "income_group", "lending_type"):
        df[c] = df[c].astype(str).str.strip()
    df["income_group"] = df["income_group"].replace({"Not classified": None})
    df = df.drop_duplicates("iso3")
    df = df[["iso3", "country_name", "region", "income_group", "lending_type"]]
    out = out_dir / "clean_countries.csv"
    df.to_csv(out, index=False)
    return {"path": str(out), "rows_in": n_in, "rows_out": int(len(df))}


def merge_datasets(indicators_csv: Path, countries_csv: Path, out_dir: Path) -> dict:
    """Inner-join indicators with country metadata; build the modelling target."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ind = pd.read_csv(indicators_csv)
    cty = pd.read_csv(countries_csv)
    merged = ind.merge(cty, on="iso3", how="inner", validate="many_to_one")
    unmatched = sorted(set(ind["iso3"]) - set(cty["iso3"]))
    merged = merged.dropna(subset=["life_expectancy"])
    feature_cols = [
        "gdp_per_capita",
        "electricity_access_pct",
        "urban_population_pct",
        "fertility_rate",
    ]
    # Require at least 3 of the 4 socioeconomic features to keep the row.
    merged = merged[merged[feature_cols].notna().sum(axis=1) >= 3]
    merged[TARGET] = (merged["life_expectancy"] >= LIFE_EXPECTANCY_THRESHOLD).astype(int)
    merged = merged.rename(columns={"country_name": "country"})
    cols = [
        "iso3",
        "country",
        "year",
        "region",
        "income_group",
        *feature_cols,
        "life_expectancy",
        TARGET,
    ]
    merged = merged[cols].sort_values(["iso3", "year"]).reset_index(drop=True)
    out = out_dir / "country_year_features.csv"
    merged.to_csv(out, index=False)
    return {
        "path": str(out),
        "rows": int(len(merged)),
        "countries": int(merged["iso3"].nunique()),
        "unmatched_iso3": unmatched[:20],
        "target_rate": round(float(merged[TARGET].mean()), 4),
    }


def quality_checks(df: pd.DataFrame) -> dict:
    """Data-quality gate: fail loudly instead of loading bad data."""
    problems = []
    if len(df) < 1000:
        problems.append(f"completeness: only {len(df)} rows (expected >= 1000)")
    dupes = int(df.duplicated(["iso3", "year"]).sum())
    if dupes:
        problems.append(f"uniqueness: {dupes} duplicate (iso3, year) keys")
    if df["life_expectancy"].isna().any():
        problems.append("completeness: null life_expectancy")
    rate = float(df[TARGET].mean())
    if not 0.1 <= rate <= 0.9:
        problems.append(f"validity: target rate {rate:.2f} outside [0.1, 0.9]")
    null_rates = {c: round(float(df[c].isna().mean()), 4) for c in df.columns}
    if problems:
        raise ValueError("data quality gate failed: " + "; ".join(problems))
    return {"rows": int(len(df)), "duplicate_keys": dupes, "target_rate": round(rate, 4), "null_rates": null_rates}
