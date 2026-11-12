"""Unit tests for pipeline/etl.py and the new agents (run against the committed source snapshots)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline import etl

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "pipeline" / "cache"


@pytest.fixture(scope="module")
def merged(tmp_path_factory) -> pd.DataFrame:
    out = tmp_path_factory.mktemp("etl")
    cleaned = []
    for slug in etl.OWID_INDICATORS:
        info = etl.clean_owid_indicator(slug, CACHE / f"owid_{slug}.csv", out)
        assert info["rows_out"] > 0
        cleaned.append(info["path"])
    ind = etl.combine_indicators(cleaned, out)
    raw = etl.ingest_worldbank_countries(out / "raw", CACHE)  # falls back to cache offline
    cty = etl.clean_countries(Path(raw["path"]), out)
    info = etl.merge_datasets(Path(ind["path"]), Path(cty["path"]), out)
    return pd.read_csv(info["path"])


def test_owid_clean_drops_aggregates_and_out_of_range(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "entity": ["World", "Chad", "Chad", "Chad", "Peru"],
            "code": ["OWID_WRL", "TCD", "TCD", "TCD", "PER"],
            "year": [2010, 2010, 2010, 1990, 2010],
            "life_expectancy_0": [70, 50, 51, 45, 300],
        }
    ).to_csv(raw, index=False)
    info = etl.clean_owid_indicator("life-expectancy", raw, tmp_path)
    df = pd.read_csv(info["path"])
    assert df["iso3"].tolist() == ["TCD"]  # aggregate, old year, invalid value removed
    assert info["duplicates_dropped"] == 1
    assert info["invalid_values_dropped"] == 1


def test_countries_exclude_world_bank_aggregates(tmp_path):
    raw = etl.ingest_worldbank_countries(tmp_path, CACHE)
    cty = pd.read_csv(etl.clean_countries(Path(raw["path"]), tmp_path)["path"])
    assert "WLD" not in set(cty["iso3"])
    assert "USA" in set(cty["iso3"])
    assert cty["iso3"].is_unique


def test_merged_dataset_passes_quality_gate(merged):
    q = etl.quality_checks(merged)
    assert q["duplicate_keys"] == 0
    assert merged["year"].between(etl.YEAR_MIN, etl.YEAR_MAX).all()
    assert set(merged[etl.TARGET].unique()) == {0, 1}
    assert ((merged["life_expectancy"] >= 75) == (merged[etl.TARGET] == 1)).all()


def test_quality_gate_fails_on_duplicates(merged):
    with pytest.raises(ValueError, match="uniqueness"):
        etl.quality_checks(pd.concat([merged, merged.head(5)]))


def test_feature_engineer_removes_leakage(merged, tmp_path):
    from solution.langgraph.graph import node_feature_engineer

    p = tmp_path / "life_expectancy_clean.csv"
    merged.to_csv(p, index=False)
    out = node_feature_engineer({"dataset_path": str(p), "target_column": etl.TARGET, "profile": {}})
    rep = out["feature_report"]
    assert "life_expectancy" in rep["dropped_leakage"]
    assert {"country", "iso3"} <= set(rep["dropped_identifiers"])
    cols = pd.read_csv(out["dataset_path"]).columns
    assert "life_expectancy" not in cols and etl.TARGET in cols
