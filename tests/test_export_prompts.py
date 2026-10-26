"""prompts.csv lists every prompt exactly once and matches the registry."""
import csv
from collections import Counter

import pytest

from paths import DATA_ROOT
from pipeline.export_prompts import COLUMNS, prompt_rows


def test_53_distinct_prompts():
    rows = prompt_rows()
    assert len(rows) == 53
    assert len({r["prompt"] for r in rows}) == 53
    assert Counter(r["condition"] for r in rows) == {
        "Base": 1, "D1": 1, "D2": 1, "S1": 10, "S2": 10, "S3": 10, "S4": 10, "S5": 10}


def test_archived_csv_matches_registry():
    path = DATA_ROOT / "prompts.csv"
    if not path.exists():
        pytest.skip(f"no archive at {DATA_ROOT}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        assert list(reader) == prompt_rows()
