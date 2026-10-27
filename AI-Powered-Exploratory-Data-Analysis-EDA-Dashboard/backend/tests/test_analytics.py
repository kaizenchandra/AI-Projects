import io
import json
import numpy as np
import polars as pl
import pytest
from scipy import stats
from eda.analytics import analyze, numeric_profile
from eda.ingestion import ingest
from eda.models import AnalysisConfig


def profile(df, **kwargs):
    return analyze(df, AnalysisConfig(**kwargs), "run", "fingerprint")


def test_descriptive_and_nulls():
    r = profile(pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, None], "y": [2.0, 4.0, 6.0, 8.0, None]}))
    s = r["profiles"][0]["statistics"]
    assert s["mean"] == 2.5
    assert s["sample_variance"] == pytest.approx(5 / 3)
    assert s["population_variance"] == 1.25
    assert s["q1"] == 1.75 and s["q3"] == 3.25
    assert s["mad"] == 1
    assert r["overview"]["missing_percentage"] == 20
    assert r["correlations"][0]["coefficient"] == pytest.approx(1)
    assert r["correlations"][0]["n"] == 4
    assert r["profiles"][0]["distribution"]["counts"] and sum(r["profiles"][0]["distribution"]["counts"]) == 4
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("method,threshold", [("iqr", 1.5), ("zscore", 2), ("mad", 3.5)])
def test_known_outlier(method, threshold):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
    _, _, o, flags = numeric_profile(a, AnalysisConfig(outlier_method=method, threshold=threshold))
    assert o["count"] == 1 and flags[-1]


def test_degenerate_nonfinite_and_precision():
    r = profile(
        pl.DataFrame(
            {
                "constant": [1.0, 1.0, 1.0, 1.0],
                "missing": [None] * 4,
                "one": [None, None, None, 2.0],
                "bad": [float("inf"), float("nan"), None, 2.0],
                "large": [2**54] * 4,
            }
        )
    )
    assert r["profiles"][0]["statistics"]["skewness"] is None
    assert r["profiles"][1]["statistics"]["length_mean"] is None
    assert r["profiles"][2]["statistics"]["sample_std"] is None
    assert r["profiles"][3]["nonfinite_count"] == 1
    assert r["profiles"][4]["precision_warning"]
    assert all(p["coefficient"] is None for p in r["correlations"])
    json.dumps(r, allow_nan=False)


def test_filter_and_identity():
    df = pl.DataFrame({"x": [1, 2, 3, 3], "user_id": ["a", "b", "c", "c"]})
    r = profile(df, filters=[{"column": "x", "op": "gt", "value": "1"}])
    assert r["overview"]["rows"] == 3 and r["overview"]["duplicate_rows"] == 1
    assert r["profiles"][1]["semantic_type"] == "identifier"
    assert df.height == 4
    assert all(i["analysis_run_id"] == "run" for i in r["insights"])
    empty = profile(df, filters=[{"column": "x", "op": "gt", "value": "10"}])
    assert empty["overview"]["rows"] == 0


def test_correlations_against_scipy():
    df = pl.DataFrame({"x": [1.0, 2.0, 2.0, 4.0, None], "y": [9.0, 5.0, 6.0, 1.0, 3.0]})
    for method in ["pearson", "spearman"]:
        r = profile(df, correlation=method)["correlations"][0]
        expected = (stats.pearsonr if method == "pearson" else stats.spearmanr)([1, 2, 2, 4], [9, 5, 6, 1])[0]
        assert r["coefficient"] == pytest.approx(expected)


def test_categorical_temporal_and_overrides():
    r = profile(
        pl.DataFrame(
            {
                "a": ["a", "b", "a", "b"],
                "b": ["x", "y", "x", "y"],
                "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            }
        )
    )
    assert r["associations"][0]["cramers_v"] == pytest.approx(1)
    assert r["profiles"][2]["semantic_type"] == "temporal"
    assert len(r["profiles"][2]["temporal"]["daily_counts"]) == 4
    r = profile(pl.DataFrame({"x": ["001", "002", "003"]}), overrides={"x": "identifier"})
    assert r["profiles"][0]["statistics"].get("mean") is None
    assert r["preview"][0]["x"] == "001"


@pytest.mark.parametrize(
    "data,name",
    [
        (b"", "x.csv"),
        (b"a,a\n1,2", "x.csv"),
        (b"a,b\n1,2,3", "x.csv"),
        (b"PKnotcsv", "x.csv"),
        (b"fake", "x.parquet"),
        (b"fake", "x.xlsx"),
        (b"abc", "x.exe"),
        (b"\xff", "x.csv"),
    ],
)
def test_invalid_ingestion(data, name):
    with pytest.raises(Exception):
        ingest(data, name)


def test_formats():
    df, _ = ingest(b"a;b\n001;hello\n002;", "x.csv", delimiter=";")
    assert df["a"][0] == "001" and df["b"][1] is None
    buffer = io.BytesIO()
    pl.DataFrame({"x": [1, 2]}).write_parquet(buffer)
    assert ingest(buffer.getvalue(), "x.parquet")[0].height == 2
    import openpyxl

    wb = openpyxl.Workbook()
    wb.active.append(["a"])
    wb.active.append([1])
    wb.create_sheet("second").append(["value"])
    wb["second"].append([42])
    buffer = io.BytesIO()
    wb.save(buffer)
    df, sheets = ingest(buffer.getvalue(), "x.xlsx", sheet="second")
    assert df["value"][0] == "42" and len(sheets) == 2
    assert ingest(b"1,2\n3,4", "x.csv", header=False)[0].height == 2


def test_iso_date_range_filter():
    df = pl.DataFrame({"when": ["2026-01-01", "2026-01-02", "bad", None]})
    r = profile(df, filters=[{"column": "when", "op": "ge", "value": "2026-01-02"}])
    assert r["overview"]["rows"] == 1
    assert r["preview"][0]["when"] == "2026-01-02"


def test_histogram_large_constant_and_nearby_values():
    for values in ([2**54] * 4, [2**54, 2**54 + 4, 2**54 + 8, 2**54 + 12]):
        _, dist, _, _ = numeric_profile(np.array(values, dtype=float), AnalysisConfig(bins=100))
        assert sum(dist["counts"]) == 4
        assert np.all(np.diff(dist["edges"]) > 0)


def test_html_and_csv_injection_rendering():
    from eda.reports import html_report

    r = profile(pl.DataFrame({"<script>alert(1)</script>": ["<img src=x onerror=alert(1)>"]}))
    report = html_report(r)
    assert "<script>alert" not in report
    assert "&lt;script&gt;" in report


def test_ingestion_limits(monkeypatch):
    import eda.ingestion as ingestion

    monkeypatch.setattr(ingestion, "MAX_ROWS", 2)
    with pytest.raises(ValueError):
        ingest(b"a\n1\n2\n3", "x.csv")


def test_report_sections_always_preserve_provenance():
    from eda.reports import html_report

    r = profile(pl.DataFrame({"x": [1, 2, 3, 4]}))
    report = html_report(r, {"overview"})
    assert "<h2>Dataset overview</h2>" in report
    assert "<h2>Quality and schema</h2>" not in report
    assert "<h2>Provenance</h2>" in report
    assert "<h2>Methods and limitations</h2>" in report
