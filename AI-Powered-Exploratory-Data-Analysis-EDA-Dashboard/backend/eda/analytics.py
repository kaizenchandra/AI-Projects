"""Exact profiles; bounded chart samples. Never mutate source values."""

import math
import re
import warnings
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import polars as pl
from scipy import stats
from .models import AnalysisConfig

VERSION = "1.0.0"


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, (datetime,)):
        return v.isoformat()
    return v


def number(v):
    try:
        return float(v) if v is not None and not isinstance(v, bool) else np.nan
    except (ValueError, TypeError, OverflowError):
        return np.nan


def date(v):
    try:
        s = str(v)
        if not re.match(r"^\d{4}-\d{2}-\d{2}", s):
            return None
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    except ValueError:
        return None


def infer(name, values):
    present = [v for v in values if v is not None]
    if not present:
        return "text"
    if re.search(r"(^id$|_id$|uuid|email|phone|ssn)", name, re.I):
        return "identifier"
    if all(str(v).lower() in ("true", "false") for v in present):
        return "boolean"
    numeric = sum(not np.isnan(number(v)) for v in present)
    if numeric / len(present) >= 0.9:
        return "numeric"
    if sum(date(v) is not None for v in present) / len(present) >= 0.9:
        return "temporal"
    return "categorical" if len(set(map(str, present))) <= max(20, len(present) * 0.1) else "text"


def filtered(df, config):
    for f in config.filters:
        if f.column not in df.columns:
            raise ValueError("Unknown filter column.")
        s = df[f.column]
        if f.op == "is_null":
            mask = s.is_null()
        elif f.op in ("gt", "ge", "lt", "le"):
            v = number(f.value)
            if not np.isfinite(v):
                target = date(f.value)
                if target is None:
                    raise ValueError("Range filters require finite numbers or ISO dates.")
                parsed = [date(value) for value in s.to_list()]
                mask = pl.Series(
                    [
                        False
                        if d is None
                        else {"gt": d > target, "ge": d >= target, "lt": d < target, "le": d <= target}[f.op]
                        for d in parsed
                    ]
                )
            else:
                a = s.cast(pl.Float64, strict=False)
                mask = {"gt": a > v, "ge": a >= v, "lt": a < v, "le": a <= v}[f.op]
        else:
            a = s.cast(pl.String)
            mask = (
                a.str.contains(f.value or "", literal=True)
                if f.op == "contains"
                else (a == f.value if f.op == "eq" else a != f.value)
            )
        df = df.filter(mask.fill_null(False))
    return df


def numeric_profile(a, config):
    n = len(a)
    if not n:
        return {}, {}, None, np.zeros(0, dtype=bool)
    q = np.quantile(a, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], method="linear")
    centered = a - a[0]
    mean, std = np.mean(a), np.std(centered)
    mad = np.median(np.abs(a - q[3]))
    m = dict(
        min=np.min(a),
        max=np.max(a),
        mean=mean,
        median=q[3],
        range=np.ptp(a),
        sample_variance=np.var(centered, ddof=1) if n > 1 else None,
        population_variance=np.var(centered),
        sample_std=np.std(centered, ddof=1) if n > 1 else None,
        population_std=std,
        q1=q[2],
        q3=q[4],
        iqr=q[4] - q[2],
        p01=q[0],
        p05=q[1],
        p95=q[5],
        p99=q[6],
        mad=mad,
        coefficient_of_variation=std / mean if mean > 0 else None,
        skewness=stats.skew(centered, bias=False) if n >= 3 and std > 0 else None,
        excess_kurtosis=stats.kurtosis(centered, bias=False) if n >= 4 and std > 0 else None,
    )
    method, t = config.outlier_method, config.threshold
    radius = t * (q[4] - q[2]) if method == "iqr" else t * std if method == "zscore" else t * mad / 0.67448975
    lower, upper = (
        (q[2] - radius, q[4] + radius)
        if method == "iqr"
        else (
            (mean if method == "zscore" else q[3]) - radius,
            (mean if method == "zscore" else q[3]) + radius,
        )
    )
    valid = n >= 4 and (method == "iqr" or radius > 0)
    flagged = (a < lower) | (a > upper) if valid else np.zeros(n, dtype=bool)
    width = 2 * (q[4] - q[2]) / np.cbrt(n)
    ratio = np.ptp(a) / width if width > 0 else 0
    suggested = (
        min(100, max(1, int(np.ceil(ratio))))
        if np.isfinite(ratio) and ratio > 0
        else min(100, max(1, int(np.sqrt(n))))
    )
    if a.min() == a.max():
        center = a[0]
        delta = max(0.5, abs(center) * 1e-12)
        lower_edge, upper_edge = center - delta, center + delta
        edges = np.array(
            [
                lower_edge if np.isfinite(lower_edge) else center,
                upper_edge if np.isfinite(upper_edge) else center,
            ]
        )
        hist = np.array([n])
    else:
        edges = np.unique(np.linspace(a.min(), a.max(), (config.bins or suggested) + 1))
        hist, edges = np.histogram(a, bins=edges)
    density = None
    if n >= 3 and std > 0:
        # Deterministic, explicitly disclosed KDE sample bounds quadratic work.
        sample = a[np.linspace(0, n - 1, min(n, 2000), dtype=int)]
        x = np.linspace(a.min(), a.max(), 100)
        try:
            density = {"x": x, "y": stats.gaussian_kde(sample)(x), "n": len(sample)}
        except (ValueError, np.linalg.LinAlgError):
            pass
    return (
        m,
        {"counts": hist, "edges": edges, "kde": density, "box": [m["min"], q[2], q[3], q[4], m["max"]]},
        {
            "method": method,
            "threshold": t,
            "lower": lower if valid else None,
            "upper": upper if valid else None,
            "count": int(flagged.sum()),
            "percentage": 100 * flagged.sum() / n,
            "eligible": valid,
        },
        flagged,
    )


def analyze(source, config: AnalysisConfig, run_id, fingerprint):
    df = filtered(source, config)
    if any(c not in df.columns for c in [*config.overrides, *config.columns]):
        raise ValueError("Unknown analytical column.")
    profiles, numeric, insights = [], {}, []

    def insight(kind, title, cols, metric, value, recommendation, severity="warning"):
        insights.append(
            dict(
                id=f"{run_id}-{len(insights)}",
                type=kind,
                severity=severity,
                title=title,
                description=title,
                affected_columns=cols,
                evidence={"metric": metric, "value": value},
                recommendation=recommendation,
                method="deterministic_rule",
                limitations=["Exploratory finding; not a causal conclusion."],
                analysis_run_id=run_id,
            )
        )

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in df.columns:
            values = df[name].to_list()
            present = [v for v in values if v is not None]
            inferred = infer(name, values)
            if df[name].dtype.is_numeric() and inferred != "identifier":
                inferred = "numeric"
            semantic = config.overrides.get(name, inferred)
            counts = Counter(map(str, present))
            top = counts.most_common(20)
            n = len(values)
            p = dict(
                name=name,
                physical_type=str(df[name].dtype),
                inferred_type=inferred,
                override=config.overrides.get(name),
                semantic_type=semantic,
                count=n,
                non_null_count=len(present),
                null_count=n - len(present),
                missing_percentage=100 * (n - len(present)) / n if n else 0,
                distinct_count=len(counts),
                distinct_method="exact",
                uniqueness_ratio=len(counts) / len(present) if present else None,
                constant=len(counts) == 1,
                high_cardinality=len(counts) > max(20, n * 0.9),
                potential_sensitive=bool(re.search("email|phone|ssn|address|name", name, re.I)),
                top_values=[{"value": k, "count": v, "relative_frequency": v / len(present)} for k, v in top],
                other_count=len(present) - sum(v for _, v in top),
                mode=top[0][0] if top else None,
                rare_category_count=sum(v == 1 for v in counts.values()),
                statistics={},
                distribution={},
                outliers=None,
            )
            if semantic == "numeric":
                raw = np.array([number(v) for v in values])
                finite = np.isfinite(raw)
                a = raw[finite]
                numeric[name] = raw
                p["invalid_count"] = sum(v is not None and np.isnan(number(v)) for v in values)
                p["nonfinite_count"] = int(np.isinf(raw).sum())
                p["nan_count"] = sum(isinstance(v, float) and np.isnan(v) for v in values)
                p["precision_warning"] = any(abs(number(v)) > 2**53 for v in present)
                p["statistics"], p["distribution"], p["outliers"], flags = numeric_profile(a, config)
                indices = np.flatnonzero(finite)[flags]
                p["outlier_records"] = [{"row_index": int(i), "value": values[i]} for i in indices[:200]]
                p["outlier_records_total"] = len(indices)
                if p["outliers"] and p["outliers"]["count"]:
                    insight(
                        "outlier",
                        f"{name}: potential univariate anomalies",
                        [name],
                        "outlier_count",
                        p["outliers"]["count"],
                        "Investigate context; do not automatically remove records.",
                    )
                if p["statistics"].get("skewness") is not None and abs(p["statistics"]["skewness"]) > 1:
                    insight(
                        "distribution",
                        f"{name}: asymmetric distribution",
                        [name],
                        "skewness",
                        p["statistics"]["skewness"],
                        "Compare robust summaries and inspect the distribution.",
                        "info",
                    )
            elif semantic == "temporal":
                dates = [date(v) for v in present]
                valid = sorted(d for d in dates if d is not None)
                p["invalid_count"] = len(dates) - len(valid)
                periods = Counter(d.date().isoformat() for d in valid)
                p["temporal"] = {
                    "earliest": valid[0].isoformat() if valid else None,
                    "latest": valid[-1].isoformat() if valid else None,
                    "daily_counts": sorted(periods.items())[:1000],
                    "periods_total": len(periods),
                    "timezone": "UTC",
                    "cadence": "Not inferred; gaps are not assumed to be missing records.",
                }
            else:
                lengths = [len(str(v)) for v in present]
                p["statistics"] = {
                    "length_min": min(lengths) if lengths else None,
                    "length_max": max(lengths) if lengths else None,
                    "length_mean": np.mean(lengths) if lengths else None,
                    "blank_count": sum(not str(v).strip() for v in present),
                }
                if top and top[0][1] / len(present) >= 0.9:
                    insight(
                        "imbalance",
                        f"{name}: dominant value",
                        [name],
                        "dominant_fraction",
                        top[0][1] / len(present),
                        "Check whether imbalance reflects the population.",
                        "info",
                    )
            if p["missing_percentage"] >= 20:
                insight(
                    "data_quality",
                    f"{name}: substantial missingness",
                    [name],
                    "missing_percentage",
                    p["missing_percentage"],
                    "Investigate the missingness mechanism before imputation.",
                )
            profiles.append(p)
        selected = config.columns or list(numeric)[:20]
        selected = [c for c in selected if c in numeric]
        pairs = []
        for i, a in enumerate(selected):
            for b in selected[i + 1 :]:
                mask = np.isfinite(numeric[a]) & np.isfinite(numeric[b])
                x, y = numeric[a][mask], numeric[b][mask]
                r = None
                if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                    r = float(
                        (stats.pearsonr if config.correlation == "pearson" else stats.spearmanr)(x, y)[0]
                    )
                ix = np.linspace(0, len(x) - 1, min(500, len(x)), dtype=int) if len(x) else []
                pairs.append(
                    {
                        "x": a,
                        "y": b,
                        "coefficient": r,
                        "n": len(x),
                        "points": list(zip(x[ix], y[ix])),
                        "plotted_n": len(ix),
                    }
                )
                if r is not None and abs(r) >= 0.8:
                    insight(
                        "association",
                        f"{a} and {b}: strong association",
                        [a, b],
                        config.correlation,
                        r,
                        "Investigate confounders and redundant features; correlation does not imply causation.",
                        "info",
                    )
        associations = []
        cats = [
            p["name"]
            for p in profiles
            if p["semantic_type"] in ("categorical", "boolean") and 1 < p["distinct_count"] <= 30
        ][:10]
        for i, a in enumerate(cats):
            for b in cats[i + 1 :]:
                rows = df.select(a, b).drop_nulls().rows()
                av, bv = sorted({str(r[0]) for r in rows}), sorted({str(r[1]) for r in rows})
                if len(av) < 2 or len(bv) < 2:
                    continue
                table = np.zeros((len(av), len(bv)))
                for x, y in rows:
                    table[av.index(str(x)), bv.index(str(y))] += 1
                v = np.sqrt(
                    stats.chi2_contingency(table, correction=False)[0]
                    / (len(rows) * min(len(av) - 1, len(bv) - 1))
                )
                associations.append({"x": a, "y": b, "cramers_v": v, "n": len(rows)})
    duplicates = df.height - df.unique().height
    if duplicates:
        insight(
            "data_quality",
            "Duplicate records detected",
            [],
            "duplicate_rows",
            duplicates,
            "Determine whether repeated observations are legitimate.",
        )
    missing = sum(p["null_count"] for p in profiles)
    return clean(
        dict(
            run_id=run_id,
            fingerprint=fingerprint,
            algorithm_version=VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            config=config.model_dump(),
            overview={
                "rows": df.height,
                "source_rows": source.height,
                "columns": df.width,
                "missing_cells": missing,
                "missing_percentage": 100 * missing / (df.height * df.width) if df.height else 0,
                "duplicate_rows": duplicates,
                "duplicate_percentage": 100 * duplicates / df.height if df.height else 0,
                "memory_bytes": df.estimated_size(),
                "type_breakdown": dict(Counter(p["semantic_type"] for p in profiles)),
            },
            profiles=profiles,
            correlations=sorted(
                pairs,
                key=lambda p: abs(p["coefficient"]) if p["coefficient"] is not None else -1,
                reverse=True,
            ),
            associations=associations,
            insights=insights,
            preview=df.head(50).cast({c: pl.String for c in df.columns}).to_dicts(),
            missingness={
                "columns": df.columns,
                "matrix": df.head(200).select(pl.all().is_null()).rows(),
                "shown_rows": min(200, df.height),
            },
            methodology={
                "nulls": "Excluded per column; correlations use finite pairwise deletion. Empty CSV fields are null; other strings are preserved.",
                "variance": "Both population (ddof=0) and sample (ddof=1).",
                "quantiles": "Linear interpolation",
                "outliers": "Univariate; minimum four finite observations. MAD and z-score require nonzero dispersion.",
                "sampling": "Profiles and correlations exact. Scatter: evenly spaced at most 500 pairs. KDE: evenly spaced at most 2000 values. Preview first 50; missingness first 200; outliers first 200.",
                "correlation_columns": selected,
                "correlation_omitted": len(numeric) - len(selected),
                "limitations": [
                    "Float64 calculations may lose precision above 2^53.",
                    "Temporal counts limited to first 1000 days; no seasonality or cadence inferred.",
                    "Cramér’s V is uncorrected and limited to first ten categorical columns with at most 30 levels.",
                ],
            },
        )
    )
