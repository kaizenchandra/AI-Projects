"""Run explicitly; measurements are printed, never used as dashboard data."""

import json
import resource
import sys
import time
import numpy as np
import polars as pl
from eda.analytics import analyze
from eda.models import AnalysisConfig

rng = np.random.default_rng(42)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
x = rng.normal(size=n)
df = pl.DataFrame(
    {
        "measurement": x,
        "response": 2 * x + rng.normal(size=n),
        "category": rng.choice(["A", "B", "C"], n),
        "observed_at": ["2026-01-01"] * n,
    }
)
start = time.perf_counter()
r = analyze(df, AnalysisConfig(), "benchmark", "seed-42")
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(
    json.dumps(
        {
            "rows": n,
            "columns": df.width,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "process_peak_rss_mib": round(rss / (1024**2 if sys.platform == "darwin" else 1024), 2),
            "result_bytes": len(json.dumps(r)),
        }
    )
)
