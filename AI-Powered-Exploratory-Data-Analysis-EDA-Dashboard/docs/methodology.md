# Statistical methodology

## Values, types and missingness

CSV lexical values are retained; an empty CSV field is null, while whitespace and strings such as `NA` remain original strings. XLSX formula cells are retained as formula text and never executed. Physical Parquet types are retained. Physical dtype, inferred semantic type and user override are separate. Identifier-like column names are excluded from numeric analysis unless explicitly overridden. Semantic inference is heuristic, not a transformation; review overrides before drawing conclusions.

General null counts use physical nulls. Invalid numeric conversions and infinity are separate from nulls. Native NaN is reported separately; only finite numeric values enter numerical calculations. Float64 conversion may lose integer precision above 2^53; affected columns are flagged. Extreme magnitudes can exceed Float64 arithmetic capability and cause safe job failure. All-null, empty-after-filter, singleton and constant data return unavailable metrics where mathematically undefined.

## Descriptive statistics

Means, medians, min/max/range, quartiles, percentiles and MAD use finite values. NumPy linear quantile interpolation is used. Population variance/std uses ddof=0; sample variance/std uses ddof=1 and needs two values. Skewness uses SciPy bias=False with at least three values and nonzero variance; Fisher excess kurtosis uses bias=False with at least four. Coefficient of variation is population std / mean only for positive means; users must decide whether the variable has a meaningful ratio scale. General mode uses lexical frequency; ties are resolved by first occurrence. No uniqueness estimation is used.

Categorical top-20 frequencies disclose the remaining count. Temporal ISO dates are interpreted in UTC; naive dates are explicitly assumed UTC. Daily counts are observations per date, not measured values or forecasts. Invalid temporal strings are counted. No cadence or seasonal pattern is asserted.

## Distributions and chart coverage

Histograms use Freedman–Diaconis width with a sqrt(n) fallback, capped at 100 bins; users can request 1–100 bins. Constant inputs use one bin with representable expanded edges; repeated Float64 edges are deduplicated for narrow ranges. Histograms include every finite observation. KDE uses SciPy Gaussian KDE on at most 2000 evenly spaced row positions and 100 evaluation points. This deterministic sample is not a random or representative sampling guarantee; ordered data can bias it. Mirrored KDE supplies the violin density. Box charts display quartiles with min/max whiskers, explicitly labelled (not Tukey fences).

## Relationships

Pearson and Spearman use finite pairwise deletion, at least three pairs and nonconstant variables. No p-values or multiple-comparison significance claims are produced. At most 20 numeric columns are processed unless explicitly selected within that limit. Heatmap diagonal is unavailable rather than inventing identity for constant columns. Scatter plots show at most 500 evenly spaced valid pairs and disclose the full pair count.

Cramér’s V uses uncorrected chi-square / (n × min(r−1,c−1)), square-rooted. It excludes null pairs, needs at least two observed levels on each axis, and is bounded to ten categorical columns of at most 30 levels. This exploratory uncorrected estimator can be upward-biased with small sparse samples.

## Outliers

All methods are univariate, non-destructive and require at least four finite observations. IQR fences are Q1−k×IQR and Q3+k×IQR; IQR zero may still flag deviations from an otherwise constant central mass. Z-score uses population standard deviation and requires it to be positive. Modified Z-score uses 0.67448975×(x−median)/MAD and requires positive MAD. Strictly outside fences is flagged; equality is not. The UI changes the suggested threshold when changing method. Outlier percentages use finite numerical observations as denominator. Row indices refer to the filtered dataset and are zero-based.

## Insights and reproducibility

Rules report missingness ≥20%, duplicate records, absolute skewness >1, absolute correlation ≥0.8, dominant category ≥90%, and nonzero potential outlier counts. Each finding carries the metric, value, rule method, affected columns, limitations and run identity. These are exploratory thresholds, not statistical significance tests. Reproducibility is tied to input fingerprint, parsing options, run config, algorithm version and dependency environment. Capture dependency versions in deployment artifacts when exact cross-machine reproducibility is required.
