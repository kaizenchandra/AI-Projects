"""Reproducible aggregate EDA and evaluation figures (never raw borrower records)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loan_risk.contracts import CATEGORICAL, NUMERIC, TARGET


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def eda(train: pd.DataFrame, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    numeric = train[NUMERIC]
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    report = {
        "scope": "training partition only; associations are not causal",
        "shape": list(train.shape),
        "dtypes": train.dtypes.astype(str).to_dict(),
        "missing_fraction": train.isna().mean().to_dict(),
        "class_counts": {str(k): int(v) for k, v in train[TARGET].value_counts().items()},
        "numeric_summary": numeric.describe().replace({np.nan: None}).to_dict(),
        "categorical_counts": {
            c: train[c].fillna("missing").value_counts().to_dict() for c in CATEGORICAL
        },
        "iqr_outlier_counts": ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr))
        .sum()
        .to_dict(),
        "correlations": numeric.corr().replace({np.nan: None}).to_dict(),
        "yearly_population": train.groupby(train.origination_date.dt.year)[NUMERIC + [TARGET]]
        .mean()
        .to_dict(),
    }
    # pandas/numpy integers are converted via pandas JSON for portable JSON encoding.
    report = json.loads(pd.Series(report).to_json())
    write_json(output / "eda.json", report)
    numeric.hist(bins=25, figsize=(14, 10))
    plt.tight_layout()
    plt.savefig(output / "distributions.png")
    plt.close("all")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, column in zip(
        axes.flat,
        [
            "annual_income",
            "debt_to_income",
            "credit_utilization",
            "loan_amount",
            "loan_term_months",
        ],
        strict=False,
    ):
        groups = pd.qcut(train[column], 5, duplicates="drop")
        train.groupby(groups, observed=True)[TARGET].mean().plot.bar(ax=ax)
        ax.set_title(column)
        ax.set_ylabel("Observed default fraction")
    train.repayment_history.fillna("missing").value_counts().plot.bar(ax=axes.flat[-1])
    fig.tight_layout()
    fig.savefig(output / "associations.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(numeric.corr(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(NUMERIC)), NUMERIC, rotation=90)
    ax.set_yticks(range(len(NUMERIC)), NUMERIC)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "correlations.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    yearly = train.groupby(train.origination_date.dt.year)[TARGET].agg(["mean", "size"])
    yearly["mean"].plot(marker="o", ax=axes[0], title="Training cohort default rate over time")
    axes[0].set_ylabel("Default fraction")
    axes[1].scatter(train.loan_term_months, train.loan_amount, alpha=0.15, s=8)
    axes[1].set(xlabel="Loan term (months)", ylabel="Loan amount", title="Amount versus term")
    fig.tight_layout()
    fig.savefig(output / "population_and_terms.png")
    plt.close(fig)
    (output / "EDA.md").write_text(
        "# Training-partition EDA\n\n"
        f"Rows: {len(train)}. Default fraction: {train[TARGET].mean():.3f}. "
        "See eda.json for missingness, IQR outliers, distributions, correlations and yearly means. "
        "Ranges were validated before this report; IQR flags are descriptive, not removal rules. "
        "Income, debt obligations and engineered ratios are correlated by construction. "
        "Plots describe associations, not causal effects. Test records are excluded from EDA.\n"
    )
    return report


def evaluation_plots(all_curves: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, item in all_curves.items():
        axes[0].plot(item["roc"]["fpr"], item["roc"]["tpr"], label=name)
        axes[1].plot(
            item["precision_recall"]["recall"], item["precision_recall"]["precision"], label=name
        )
        axes[2].plot(
            item["calibration"]["predicted"], item["calibration"]["observed"], "o-", label=name
        )
    axes[2].plot([0, 1], [0, 1], "k--")
    for ax, title, xlabel, ylabel in zip(
        axes,
        ["ROC", "Precision-recall", "Calibration"],
        ["False positive rate", "Recall", "Predicted probability"],
        ["True positive rate", "Precision", "Observed fraction"],
        strict=True,
    ):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "evaluation.png")
    plt.close(fig)
