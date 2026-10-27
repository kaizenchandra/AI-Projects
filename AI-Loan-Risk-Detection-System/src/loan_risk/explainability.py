"""Global rankings and bounded local reference-replacement sensitivity."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from loan_risk.contracts import FEATURES


def global_importance(raw_forest, calibrated, X, y, seed: int, output) -> dict:
    names = raw_forest[:-1].get_feature_names_out()
    impurity = dict(zip(names, raw_forest[-1].feature_importances_.tolist(), strict=True))
    top = sorted(impurity, key=lambda name: impurity[name], reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top[::-1], [impurity[name] for name in top[::-1]])
    ax.set_xlabel("Random Forest impurity importance (biased; transformed features)")
    fig.tight_layout()
    fig.savefig(output / "impurity_importance.png")
    plt.close(fig)
    result = permutation_importance(
        calibrated, X, y, scoring="neg_brier_score", n_repeats=5, random_state=seed, n_jobs=1
    )
    permutation = {
        feature: {"mean": float(mean), "std": float(std)}
        for feature, mean, std in zip(
            FEATURES, result.importances_mean, result.importances_std, strict=True
        )
    }
    ranked = sorted(permutation, key=lambda f: permutation[f]["mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        ranked[::-1],
        [permutation[f]["mean"] for f in ranked[::-1]],
        xerr=[permutation[f]["std"] for f in ranked[::-1]],
    )
    ax.set_xlabel("Increase in Brier loss after permutation (validation)")
    fig.tight_layout()
    fig.savefig(output / "permutation_importance.png")
    plt.close(fig)
    return {
        "impurity_transformed_features": impurity,
        "permutation_raw_features": permutation,
        "scope": "validation partition; five shuffle repeats measure permutation variability",
        "limitations": "Impurity favors continuous/high-cardinality features; correlated features "
        "can substitute for one another. Rankings have different feature spaces.",
    }


def local_explanation(model, record: pd.DataFrame, reference: dict, top_k: int = 5) -> dict:
    if len(record) != 1 or list(record.columns) != FEATURES:
        raise ValueError("Local explanation requires one ordered feature record")
    original = float(model.predict_proba(record)[0, 1])
    variants = pd.concat([record] * len(FEATURES), ignore_index=True)
    for index, feature in enumerate(FEATURES):
        variants.loc[index, feature] = reference[feature]
    replaced = model.predict_proba(variants)[:, 1]
    contributions: list[dict] = [
        {"feature": name, "probability_difference": float(original - value)}
        for name, value in zip(FEATURES, replaced, strict=True)
    ]
    contributions = sorted(
        contributions,
        key=lambda item: abs(item["probability_difference"]),
        reverse=True,
    )
    return {
        "method": "one-feature reference replacement sensitivity",
        "default_probability": original,
        "top_features": contributions[:top_k],
        "limitations": "Not SHAP, not additive or causal; replacements can be off-manifold. "
        "Not validated adverse-action reasons. Reference uses training medians/modes.",
    }
