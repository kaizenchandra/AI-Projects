from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_diagnostics(y_true, predictions, selected_name, loss_history, output_dir: Path):
    """Write offline plots for the selected holdout predictions and GD convergence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    residuals = actual - predicted

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(actual, predicted, alpha=0.7)
    bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axis.plot(bounds, bounds, "--", color="black")
    axis.set(title=f"Actual vs predicted · {selected_name}", xlabel="Actual price", ylabel="Predicted price")
    figure.tight_layout(); figure.savefig(output_dir / "actual_vs_predicted.png", dpi=140); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(predicted, residuals, alpha=0.7)
    axis.axhline(0, linestyle="--", color="black")
    axis.set(title="Residuals versus predictions", xlabel="Predicted price", ylabel="Residual")
    figure.tight_layout(); figure.savefig(output_dir / "residuals_vs_predictions.png", dpi=140); plt.close(figure)

    if loss_history:
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(loss_history)
        axis.set(title="Gradient descent convergence", xlabel="Iteration", ylabel="MSE loss")
        figure.tight_layout(); figure.savefig(output_dir / "gradient_descent_loss.png", dpi=140); plt.close(figure)
