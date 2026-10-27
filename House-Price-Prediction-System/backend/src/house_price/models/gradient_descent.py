import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class GradientDescentRegressor(BaseEstimator, RegressorMixin):
    """Batch GD for MSE: yhat=Xw+b; the intercept is not regularized."""
    def __init__(self, learning_rate=0.03, max_iter=3000, tol=1e-8):
        self.learning_rate, self.max_iter, self.tol = learning_rate, max_iter, tol

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError("Non-finite training data")
        if self.learning_rate <= 0 or self.max_iter < 1: raise ValueError("Invalid optimizer settings")
        n, p = X.shape; self.coef_ = np.zeros(p); self.intercept_ = 0.0; self.loss_history_ = []
        previous = np.inf
        for _ in range(self.max_iter):
            error = X @ self.coef_ + self.intercept_ - y
            loss = float(np.mean(error ** 2)); self.loss_history_.append(loss)
            if not np.isfinite(loss): raise FloatingPointError("Gradient descent diverged")
            gradient_w, gradient_b = self._gradients(X, y)
            self.coef_ -= self.learning_rate * gradient_w; self.intercept_ -= self.learning_rate * gradient_b
            if abs(previous - loss) <= self.tol: break
            previous = loss
        return self

    def _gradients(self, X, y):
        error = X @ self.coef_ + self.intercept_ - y
        return 2 / len(y) * X.T @ error, 2 / len(y) * error.sum()

    def predict(self, X):
        predictions = np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
        if not np.isfinite(predictions).all(): raise FloatingPointError("Non-finite prediction")
        return predictions
