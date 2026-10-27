import numpy as np

class SimpleLinearRegression:
    """One-variable OLS: beta1=sum((x-xbar)(y-ybar))/sum((x-xbar)^2)."""
    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float); denominator = np.sum((x-x.mean())**2)
        if denominator == 0: raise ValueError("x must vary")
        self.slope_ = np.sum((x-x.mean())*(y-y.mean())) / denominator
        self.intercept_ = y.mean() - self.slope_ * x.mean(); return self
    def predict(self, x): return self.intercept_ + self.slope_ * np.asarray(x, float)
