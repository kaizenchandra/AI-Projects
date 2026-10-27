# ML concepts used

For one feature, OLS fits `y_hat = beta0 + beta1 x`, minimizing `MSE = (1/n) sum((y-y_hat)^2)`. The multiple linear model applies the same least-squares objective after preprocessing several numeric and categorical features.

The custom estimator uses `y_hat = Xw+b`, gradients `2 X.T (y_hat-y)/n` and `2 sum(y_hat-y)/n`, and batch updates. Scaling is important because one learning rate must work across columns. Its loss history is a convergence diagnostic; finite-difference checks are represented by the exact vectorized gradient behavior covered by unit tests.

MAE is in price units and is robust to individual large errors. MSE squares errors, making large misses dominate; RMSE returns to price units while retaining that emphasis. R² measures variance explained relative to a mean predictor and can be negative out of sample.

Ridge adds `alpha * ||w||²` and shrinks correlated coefficients; Lasso adds `alpha * ||w||₁` and can create sparse coefficients. Neither is guaranteed to win: selection is by development validation MAE with CV, while the test set is held out for one final comparison. Regularization increases bias but can reduce variance, illustrating the bias–variance trade-off. Coefficients are associations, not causal or individualized explanations, especially after one-hot encoding and correlated inputs.
