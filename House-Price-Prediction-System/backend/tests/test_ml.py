import numpy as np
from house_price.models.gradient_descent import GradientDescentRegressor
from house_price.models.simple import SimpleLinearRegression
from house_price.training.metrics import regression_metrics

def test_simple_ols():
    model = SimpleLinearRegression().fit([1,2,3], [2,4,6]); assert np.isclose(model.slope_, 2); assert np.isclose(model.intercept_, 0)
def test_gradient_descent_matches_linear_reference():
    rng=np.random.default_rng(2); X=rng.normal(size=(100,2)); y=3*X[:,0]-2*X[:,1]+4
    model=GradientDescentRegressor(learning_rate=.1, max_iter=5000, tol=1e-12).fit(X,y)
    assert np.allclose(model.coef_, [3,-2], atol=1e-4); assert np.isclose(model.intercept_,4,atol=1e-4)

def test_gradient_matches_finite_difference():
    X = np.array([[1.0, 2.0], [2.0, -1.0], [-1.0, 3.0]])
    y = np.array([2.0, -1.0, 4.0])
    model = GradientDescentRegressor().fit(X, y)
    model.coef_ = np.array([0.4, -0.7]); model.intercept_ = 0.2
    gradient_w, gradient_b = model._gradients(X, y)
    epsilon = 1e-6
    def loss(weights, intercept):
        return np.mean((X @ weights + intercept - y) ** 2)
    numerical_w = np.array([(loss(model.coef_ + np.eye(2)[i] * epsilon, model.intercept_) - loss(model.coef_ - np.eye(2)[i] * epsilon, model.intercept_)) / (2 * epsilon) for i in range(2)])
    numerical_b = (loss(model.coef_, model.intercept_ + epsilon) - loss(model.coef_, model.intercept_ - epsilon)) / (2 * epsilon)
    assert np.allclose(gradient_w, numerical_w, atol=1e-5)
    assert np.isclose(gradient_b, numerical_b, atol=1e-5)
def test_metrics():
    m=regression_metrics([1,2,3],[1,4,2]); assert m["mae"] == 1; assert m["mse"] == 5/3; assert m["rmse"] > 0
