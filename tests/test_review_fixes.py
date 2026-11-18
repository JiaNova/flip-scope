"""Scientific checks for the three reporting/variance corrections."""
import itertools
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from scipy import sparse
from scipy.special import expit
from experiments.core import run_w1_w4 as work


def test_conditional_sandwich_matches_enumerated_linear_bootstrap():
    x = np.array([[1., -1.5], [1., -.2], [1., .8], [1., 1.7]])
    y = np.array([0., 1., 0., 1.])
    beta, _ = work._fit_irls(sparse.csr_matrix(x), y)
    p = expit(x @ beta)
    h = x.T @ ((p * (1 - p))[:, None] * x) + np.diag([0., 1.])
    u = (p - y)[:, None] * x
    deltas = []
    for sample in itertools.product(range(4), repeat=4):
        weights = np.bincount(sample, minlength=4)
        dbeta = -np.linalg.solve(h, u.T @ (weights - 1))
        deltas.append(x @ dbeta)
    v0, vr, _ = work._fit_if_variance(sparse.csr_matrix(x), y, beta, sparse.csr_matrix(x))
    np.testing.assert_allclose(v0, np.var(deltas, axis=0), rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(v0, vr)


def test_if_weight_derivative_matches_weighted_objective_refits():
    x = np.array([[1., -1.5], [1., -.2], [1., .8], [1., 1.7]])
    y = np.array([0., 1., 0., 1.])
    xs = sparse.csr_matrix(x)
    beta, _ = work._fit_irls(xs, y)
    p = expit(x @ beta)
    h = x.T @ ((p * (1 - p))[:, None] * x) + np.diag([0., 1.])
    direction = np.array([1., -1., 0., 0.])
    predicted = -np.linalg.solve(h, x.T @ ((p - y) * direction))
    eps = 1e-4
    plus, _ = work._fit_irls(xs, y, 1 + eps * direction)
    minus, _ = work._fit_irls(xs, y, 1 - eps * direction)
    np.testing.assert_allclose((plus - minus) / (2 * eps), predicted, rtol=1e-5, atol=1e-7)


def test_outcome_population_and_alert_denominators_are_distinct():
    result = work._outcome_metric(np.array([1., 0., 1., 0.]),
                                  np.array([1., 1., 0., 0.]), np.zeros(4), 5)
    assert result["PPV"] == .5
    assert result["D0_true_positive_retention"] == .5
    assert result["population_sensitivity"] == .2
    assert "Sensitivity" not in result
