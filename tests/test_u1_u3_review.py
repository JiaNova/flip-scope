"""Scientific checks for overlap, finite-population variance and cost direction."""
import itertools
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit
from experiments.core import run_u1_u3 as work


def test_overlapping_independent_draws_have_zero_conditional_mean_covariance():
    population = np.array([-1., 0., 2., 4.])
    draws = list(itertools.combinations(range(4), 2))
    pairs = list(itertools.product(draws, repeat=2))
    means = np.array([[population[list(a)].mean(), population[list(b)].mean()] for a, b in pairs])
    assert np.mean([len(set(a) & set(b)) for a, b in pairs]) == 1.
    assert abs(np.cov(means.T, bias=True)[0, 1]) < 1e-14


def test_two_pool_if_matches_exhaustive_linearized_srs_and_census_endpoint():
    a = np.array([[1., -2.], [1., -.4], [1., .8], [1., 2.]])
    b = np.array([[1., -1.], [1., .2], [1., 1.3]])
    ya, yb = np.array([0., 0., 1., 1.]), np.array([0., 1., 1.])
    beta = np.array([.1, .3])
    pa, pb = expit(a @ beta), expit(b @ beta)
    ua, ub = (pa - ya)[:, None] * a, (pb - yb)[:, None] * b
    k, l = 2, 1
    h = (k / len(a)) * (a.T @ ((pa * (1 - pa))[:, None] * a))
    h += (l / len(b)) * (b.T @ ((pb * (1 - pb))[:, None] * b)) + np.diag([0., 1.])
    predictions = []
    for ia, ib in itertools.product(itertools.combinations(range(len(a)), k), itertools.combinations(range(len(b)), l)):
        delta = -np.linalg.solve(h, ua[list(ia)].sum(axis=0) + ub[list(ib)].sum(axis=0))
        predictions.append(a @ delta)
    _, actual, _, _ = work._finite_population_if(sparse.csr_matrix(a), ya, sparse.csr_matrix(b), yb,
                                                beta, k, l, sparse.csr_matrix(a), sparse.csr_matrix(a))
    np.testing.assert_allclose(actual, np.var(predictions, axis=0), rtol=1e-12, atol=1e-13)
    _, census, _, _ = work._finite_population_if(sparse.csr_matrix(a), ya, sparse.csr_matrix(b), yb,
                                                beta, len(a), len(b), sparse.csr_matrix(a), sparse.csr_matrix(a))
    np.testing.assert_array_equal(census, np.zeros(len(a)))


def test_dispersion_direction_and_zero_variance_are_explicit():
    logits = np.tile([0., 1., 2.], (3, 1))  # sample variance is one
    assert work._cov(np.full(3, 2.), logits)["mean_dispersion_relation"] == "IF_OVERDISPERSED"
    assert work._cov(np.full(3, .5), logits)["mean_dispersion_relation"] == "IF_UNDERDISPERSED"
    assert work._cov(np.zeros(3), logits)["AUDIT_over_IF_mean_ratio"] is None
    assert work._cov(np.zeros(3), np.zeros_like(logits))["mean_dispersion_relation"] == "BOTH_ZERO"
    with pytest.raises(ValueError): work._cov(np.array([1., -1., 1.]), logits)


def test_cost_crossing_supports_either_slope_order_and_input_units():
    assert work._cost_crossing(1., 3., 2., 1.) == (2000., "NONNEGATIVE_CROSSING")
    assert work._cost_crossing(3., 1., 1., 2.) == (2000., "NONNEGATIVE_CROSSING")
    assert work._cost_crossing(1., 1., 2., 2.) == (None, "COINCIDENT")
    assert work._cost_crossing(1., 2., 1., 2.)[0] is None
    with pytest.raises(ValueError): work._cost_crossing(1., 2., float("nan"), 2.)


def test_impossible_within_model_draw_is_rejected():
    with pytest.raises(ValueError): work._sample(1, 3, 4)
    with pytest.raises(ValueError): work._srs_score_sum_covariance(3, 4, np.eye(2), np.zeros(2))


def test_nonfinite_or_nonbinary_inputs_do_not_become_success_metrics():
    with pytest.raises(ValueError): work._metrics(np.array([np.nan]), np.array([[0, 1]]))
    with pytest.raises(ValueError): work._metrics(np.array([.5]), np.array([[.4, 1.]]))
    with pytest.raises(ValueError): work._fit_brier(np.array([[1., np.inf]]), np.array([.5]), 0.)


def test_valid_concentrated_bootstrap_is_not_rejected_or_redrawn(tmp_path,monkeypatch):
    class FixedDraw:
        def multinomial(self,n,p):
            return np.array([4,0,0,0])
    monkeypatch.setattr(work.np.random,"default_rng",lambda seed:FixedDraw())
    monkeypatch.setattr(work,"_fit_saved",lambda *args:(np.zeros(2),{"fit_scope":"synthetic fixture"}))
    _,_,weights=work._fit_poola_bootstrap(sparse.csr_matrix(np.ones((4,2))),np.array([0,1,0,1]),
                                         0,"INT",1,tmp_path,["a","b","c","d"])
    np.testing.assert_array_equal(weights,[4,0,0,0])
