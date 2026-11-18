"""Check covariance approximation at its mathematically exact endpoints."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from scipy import sparse
from experiments.core.run_if_approx_pilot import factors, variance


def test_full_rank_projection_matches_dense_covariance_including_singular_case():
    x = np.array([[1., 0., -2.], [0., 3., 0.], [1., 1., 1.]])
    for b in (np.array([[1., 2., 0.], [0., 1., 1.], [2., 0., 1.]]), np.ones((3, 1))):
        sigma = b @ b.T
        residual, low = factors(sigma, len(sigma))
        expected = np.diag(x @ sigma @ x.T)
        np.testing.assert_allclose(variance(sparse.csr_matrix(x), residual, low), expected, atol=1e-12)


def test_diagonal_projection_is_exact_for_diagonal_covariance():
    sigma = np.diag([.2, 0., 3.])
    x = np.array([[1., 4., -2.], [0., 3., 0.]])
    residual, low = factors(sigma, 0)
    np.testing.assert_allclose(variance(sparse.csr_matrix(x), residual, low), np.diag(x @ sigma @ x.T))
