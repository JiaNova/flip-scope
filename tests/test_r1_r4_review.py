import itertools
import numpy as np
import pytest
from experiments.core.review_r1_r4 import metrics, shapley
from experiments.core.run_r1_r3_replay import score_metrics


def test_ece_is_probability_binned_and_mass_weighted_including_endpoints():
    q = np.array([0., .05, .25, 1.])
    a = np.array([[1, 1], [0, 0], [1, 0], [0, 0]])
    # Prediction-bin contributions: |0+.05-1| + |.25-.5| + |1-0|.
    expected = (.95 + .25 + 1.) / 4
    assert metrics(q, a)["ECE10"] == pytest.approx(expected)
    assert score_metrics(q, a)["ECE10"] == pytest.approx(expected)


def test_fractional_tie_risk_and_finite_reference_auc():
    q = np.array([.1, .1, .9])
    a = np.array([[0, 0], [1, 0], [1, 1]])
    result = metrics(q, a)
    # 70% of 3 rows: keep tied low-score group plus .1 of last row.
    assert result["R_070"] == pytest.approx((.5 + .1)/2.1)
    from sklearn.metrics import roc_auc_score
    assert result["AUC_flip"] == pytest.approx(roc_auc_score(a.ravel(), np.repeat(q, 2)))


@pytest.mark.parametrize("function", [metrics, score_metrics])
def test_nonbinary_reference_is_rejected_before_integer_cast(function):
    with pytest.raises(ValueError):
        function(np.array([.1]), np.array([[.9]]))


def test_three_factor_decomposition_preserves_total_and_interactions():
    cube = np.empty((2,2,2))
    for i,j,k in itertools.product(range(2), repeat=3):
        cube[i,j,k] = 2*i + 3*j + 4*k + 6*i*j
    assert shapley(cube, "fixture") == pytest.approx([5., 6., 4.])
