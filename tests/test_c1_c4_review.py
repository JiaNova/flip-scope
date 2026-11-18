import numpy as np
import pytest
from experiments.core import run_c1_c4_v28 as work
from experiments.core.review_c1_c4_v28 import posterior_counterexample


def test_independent_bank_jackknife_matches_variance_of_stratified_mean():
    columns = np.array([[1.,2.,4.],[-2.,0.,3.]])
    k,b=columns.shape
    point=columns.mean()
    deletions=point+(columns.sum(axis=1,keepdims=True)-columns)/(b-1)/k-columns.mean(axis=1,keepdims=True)/k
    expected=np.sqrt(np.sum(columns.var(axis=1,ddof=1))/b)/k
    assert work._stratified_jackknife_se(deletions)==pytest.approx(expected)


def test_se_is_invariant_to_bankwise_column_relabeling():
    deleted=np.array([[-1.,0.,1.],[-1.,0.,1.]])
    permuted=deleted.copy();permuted[1]=permuted[1,::-1]
    assert work._stratified_jackknife_se(deleted)==pytest.approx(work._stratified_jackknife_se(permuted))
    # The legacy synchronized formula can even collapse to zero after relabeling.
    assert np.var(deleted.sum(axis=0))>0
    assert np.var(permuted.sum(axis=0))==0


def test_weighted_recut_matches_explicit_patient_duplication_with_ties():
    q=np.array([.1,.1,.3,.9]);target=np.array([0.,1.,.4,.8]);w=np.array([0,3,1,2])
    result=work._WeightedRisk(q).risks(target,w,np.array([.3,.7,1.]))
    from experiments.adult_support.run_p1_instance_transfer import compute_tie_aware_risk
    qq=np.repeat(q,w);tt=np.repeat(target,w)
    expected=[compute_tie_aware_risk(qq,tt,c) for c in (.3,.7,1.)]
    assert result==pytest.approx(expected)


def test_metric_validation_precedes_binary_integer_cast():
    with pytest.raises(ValueError): work._metric_row(np.array([.2]),np.array([[.9]]))


def test_correct_conditional_moments_do_not_make_mixture_cdf_gaussian():
    result=posterior_counterexample()
    assert result["exact_flip_probability"]==pytest.approx(.03991796461050027)
    assert result["Gaussian_moment_approximation"]-result["exact_flip_probability"]>.004
