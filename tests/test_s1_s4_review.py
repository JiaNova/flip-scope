"""Check score-coordinate matching when replaying a saved calibration map."""
import json
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from experiments.core import run_s3_s4 as work


def test_legacy_operator_map_is_not_applied_to_new_score_coordinates(tmp_path, monkeypatch):
    current = {"center": 2., "scale": 1., "coefficients": [-1., -2.]}
    legacy = {"center": 2. / np.sqrt(2), "scale": 1. / np.sqrt(2), "coefficients": [-1., -2.]}
    saved = {name: current for name in ["margin_logistic", "fusion_mean_only", "qMV", "D0_center_IF"]}
    saved["operator_center_IF"] = legacy
    source = tmp_path / "w2b_fulltrain/bank0/calibrators.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(saved))
    monkeypatch.setattr(work, "RUNS", tmp_path)
    def reject_fit(*args, **kwargs):
        raise AssertionError("Map correction must not refit")
    monkeypatch.setattr(work.base, "_cal_stability", reject_fit)
    score = np.array([.5, 1., 3.])
    raw = {key: score for key in ["margin", "fusion", "qmv", "condition_if"]}
    probabilities, maps = work._get_calref_maps(0, raw, np.zeros(3), raw, raw, tmp_path / "out")
    expected = work.base._apply_cal_map(current, score)
    equivalent_legacy = work.base._apply_cal_map(legacy, score / np.sqrt(2))
    np.testing.assert_allclose(expected, equivalent_legacy, atol=1e-15)
    np.testing.assert_allclose(probabilities["condition_if"], expected, atol=1e-15)
    assert maps["operator_center_IF"] == current
