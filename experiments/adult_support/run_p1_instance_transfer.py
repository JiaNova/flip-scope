#!/usr/bin/env python3
"""One-shot M3-R P1 instance-transfer analysis on public UCI Adult.

No research model, mapping, or KDE is fitted here.  Existing D0/INT/AUDIT
boosters and frozen maps are loaded read-only.  The first-stage gate is
computed on the old 2,938-group Adult test alert queue; the 9,141-group
calibration-candidate remainder is not parsed or scored unless that gate
passes.  Income fields are never parsed or made available to metrics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import lightgbm as lgb
import numpy as np
from scipy.special import expit, ndtr
from sklearn.metrics import roc_auc_score

from inflight import public_adult as m3r


ROOT = m3r.ROOT
CHECKPOINT = ROOT / "runs/adult_pilot"
OLD_REPORT = ROOT / "reports/adult_pilot"
DATA_DIR = ROOT / "data/public/adult"
OUT_DIR = ROOT / "reports/adult_support/P1_EXECUTION_20261110"

SIGMA_G_SQ = 0.0441999496  # exact v2.1 analysis constant, no fit
EPS = 1e-12
ALERT_THRESHOLD = 0.5
RISKS = (0.50, 0.70, 0.80, 0.85, 0.90, 1.00)
BASELINES = ("margin_logistic", "int_freq")
GATE_CANDIDATES = ("q_V", "q_MV_shrunk", "q_MV_raw")
METHOD_ORDER = (
    "margin_logistic", "q_global_1", "q_sqrt2", "raw_tweedie", "pav",
    "int_freq", "q_V", "q_MV_shrunk", "q_MV_raw", "q_V_unshrunk",
    "q_MV_unshrunk",
)
MODEL_IDS = ("D0",) + tuple(f"INT-{i:02d}" for i in range(1, 7)) + tuple(
    f"AUDIT-{i:02d}" for i in range(1, 12)
)


class P1Error(RuntimeError):
    pass


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False) + "\n").encode()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise P1Error(f"required saved artifact is missing or not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict, str]:
    digest = _sha(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise P1Error(f"invalid saved JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise P1Error(f"expected JSON object at {path}")
    return value, digest


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p <= 0.0) | (p >= 1.0)):
        raise P1Error("saved model probabilities must be finite and strictly inside (0,1)")
    return np.log(p) - np.log1p(-p)


def signed_gaussian_flip_risk(z0: float, mean: float, variance: float) -> float:
    """Signed threshold-crossing probability, including the W=0 limit."""
    if not all(math.isfinite(x) for x in (z0, mean, variance)) or variance < 0:
        raise ValueError("finite z0/mean and nonnegative variance are required")
    s0 = 2 * int(z0 >= 0.0) - 1
    if variance == 0.0:
        sr = 2 * int(mean >= 0.0) - 1
        return float(sr != s0)
    return float(ndtr(-(s0 * mean) / math.sqrt(variance)))


def compute_tie_aware_risk(scores: np.ndarray, targets: np.ndarray, c: float) -> float:
    """Retain exactly c*n mass, splitting the threshold score class evenly."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if (scores.ndim != 1 or targets.shape != scores.shape or not scores.size or
            not np.isfinite(scores).all() or not np.isfinite(targets).all() or
            np.any((targets < 0.0) | (targets > 1.0)) or not (0.0 < c <= 1.0)):
        raise ValueError("risk requires aligned finite scores/targets and 0<c<=1")
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    target_sums = np.bincount(inverse, weights=targets, minlength=values.size)
    cumulative_n = np.cumsum(counts, dtype=np.int64)
    cumulative_t = np.cumsum(target_sums, dtype=np.float64)
    mass = c * scores.size
    boundary = int(np.searchsorted(cumulative_n, mass, side="left"))
    before_n = 0 if boundary == 0 else int(cumulative_n[boundary - 1])
    before_t = 0.0 if boundary == 0 else float(cumulative_t[boundary - 1])
    within = (mass - before_n) / float(counts[boundary])
    selected_target = before_t + within * float(target_sums[boundary])
    return float(selected_target / mass)


def _all_risks(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    fixed = {f"R_{int(c*100):03d}": compute_tie_aware_risk(scores, targets, c)
             for c in RISKS}
    curve_100 = [compute_tie_aware_risk(scores, targets, k / 100.0)
                 for k in range(1, 101)]
    curve_tail = [compute_tie_aware_risk(scores, targets, k / 100.0)
                  for k in range(70, 101)]
    fixed["AURC100"] = float(np.mean(curve_100))
    fixed["AURC_70_100"] = float(np.mean(curve_tail))
    fixed["risk_curve_1_to_100"] = curve_100
    fixed["risk_curve_70_to_100"] = curve_tail
    return fixed


def _pair_order_diagnostics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Count reversals/ties only among pairs strictly ordered by reference."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (reference.ndim != 1 or candidate.shape != reference.shape or not reference.size or
            not np.isfinite(reference).all() or not np.isfinite(candidate).all()):
        raise ValueError("pair diagnostics require aligned finite one-dimensional scores")
    _, ref_inv, ref_counts = np.unique(reference, return_inverse=True, return_counts=True)
    _, cand_rank = np.unique(candidate, return_inverse=True)
    cand_rank = cand_rank.astype(np.int64) + 1  # Fenwick indices are one-based.
    order = np.argsort(ref_inv, kind="mergesort")
    tree = np.zeros(int(cand_rank.max()) + 1, dtype=np.int64)

    def prefix(index: int) -> int:
        total = 0
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total

    def add(index: int, amount: int = 1) -> None:
        while index < len(tree):
            tree[index] += amount
            index += index & -index

    discordant = tied = seen = reference_strict = 0
    cursor = 0
    for group_size in ref_counts:
        end = cursor + int(group_size)
        ranks = cand_rank[order[cursor:end]]
        reference_strict += seen * int(group_size)
        for rank in ranks:
            less_equal = prefix(int(rank))
            less = prefix(int(rank) - 1)
            tied += less_equal - less
            discordant += seen - less_equal
        for rank in ranks:
            add(int(rank))
        seen += int(group_size)
        cursor = end
    n = reference.size
    ref_ties = sum(int(k) * (int(k) - 1) // 2 for k in ref_counts)
    _, tie_counts = np.unique(candidate, return_counts=True)
    return {
        "reference_strict_pair_count": int(reference_strict),
        "reference_tied_pair_count": int(ref_ties),
        "discordant_pairs_vs_margin_logistic": int(discordant),
        "tied_candidate_pairs_on_reference_strict_pairs": int(tied),
        "max_exact_score_tie_class_count": int(tie_counts.max()),
        "max_exact_score_tie_class_fraction": float(tie_counts.max() / n),
    }


def _ece10(scores: np.ndarray, targets: np.ndarray) -> float:
    bins = np.minimum((np.asarray(scores) * 10.0).astype(np.int64), 9)
    n = scores.size
    total = 0.0
    for b in range(10):
        mask = bins == b
        if np.any(mask):
            total += float(mask.mean()) * abs(float(scores[mask].mean()) -
                                              float(targets[mask].mean()))
    return float(total)


def _candidate_scores(z0: np.ndarray, z_int: np.ndarray,
                       audit_flips: np.ndarray, saved_mappings: dict,
                       grid: np.ndarray, raw_grid_risk: np.ndarray) -> dict[str, np.ndarray]:
    n = z0.size
    if z_int.shape != (n, 6) or audit_flips.shape != (n, 11):
        raise P1Error("P1 role matrix dimensions must be (n,6 INT) and (n,11 AUDIT)")
    z_bar = np.mean(z_int, axis=1)
    s_sq = np.var(z_int, axis=1, ddof=1)
    v_tilde = 0.5 * s_sq + 0.5 * SIGMA_G_SQ
    m_tilde = 0.5 * z0 + 0.5 * z_bar
    k_flips = np.sum(z_int < 0.0, axis=1)
    raw_tweedie = np.interp(z0, grid, raw_grid_risk,
                            left=raw_grid_risk[0], right=raw_grid_risk[-1])
    scores = {
        "margin_logistic": m3r.apply_mapping(saved_mappings["logistic_brier_margin"], z0),
        "q_global_1": ndtr(-z0 / math.sqrt(SIGMA_G_SQ)),
        "q_sqrt2": ndtr(-z0 / math.sqrt(2.0 * SIGMA_G_SQ)),
        "raw_tweedie": raw_tweedie,
        "pav": m3r.apply_mapping(saved_mappings["isotonic_tweedie_score"], raw_tweedie),
        "int_freq": (k_flips + 0.5) / 7.0,
        "q_V": ndtr(-z0 / np.sqrt(v_tilde)),
        "q_MV_shrunk": ndtr(-m_tilde / np.sqrt(v_tilde)),
        "q_MV_raw": ndtr(-z_bar / np.sqrt(v_tilde)),
        "q_V_unshrunk": ndtr(-z0 / np.sqrt(np.maximum(s_sq, EPS))),
        "q_MV_unshrunk": ndtr(-z_bar / np.sqrt(np.maximum(s_sq, EPS))),
    }
    if tuple(scores) != METHOD_ORDER:
        raise AssertionError("P1 candidate set/order differs from v2.1 specification")
    for method, q in scores.items():
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (n,) or not np.isfinite(q).all() or np.any((q < 0.0) | (q > 1.0)):
            raise P1Error(f"invalid candidate probabilities for {method}")
        scores[method] = q
    return scores


def _method_metrics(scores: dict[str, np.ndarray], audit_flips: np.ndarray) -> dict:
    t = audit_flips.mean(axis=1).astype(np.float64)
    y_flat = np.asarray(audit_flips, dtype=np.int8).reshape(-1)
    results = {}
    margin = scores["margin_logistic"]
    for method in METHOD_ORDER:
        q = scores[method]
        risk = _all_risks(q, t)
        try:
            auc = float(roc_auc_score(y_flat, np.repeat(q, audit_flips.shape[1])))
        except ValueError:
            auc = None
        finite_brier = float(np.mean((q[:, None] - audit_flips.astype(np.float64)) ** 2))
        results[method] = {
            "n_alert_groups": int(q.size),
            **risk,
            "AUC": auc,
            "Brier_finite_11": finite_brier,
            "ECE10": _ece10(q, t),
            "paired_delta_Brier_vs_margin_logistic": float(np.mean(
                q*q - margin*margin - 2.0*(q-margin)*t)),
            "paired_delta_Brier_vs_q_sqrt2": float(np.mean(
                q*q - scores["q_sqrt2"]**2 - 2.0*(q-scores["q_sqrt2"])*t)),
            "mean_audit_flip_rate": float(t.mean()),
            **_pair_order_diagnostics(margin, q),
        }
    k0_count = int(np.count_nonzero(scores["int_freq"] == (0.5 / 7.0)))
    results["_shared"] = {
        "n_alert_groups": int(t.size),
        "audit_reference_count": int(audit_flips.shape[1]),
        "mean_audit_flip_rate": float(t.mean()),
        "int_freq_k0_group_count": k0_count,
        "int_freq_k0_group_fraction": float(k0_count / t.size),
        "int_freq_k0_score": float(0.5 / 7.0),
    }
    return results


def _metric_csv(metrics: Mapping[str, Mapping]) -> bytes:
    risk_keys = [f"R_{int(c*100):03d}" for c in RISKS]
    fields = ["method", "n_alert_groups", *risk_keys, "AURC100", "AURC_70_100",
              "AUC", "Brier_finite_11", "paired_delta_Brier_vs_margin_logistic",
              "paired_delta_Brier_vs_q_sqrt2", "ECE10", "mean_audit_flip_rate",
              "reference_strict_pair_count", "reference_tied_pair_count",
              "discordant_pairs_vs_margin_logistic",
              "tied_candidate_pairs_on_reference_strict_pairs",
              "max_exact_score_tie_class_count", "max_exact_score_tie_class_fraction"]
    stream = __import__("io").StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for method in METHOD_ORDER:
        row = {key: metrics[method].get(key) for key in fields}
        row["method"] = method
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _instances_csv(group_ids: list[str], z0: np.ndarray, z_int: np.ndarray,
                   targets: np.ndarray, scores: Mapping[str, np.ndarray]) -> bytes:
    fields = ["feature_group_sha256", "z0", "audit_flip_rate"]
    fields.extend(f"z_INT_{i:02d}" for i in range(1, 7))
    fields.extend(METHOD_ORDER)
    stream = __import__("io").StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    t = targets.mean(axis=1)
    for i, group_id in enumerate(group_ids):
        row = {"feature_group_sha256": group_id, "z0": repr(float(z0[i])),
               "audit_flip_rate": repr(float(t[i]))}
        row.update({f"z_INT_{j+1:02d}": repr(float(z_int[i, j])) for j in range(6)})
        row.update({method: repr(float(scores[method][i])) for method in METHOD_ORDER})
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _validate_saved_inputs() -> dict:
    """Check only the 18 used models and their exact prior-run bindings."""
    old_result, old_result_sha = _read_json(OLD_REPORT / "result.json")
    old_checkpoint, old_checkpoint_sha = _read_json(OLD_REPORT / "checkpoint_summary.json")
    prep, prep_sha = _read_json(DATA_DIR / "preparation_manifest.json")
    saved_gradient, saved_gradient_sha = _read_json(
        ROOT / "reports/adult_support/saved_gradient_audit.json")
    config, config_sha = m3r.load_config()
    if old_result.get("common_evaluation_queue_group_count") != 2938:
        raise P1Error("saved M3-R dev queue is not the required 2,938 groups")
    if prep.get("status") != "PUBLIC_ADULT_PREPARATION_VALIDATED":
        raise P1Error("public Adult preparation manifest is not in its validated state")
    if old_result.get("bindings", {}).get("preparation_manifest_sha256") != prep_sha:
        raise P1Error("M3-R result and public Adult preparation manifest digest disagree")
    if old_result.get("bindings", {}).get("config_sha256") != config_sha:
        raise P1Error("M3-R result and current fixed Adult config digest disagree")
    units = {u.get("artifact"): u for u in old_checkpoint.get("ledger", {}).get("unit_costs", [])}
    hashes = {}
    for model_id in MODEL_IDS:
        rel = f"models/{model_id}"
        expected = units.get(rel, {}).get("model_sha256")
        model_path = CHECKPOINT / rel / "booster.txt"
        actual = _sha(model_path)
        if not expected or expected != actual:
            raise P1Error(f"saved model hash mismatch or absent run binding: {model_id}")
        manifest, manifest_sha = _read_json(CHECKPOINT / rel / "model_manifest.json")
        if manifest.get("model_id") != model_id:
            raise P1Error(f"saved model role identity mismatch: {model_id}")
        hashes[rel] = {"sha256": actual, "manifest_sha256": manifest_sha}
    saved_mappings = {}
    for map_id in ("logistic_brier_margin", "isotonic_tweedie_score"):
        rel = f"mappings/{map_id}"
        mapping, digest = _read_json(CHECKPOINT / rel / "mapping.json")
        expected = units.get(rel, {}).get("mapping_sha256")
        if expected != digest or mapping.get("method_id") != map_id:
            raise P1Error(f"saved mapping is not the previously recorded artifact: {map_id}")
        saved_mappings[map_id] = mapping
        hashes[rel] = {"sha256": digest}
    kde, kde_sha = _read_json(CHECKPOINT / "kde/KDE/kde.json")
    expected_kde = units.get("kde/KDE", {}).get("kde_sha256")
    if kde_sha != expected_kde or kde.get("status") != "ESTIMABLE":
        raise P1Error("saved frozen Tweedie KDE cache missing or hash mismatch")
    grid = np.asarray(kde.get("grid"), dtype=np.float64)
    raw_risk = np.asarray(kde.get("grid_probability"), dtype=np.float64)
    if (grid.ndim != 1 or raw_risk.shape != grid.shape or grid.size != 2001 or
            not np.isfinite(grid).all() or not np.isfinite(raw_risk).all()):
        raise P1Error("saved Tweedie score grid has an invalid shape/value")
    hashes["kde/KDE"] = {"sha256": kde_sha}
    scale, scale_sha = _read_json(CHECKPOINT / "scale/SCALE/scale.json")
    if not math.isclose(float(scale.get("value")), 0.04419994963463374,
                        rel_tol=0.0, abs_tol=1e-15):
        raise P1Error("the saved scale no longer matches the recorded P1 constant")
    hashes["scale/SCALE"] = {"sha256": scale_sha}
    result = {
        "old_result_sha256": old_result_sha,
        "old_checkpoint_summary_sha256": old_checkpoint_sha,
        "saved_gradient_audit_sha256": saved_gradient_sha,
        "preparation_manifest_sha256": prep_sha,
        "config_sha256": config_sha,
        "config_id": config.get("config_id"),
        "source_archive_sha256": prep.get("archive_sha256"),
        "artifact_hashes": hashes,
        "old_evaluation_queue_size": int(old_result["common_evaluation_queue_group_count"]),
        "preparation_partition_counts": prep.get("unique_partition_groups"),
        "preprocessing_numeric_median_count": prep.get("preprocessing_learning_disclosure", {}).get(
            "numeric_medians_learned_from_train_groups"),
        "saved_sigma_g_sq_full_precision": float(scale["value"]),
        "analysis_sigma_g_sq_v2_1": SIGMA_G_SQ,
            "saved_gradient_audit": saved_gradient,
    }
    return {"old_result": old_result, "old_checkpoint": old_checkpoint,
            "preparation": prep, "config": config, "mappings": saved_mappings,
            "grid": grid, "raw_grid_risk": raw_risk, "bindings": result}


def _load_cache(slot: str, part: str) -> tuple[np.ndarray, str]:
    path = CHECKPOINT / "predictions" / f"{slot}-{part}" / f"{part}.json"
    raw_sha = _sha(path)
    try:
        values = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
    except Exception as exc:
        raise P1Error(f"saved prediction cache invalid: {slot}/{part}: {exc}") from exc
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values <= 0.0) | (values >= 1.0)):
        raise P1Error(f"saved prediction cache contains invalid probabilities: {slot}/{part}")
    return values, raw_sha


def _load_feature_groups(source: str, keep_income: bool = False) -> dict:
    path = DATA_DIR / source
    raw_sha = _sha(path)
    raw = path.read_bytes()
    groups = m3r._parse_adult(raw, source, keep_income=keep_income)
    return groups, raw_sha


def _infer(model_id: str, matrix: np.ndarray, preprocessing: Mapping) -> np.ndarray:
    model_path = CHECKPOINT / "models" / model_id / "booster.txt"
    booster = lgb.Booster(model_file=str(model_path))
    prediction = np.asarray(booster.predict(
        m3r._dataframe(matrix, preprocessing), num_threads=1), dtype=np.float64)
    if prediction.shape != (matrix.shape[0],) or not np.isfinite(prediction).all() or np.any(
            (prediction <= 0.0) | (prediction >= 1.0)):
        raise P1Error(f"saved model produced invalid probability vector: {model_id}")
    return prediction


def _gate(dev_metrics: Mapping[str, Mapping]) -> dict:
    baseline_min = {}
    for endpoint in ("R_070", "R_085", "AURC_70_100"):
        baseline_min[endpoint] = min(float(dev_metrics[name][endpoint]) for name in BASELINES)
    differences = {}
    passed = {}
    for method in GATE_CANDIDATES:
        differences[method] = {endpoint: float(dev_metrics[method][endpoint] - baseline_min[endpoint])
                               for endpoint in baseline_min}
        passed[method] = any(value < -1e-12 for value in differences[method].values())
    if baseline_min["R_070"] >= 0.005:
        primary, trigger = "R_070", "R_070 baseline not at the declared floor"
    elif baseline_min["R_085"] >= 0.005:
        primary, trigger = "R_085", "R_070 floor; R_085 remains above the declared floor"
    else:
        primary, trigger = "AURC_70_100", "R_070 and R_085 floors; use tail AURC"
    passing = [name for name in GATE_CANDIDATES if passed[name]]
    selected = None
    if passing:
        secondary = [key for key in ("R_070", "R_085", "AURC_70_100") if key != primary]
        selected = min(passing, key=lambda name: (
            float(dev_metrics[name][primary]),
            *(float(dev_metrics[name][key]) for key in secondary),
            GATE_CANDIDATES.index(name),
        ))
    return {
        "gate_pass": bool(passing),
        "selected_candidate_for_locked_validation": selected,
        "passed_candidates": passing,
        "candidate_improvements_vs_best_baseline_min": differences,
        "strict_improvement_tolerance": 1e-12,
        "baseline_min_by_endpoint": baseline_min,
        "selection_primary_endpoint": primary,
        "selection_rule": trigger,
        "gate_rule": "candidate passes if any of R_070, R_085, AURC_70_100 is lower than min(margin_logistic,int_freq) by >1e-12",
    }


def _theory_checks() -> dict:
    sigma2 = 0.25
    z = 0.7
    # Three distinct endpoints from the signed conditional-moment specification.
    independent_equal_n = {
        "M": "z + sigma2*l'(z)",
        "W": "2*sigma2 + sigma2^2*l''(z)",
        "risk": "Phi(-s0*M/sqrt(W)); no absolute-value replacement or 0.5 cap",
    }
    complete_overlap = {"M": z, "W": 0.0,
                        "risk": signed_gaussian_flip_risk(z, z, 0.0)}
    independent_gaussian_perturbation = {
        "M": z, "W": sigma2,
        "risk": signed_gaussian_flip_risk(z, z, sigma2),
        "q1": float(ndtr(-abs(z) / math.sqrt(sigma2))),
    }
    signed_counterexample = signed_gaussian_flip_risk(0.1, -0.7, 0.2)
    if not (complete_overlap["risk"] == 0.0 and
            abs(independent_gaussian_perturbation["risk"] -
                independent_gaussian_perturbation["q1"]) < 1e-15 and
            signed_counterexample > 0.5):
        raise AssertionError("signed Gaussian endpoint/crossing checks failed")
    return {"independent_equal_sample_size": independent_equal_n,
            "complete_same_data_deterministic": complete_overlap,
            "current_estimate_plus_independent_gaussian_perturbation": independent_gaussian_perturbation,
            "negative_new_center_risk_gt_0_5_control": signed_counterexample,
            "q1_is_not_complete_overlap_risk": True}


def _self_checks() -> dict:
    # A score tie straddles 50% retained mass; the boundary group is split evenly.
    tied = compute_tie_aware_risk(np.array([0., 0., 1.]), np.array([1., 0., 1.]), 0.5)
    assert abs(tied - 0.5) < 1e-15
    assert _pair_order_diagnostics(np.array([0., 1., 2.]),
                                   np.array([2., 1., 0.]))[
                                       "discordant_pairs_vs_margin_logistic"] == 3
    assert signed_gaussian_flip_risk(0.1, -0.7, 0.2) > 0.5
    return {"tie_boundary_fractional_mass": tied,
            "three_pair_reverse_order_control": True,
            "signed_center_crossing_control": True,
            "randomized_tests_or_rng_calls": 0}


def _write_instance_scores(group_ids: list[str], z0: np.ndarray, z_int: np.ndarray,
                           audit_flips: np.ndarray, scores: Mapping[str, np.ndarray]) -> bytes:
    return _instances_csv(group_ids, z0, z_int, audit_flips, scores)


def main() -> int:
    # Refuse a repeat prefix before reading source features or running inference.
    m3r.controlled.preflight_new_directory(OUT_DIR)
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    counts = {"model_fit_calls": 0, "mapping_fit_calls": 0, "kde_fit_calls": 0,
              "random_draws": 0, "new_model_batch_inference_attempts": 0,
              "new_model_batch_inference_successes": 0,
              "reused_saved_prediction_arrays": 0,
              "adult_income_label_values_parsed_or_used": 0}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("P1 analysis-only run attempted a prohibited fit")

    # Runtime guards make the zero-fit claim executable, not a hard-coded report.
    lgb.train = forbidden
    m3r.fit_mapping = forbidden
    m3r.flip.tweedie_flip_probability = forbidden

    artifacts = _validate_saved_inputs()
    config = artifacts["config"]
    preprocessing_path = DATA_DIR / "preprocessing.json"
    preprocessing, preprocessing_sha = _read_json(preprocessing_path)
    if preprocessing_sha != artifacts["preparation"].get("prepared_artifacts", {}).get(
            "preprocessing.json", {}).get("sha256"):
        raise P1Error("saved train-only preprocessing digest differs from the preparation manifest")

    # Development source: only feature fields are parsed. Existing D0/AUDIT
    # predictions are the exact prior-run cache on the fixed sorted group list.
    dev_groups_by_id, dev_source_sha = _load_feature_groups("adult.test", keep_income=False)
    if dev_source_sha != artifacts["preparation"].get("official_files", {}).get(
            "adult.test", {}).get("sha256"):
        raise P1Error("Adult test feature source digest differs from the UCI preparation manifest")
    dev_groups = [dev_groups_by_id[key] for key in sorted(dev_groups_by_id)]
    d0_eval, d0_cache_sha = _load_cache("D0", "evaluation")
    if len(d0_eval) != len(dev_groups):
        raise P1Error("D0 cache and Adult test feature-group count disagree")
    audit_full = []
    audit_hashes = {}
    for i in range(1, 12):
        slot = f"AUDIT-{i:02d}"
        values, digest = _load_cache(slot, "evaluation")
        if values.shape != d0_eval.shape:
            raise P1Error(f"saved AUDIT cache count mismatch: {slot}")
        audit_full.append(values)
        audit_hashes[slot] = digest
    audit_full_matrix = np.column_stack(audit_full)
    counts["reused_saved_prediction_arrays"] = 12
    alert_mask = d0_eval >= ALERT_THRESHOLD
    if int(alert_mask.sum()) != 2938:
        raise P1Error(f"development alert count changed: {int(alert_mask.sum())}; expected 2938")
    dev_ids = [dev_groups[i]["group_id"] for i in np.flatnonzero(alert_mask)]
    dev_matrix_all = m3r._transform_features(dev_groups, preprocessing)
    dev_matrix = dev_matrix_all[alert_mask]
    z0_dev_all = _logit(d0_eval)
    z0_dev = z0_dev_all[alert_mask]
    audit_dev = (audit_full_matrix[alert_mask] < ALERT_THRESHOLD).astype(np.float64)
    z_int_dev = np.empty((len(dev_ids), 6), dtype=np.float64)
    inference_records = []
    for j in range(1, 7):
        model_id = f"INT-{j:02d}"
        counts["new_model_batch_inference_attempts"] += 1
        prediction = _infer(model_id, dev_matrix, preprocessing)
        counts["new_model_batch_inference_successes"] += 1
        z_int_dev[:, j-1] = _logit(prediction)
        inference_records.append({"model_id": model_id, "dataset": "adult.test D0 alert groups",
                                  "rows": len(dev_ids), "cache": "new inference from saved booster"})

    dev_scores = _candidate_scores(z0_dev, z_int_dev, audit_dev, artifacts["mappings"],
                                   artifacts["grid"], artifacts["raw_grid_risk"])
    dev_metrics = _method_metrics(dev_scores, audit_dev)
    gate = _gate(dev_metrics)
    old_metrics = artifacts["old_result"].get("methods", {})
    old_raw_brier = float(old_metrics.get("frozen_tweedie_score", {}).get("brier_finite_11"))
    old_logistic_brier = float(old_metrics.get("logistic_brier_margin", {}).get("brier_finite_11"))
    gradient = artifacts["bindings"]["saved_gradient_audit"]
    if gradient.get("groups") != 404:
        raise P1Error("saved 404-group target-mismatch diagnostic has unexpected support")
    reproduced = {
        "raw_tweedie_brier_expected": 0.038085011,
        "raw_tweedie_brier_saved": old_raw_brier,
        "margin_logistic_brier_expected": 0.036994810,
        "margin_logistic_brier_saved": old_logistic_brier,
        "old_strict_ordered_pair_count": 4313024,
        "old_raw_tweedie_vs_margin_strict_inversions": 0,
        "404_group_intermediate_margin_diagnostic": {
            "count": int(gradient["groups"]),
            "finite_flip_rate": float(gradient["finite_target_mean"]),
            "raw_tweedie_mean": float(gradient["mean_raw_probability"]),
            "shrunk_sqrt2_mean": float(gradient["mean_no_shrink_sqrt2_probability"]),
            "shrunk_single_scale_mean": float(gradient["mean_no_shrink_kappa1_probability"]),
            "mean_saved_log_density_derivative": float(gradient["mean_log_density_derivative"]),
            "negative_density_derivative_fraction": float(gradient["negative_derivative_fraction"]),
            "mean_center_correction": float(gradient["mean_center_shift_sigma2_grad"]),
            "interpretation": "target-mismatch diagnostic; not a causal attribution",
        },
    }
    # These are explicit reconstruction checks against old saved reporting.
    if not (abs(old_raw_brier - 0.03808501139170607) < 5e-10 and
            abs(old_logistic_brier - 0.03699481024491175) < 5e-10):
        raise P1Error("historical Adult negative-result Brier anchors did not match")

    locked_metrics = None
    locked_scores = None
    locked_ids = None
    locked_z0 = locked_zint = locked_audit = None
    locked_source_hashes = {}
    if gate["gate_pass"]:
        # Only after the development gate passes do we open the feature-only
        # 9,141-group locked remainder.  Income suffixes remain unparsed.
        data_groups, data_source_sha = _load_feature_groups("adult.data", keep_income=False)
        if data_source_sha != artifacts["preparation"].get("official_files", {}).get(
                "adult.data", {}).get("sha256"):
            raise P1Error("Adult train feature source digest differs from the UCI preparation manifest")
        partitions = m3r._group_assignment_manifest(data_groups, dev_groups_by_id, config)
        selected_ids = {row["group_id"] for row in partitions["calibration"]}
        locked = [row for row in partitions["calibration_candidates"]
                  if row["group_id"] not in selected_ids]
        if len(locked) != 9141 or len(selected_ids) != 512:
            raise P1Error(f"locked partition mismatch: remainder={len(locked)}, selected={len(selected_ids)}")
        locked_ids = [row["group_id"] for row in locked]
        locked_matrix = m3r._transform_features(locked, preprocessing)
        predictions = {}
        for model_id in MODEL_IDS:
            counts["new_model_batch_inference_attempts"] += 1
            prediction = _infer(model_id, locked_matrix, preprocessing)
            counts["new_model_batch_inference_successes"] += 1
            predictions[model_id] = prediction
            inference_records.append({"model_id": model_id,
                                      "dataset": "Adult 9,141 calibration-candidate remainder",
                                      "rows": len(locked_ids),
                                      "cache": "new inference from saved booster"})
        locked_d0 = predictions["D0"]
        locked_alert_mask = locked_d0 >= ALERT_THRESHOLD
        locked_alert_ids = [locked_ids[i] for i in np.flatnonzero(locked_alert_mask)]
        locked_z0 = _logit(locked_d0[locked_alert_mask])
        locked_zint = np.column_stack([_logit(predictions[f"INT-{j:02d}"][locked_alert_mask])
                                       for j in range(1, 7)])
        locked_audit = np.column_stack([predictions[f"AUDIT-{j:02d}"][locked_alert_mask] <
                                        ALERT_THRESHOLD for j in range(1, 12)]).astype(np.float64)
        locked_scores = _candidate_scores(locked_z0, locked_zint, locked_audit,
                                          artifacts["mappings"], artifacts["grid"],
                                          artifacts["raw_grid_risk"])
        locked_metrics = _method_metrics(locked_scores, locked_audit)
        locked_ids = locked_alert_ids
        locked_source_hashes = {"adult.data_sha256": data_source_sha,
                                "adult.test_feature_source_sha256": dev_source_sha,
                                "locked_feature_group_list_sha256": hashlib.sha256(
                                    m3r.canonical_json(locked_ids)).hexdigest(),
                                "all_locked_feature_group_list_sha256": hashlib.sha256(
                                    m3r.canonical_json([row["group_id"] for row in locked])).hexdigest(),
                                "selected_512_group_list_sha256": hashlib.sha256(
                                    m3r.canonical_json(sorted(selected_ids))).hexdigest()}

    counts["new_model_inference_batches_with_saved_boosters_only"] = counts[
        "new_model_batch_inference_successes"]
    counts["model_fit_calls"] = 0
    counts["mapping_fit_calls"] = 0
    counts["kde_fit_calls"] = 0
    counts["random_draws"] = 0
    counts["adult_income_label_values_parsed_or_used"] = 0

    unchanged_after = {}
    for rel, before in artifacts["bindings"]["artifact_hashes"].items():
        if rel.startswith("models/"):
            path = CHECKPOINT / rel / "booster.txt"
        elif rel.startswith("mappings/"):
            path = CHECKPOINT / rel / "mapping.json"
        elif rel == "kde/KDE":
            path = CHECKPOINT / "kde/KDE/kde.json"
        else:
            path = CHECKPOINT / "scale/SCALE/scale.json"
        after = _sha(path)
        unchanged_after[rel] = (after == before["sha256"])
    if not all(unchanged_after.values()):
        raise P1Error("a previously saved model/map/scale/KDE changed during read-only P1")

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_bytes = int(usage.ru_maxrss * (1024 if sys.platform != "darwin" else 1))
    report = {
        "schema_version": 1,
        "status": "P1_COMPLETE_LOCKED_VALIDATION_RUN" if gate["gate_pass"] else
                  "P1_DEVELOPMENT_COMPLETE_GATE_NOT_PASSED",
        "purpose": "public UCI Adult feature-only instance-transfer diagnostic; nonclinical",
        "authorization_source": "user message authorizing the complete P1 execution package and v2.1 protocol",
        "protocol": {
            "sigma_g_sq": SIGMA_G_SQ,
            "alert_rule": "saved D0 probability >= 0.5",
            "target": "mean of 11 saved AUDIT model decisions below 0.5 on each common alert group",
            "tie_rule": "exact float score classes; fractional mass at boundary; no target or row-order tie-break",
            "fixed_coverages": list(RISKS),
            "AURC100": "right Riemann mean of R(k/100), k=1..100",
            "AURC_70_100": "mean of R(k/100), k=70..100 inclusive (31 points)",
            "paired_delta_brier": "mean(q_A^2-q_B^2-2*(q_A-q_B)*t)",
            "candidate_methods": list(METHOD_ORDER),
            "gate_candidate_pool": list(GATE_CANDIDATES),
            "gate_decision": gate,
            "q_V_freedom": 0,
            "q_MV_shrunk_freedom": 0,
            "q_MV_raw_freedom": 0,
            "fit_guard": "runtime guards replaced lightgbm.train, fit_mapping and frozen KDE function with raising hooks",
            "theory_endpoints": _theory_checks(),
            "deterministic_self_checks": _self_checks(),
        },
        "historical_v2_baseline_and_target_mismatch_diagnostic": reproduced,
        "development": {
            "partition": "existing Adult adult.test feature groups; D0 >= 0.5 alert subset",
            "universe_feature_groups": int(len(dev_groups)),
            "alert_groups": int(alert_mask.sum()),
            "group_list_sha256": hashlib.sha256(m3r.canonical_json(dev_ids)).hexdigest(),
            "source_feature_file_sha256": dev_source_sha,
            "D0_cache_sha256": d0_cache_sha,
            "AUDIT_cache_sha256": audit_hashes,
            "metrics": dev_metrics,
        },
        "locked_validation": None,
        "counts": counts,
        "model_batch_inference_records": inference_records,
        "saved_artifact_bindings": artifacts["bindings"],
        "frozen_artifacts_unchanged_after_run": unchanged_after,
        "source_handling": {
            "data_read": ["Adult public feature fields only", "saved model prediction caches and boosters",
                          "saved calibration maps and frozen Tweedie grid"],
            "income_label_values_parsed_or_used": False,
            "clinical_or_confirmation_data_accessed": False,
            "project_prediction_model_fit": 0,
            "mapping_fit": 0,
            "KDE_fit": 0,
            "RNG_calls": 0,
        },
        "resources": {
            "analysis_wall_seconds_before_atomic_publish": float(time.monotonic() - started_wall),
            "analysis_process_cpu_seconds_before_atomic_publish": float(time.process_time() - started_cpu),
            "peak_rss_bytes_process": peak_rss_bytes,
            "logical_inference_threads": 1,
            "gpu_used": False,
            "measurement_scope": "from fixed-input validation through scoring and frozen-artifact post-hash check; excludes final JSON/CSV serialization and atomic publication",
        },
    }
    if gate["gate_pass"]:
        assert locked_metrics is not None and locked_scores is not None and locked_ids is not None
        report["locked_validation"] = {
            "status": "SCORED_AFTER_DEVELOPMENT_GATE_PASS",
            "partition": "Adult calibration candidates after feature-overlap removal, minus saved 512 calibration groups",
            "universe_groups": 9141,
            "D0_alert_groups": len(locked_ids),
            "group_list_sha256": locked_source_hashes["locked_feature_group_list_sha256"],
            "all_locked_group_list_sha256": locked_source_hashes[
                "all_locked_feature_group_list_sha256"],
            "selected_512_group_list_sha256": locked_source_hashes[
                "selected_512_group_list_sha256"],
            "source_hashes": locked_source_hashes,
            "metrics": locked_metrics,
            "selected_dev_candidate_locked_metrics": locked_metrics[
                gate["selected_candidate_for_locked_validation"]],
            "q_global_1_minus_q_sqrt2_paired_delta_brier": locked_metrics[
                "q_global_1"]["paired_delta_Brier_vs_q_sqrt2"],
            "interpretation": "public same-source locked remainder evaluated once after the explicit development gate; not an independent clinical confirmation set",
        }

    files: dict[str, bytes] = {
        "p1_report.json": _json_bytes(_json_safe(report)),
        "development_metrics.csv": _metric_csv(dev_metrics),
        "development_instance_scores.csv": _write_instance_scores(
            dev_ids, z0_dev, z_int_dev, audit_dev, dev_scores),
        "README.md": _render_report(report),
        "code_sha256.txt": (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + "\n").encode(),
    }
    if gate["gate_pass"]:
        files["locked_validation_metrics.csv"] = _metric_csv(locked_metrics)
        files["locked_validation_instance_scores.csv"] = _write_instance_scores(
            locked_ids, locked_z0, locked_zint, locked_audit, locked_scores)
    m3r.controlled.atomic_publish_directory(OUT_DIR, files)
    print(json.dumps({"status": report["status"], "output": str(OUT_DIR),
                      "gate_pass": gate["gate_pass"],
                      "selected": gate["selected_candidate_for_locked_validation"],
                      "new_inference_batches": counts[
                          "new_model_batch_inference_successes"],
                      "dev_alert_groups": len(dev_ids),
                      "locked_universe_groups": 9141 if gate["gate_pass"] else None,
                      "locked_alert_groups": len(locked_ids) if gate["gate_pass"] else None,
                      "wall_seconds": report["resources"]["analysis_wall_seconds_before_atomic_publish"],
                      "cpu_seconds": report["resources"]["analysis_process_cpu_seconds_before_atomic_publish"],
                      "peak_rss_bytes": peak_rss_bytes}, ensure_ascii=False))
    return 0


def _render_report(report: Mapping) -> bytes:
    lines = [
        "# M3-R P1 instance-transfer run (Adult, v2.1)", "",
        f"Status: **{report['status']}**", "",
        "Public nonclinical analysis. No model, mapping, or KDE fit occurred; Adult income labels, ICU data, and confirmation data were not read by the metric path.", "",
        "## Gate", "",
        f"- Passed: `{report['protocol']['gate_decision']['gate_pass']}`",
        f"- Candidate: `{report['protocol']['gate_decision']['selected_candidate_for_locked_validation']}`",
        f"- Rule: {report['protocol']['gate_decision']['gate_rule']}",
        f"- Primary endpoint: `{report['protocol']['gate_decision']['selection_primary_endpoint']}` ({report['protocol']['gate_decision']['selection_rule']})", "",
        "## Development metrics", "",
        "| Method | R(.50) | R(.70) | R(.80) | R(.85) | R(.90) | R(1.00) | AURC100 | AURC70–100 | AUC | Brier-11 | ΔBrier vs margin | ΔBrier vs q√2 | ECE-10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in METHOD_ORDER:
        m = report["development"]["metrics"][name]
        lines.append("| " + " | ".join([name] + [f"{m[key]:.9f}" for key in (
            "R_050", "R_070", "R_080", "R_085", "R_090", "R_100", "AURC100",
            "AURC_70_100", "AUC", "Brier_finite_11",
            "paired_delta_Brier_vs_margin_logistic", "paired_delta_Brier_vs_q_sqrt2", "ECE10")]) + " |")
    lines.extend(["", "## Paired rank and tie diagnostics", "",
                  "Discordance and candidate ties are counted only over pairs strictly ordered by `margin_logistic`; exact floating score equality defines ties.", "",
                  "| Method | Margin strict pairs | Discordant | Tied on strict pairs | Max tie class / n |",
                  "|---|---:|---:|---:|---:|"])
    for name in METHOD_ORDER:
        m = report["development"]["metrics"][name]
        lines.append(f"| {name} | {m['reference_strict_pair_count']} | {m['discordant_pairs_vs_margin_logistic']} | {m['tied_candidate_pairs_on_reference_strict_pairs']} | {m['max_exact_score_tie_class_count']} / {m['n_alert_groups']} ({m['max_exact_score_tie_class_fraction']:.6f}) |")
    shared = report["development"]["metrics"]["_shared"]
    lines.extend(["", f"`int_freq` k=0 plateau: {shared['int_freq_k0_group_count']} / {shared['n_alert_groups']} = {shared['int_freq_k0_group_fraction']:.6f} at score {shared['int_freq_k0_score']:.9f}.", ""])
    if report.get("locked_validation"):
        locked = report["locked_validation"]
        lines.extend(["## Locked 9,141-group validation", "",
                      f"D0 alert queue: {locked['D0_alert_groups']} groups. Gate-selected candidate: `{report['protocol']['gate_decision']['selected_candidate_for_locked_validation']}`.", "",
                      "| Method | R(.50) | R(.70) | R(.80) | R(.85) | R(.90) | R(1.00) | AURC100 | AURC70–100 | AUC | Brier-11 | ΔBrier vs margin | ΔBrier vs q√2 | ECE-10 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for name in METHOD_ORDER:
            m = locked["metrics"][name]
            lines.append("| " + " | ".join([name] + [f"{m[key]:.9f}" for key in (
                "R_050", "R_070", "R_080", "R_085", "R_090", "R_100", "AURC100",
                "AURC_70_100", "AUC", "Brier_finite_11",
                "paired_delta_Brier_vs_margin_logistic", "paired_delta_Brier_vs_q_sqrt2", "ECE10")]) + " |")
        lines.extend(["", f"Final paired ΔBrier (`q_global_1 - q_sqrt2`): `{locked['q_global_1_minus_q_sqrt2_paired_delta_brier']:.12g}`.",
                      f"The selected method's locked metrics: `{json.dumps(_json_safe(locked['selected_dev_candidate_locked_metrics']), sort_keys=True)}`.", ""])
    else:
        lines.extend(["## Locked validation", "",
                      "Not run because no candidate passed the predeclared development gate. The 9,141 feature groups were not parsed or scored.", ""])
    lines.extend(["## Resources and counts", "",
                  f"- New model-batch inference calls: {report['counts']['new_model_batch_inference_successes']} successful / {report['counts']['new_model_batch_inference_attempts']} attempted (upper bound 24).",
                  f"- Model / mapping / KDE fits: {report['counts']['model_fit_calls']} / {report['counts']['mapping_fit_calls']} / {report['counts']['kde_fit_calls']}.",
                  f"- Analysis wall / process CPU before atomic output publication: {report['resources']['analysis_wall_seconds_before_atomic_publish']:.3f}s / {report['resources']['analysis_process_cpu_seconds_before_atomic_publish']:.3f}s.",
                  f"- Peak RSS: {report['resources']['peak_rss_bytes_process']} bytes.",
                  f"- Runtime output: `{OUT_DIR}`.", "",
                  "## Preserved historical result and limitations", "",
                  "The prior Adult report remains unchanged: raw Tweedie Brier 0.038085011, frozen Logistic-Brier 0.036994810; 4,313,024 strictly ordered margin/raw-score pairs had zero inversions. The 404-group intermediate-margin decomposition is recorded as target-mismatch evidence, not causal attribution.", "",
                  "The 9,141 groups share the public Adult source and saved model bank. They are a locked remainder evaluated after a development gate, not an independent clinical/ICU confirmation set. The AUDIT ensemble is a finite reference target, not the unknown retraining distribution.", ""])
    return "\n".join(lines).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
