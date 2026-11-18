#!/usr/bin/env python
"""Follow-up S1-S4 analysis, reusing the saved public Diabetes/Adult artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from experiments.core import run_w1_w4 as base
from experiments.public_benchmarks import run_public_experiments as pub

OUT = base.OUT / "S1_S4_20261111"
RUNS = base.RUNS
SEED_BASE = base.SEED_BASE
N_FULL = base.LGB_FULL_N
RIDGE = 1.0
LEARNED_L2 = 0.01


class AuditError(RuntimeError):
    pass


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(x: Any):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(_jsonable(value), f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows([_jsonable(x) for x in rows])


def _write_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    buff = io.BytesIO()
    np.savez_compressed(buff, **arrays)
    with path.open("xb") as f:
        f.write(buff.getvalue())


def _selected_splits():
    selected, audit = pub.load_selected_diabetes()
    train = selected[selected.split == "TRAIN"].reset_index(drop=True)
    cal = selected[selected.split == "CAL"].reset_index(drop=True)
    eva = selected[selected.split == "EVAL"].reset_index(drop=True)
    if (len(train), len(cal), len(eva)) != (42058, 13983, 13949):
        raise AuditError("frozen Diabetes split size changed")
    return train, cal, eva, audit


def _bank_train(train_all: pd.DataFrame, scale: str, bank: int) -> pd.DataFrame:
    if scale == "14k":
        tr = train_all[train_all.bank_id == bank].sort_values(
            "patient_nbr", kind="mergesort").reset_index(drop=True)
    elif scale == "33k":
        rng = np.random.default_rng(SEED_BASE + bank * 10000)
        take = np.sort(rng.choice(len(train_all), size=N_FULL, replace=False))
        tr = train_all.iloc[take].reset_index(drop=True)
    else:
        raise AuditError(f"unsupported scale: {scale}")
    manifest_path = RUNS / ("w2b" if scale == "14k" else "w2b_fulltrain") / f"bank{bank}/bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ids = tr.patient_nbr.astype(str).tolist()
    if _sha_bytes("\n".join(ids).encode()) != manifest["train_patient_sha256"]:
        raise AuditError(f"reconstructed training row order/hash mismatch: {scale} bank {bank}")
    if len(tr) != manifest["train_n"]:
        raise AuditError(f"training row count mismatch: {scale} bank {bank}")
    return tr


def _load_design(scale: str, bank: int):
    train_all, cal, eva, _audit = _selected_splits()
    tr = _bank_train(train_all, scale, bank)
    bdir = RUNS / ("w2b" if scale == "14k" else "w2b_fulltrain") / f"bank{bank}"
    prep = json.loads((bdir / "preprocessor.json").read_text())
    fitted = pub.fit_sparse_proxy_preprocessor(tr, prep)
    xtr = base._diabetes_sparse_features(tr, prep, fitted)
    xc = base._diabetes_sparse_features(cal, prep, fitted)
    xe = base._diabetes_sparse_features(eva, prep, fitted)
    y = tr.y_readmit30.to_numpy(np.float64)
    return tr, cal, eva, prep, fitted, xtr, xc, xe, y, bdir


def _model_beta(model_path: Path) -> tuple[np.ndarray, dict]:
    model = json.loads(model_path.read_text())
    beta = np.asarray(model["beta"], dtype=np.float64)
    if beta.ndim != 1 or not np.isfinite(beta).all():
        raise AuditError(f"invalid saved beta: {model_path}")
    return beta, model


def _bootstrap_weights(n: int, bank: int, role_index: int) -> tuple[np.ndarray, int]:
    code = 300 + role_index
    seed = SEED_BASE + 500000 + bank * 10000 + code
    rng = np.random.default_rng(seed + 50000)
    w = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
    if int(w.sum()) != n:
        raise AuditError("multinomial bootstrap weight sum mismatch")
    return w, seed


def _penalty(d: int) -> np.ndarray:
    p = np.ones(d, dtype=np.float64) * RIDGE
    p[0] = 0.0
    return p


def _hessian(x: sparse.csr_matrix, curvature_weight: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    weighted = x.multiply(np.asarray(curvature_weight, dtype=np.float64)[:, None]).tocsr()
    h = (x.T @ weighted).toarray()
    h.flat[::h.shape[0] + 1] += penalty
    return (h + h.T) * 0.5


def _cor(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _summarize_vector(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    return {"mean": float(np.mean(v)), "median": float(np.median(v)),
            "p05": float(np.quantile(v, .05)), "p95": float(np.quantile(v, .95))}


def _s1_one(scale: str, bank: int, outroot: Path) -> tuple[dict, dict]:
    t_wall, t_cpu = time.monotonic(), time.process_time()
    tr, cal, eva, prep, fitted, xtr, xc, xe, y, bdir = _load_design(scale, bank)
    model_dir = bdir / "models"
    beta0, d0_meta = _model_beta(model_dir / "D0/model.json")
    n, d = xtr.shape
    if beta0.shape != (d,):
        raise AuditError(f"D0 coefficient/design mismatch for {scale} bank {bank}")
    beta_a = []; audit_meta = []; weight_summaries = []
    for j in range(1, 12):
        beta, meta = _model_beta(model_dir / f"AUDIT-{j:02d}/model.json")
        if beta.shape != (d,):
            raise AuditError(f"AUDIT coefficient/design mismatch for {scale} bank {bank} model {j}")
        w, seed = _bootstrap_weights(n, bank, j)
        if (int(w.sum()) != int(meta["fit_weight_sum"]) or
                int(np.count_nonzero(w)) != int(meta["nonzero_weight_patients"])):
            raise AuditError(f"reconstructed AUDIT weights disagree with saved model metadata: {scale}/{bank}/{j}")
        beta_a.append(beta); audit_meta.append(meta)
        weight_summaries.append({"model": f"AUDIT-{j:02d}", "seed": seed,
            "sum": int(w.sum()), "nonzero": int(np.count_nonzero(w)),
            "sha256_int64": _sha_bytes(w.astype("<i8").tobytes())})
    beta_a = np.vstack(beta_a)
    gamma = float(logit(float(json.loads((base.OUT / ("w2b_diabetes_logistic" if scale == "14k" else "w2b_diabetes_logistic_fulltrain") / "report.json").read_text())["banks"][bank]["threshold_tau"])))
    saved_path = bdir / "predictions.npz"
    with np.load(saved_path, allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    z0c = np.asarray(xc @ beta0).reshape(-1) - gamma
    z0e = np.asarray(xe @ beta0).reshape(-1) - gamma
    zfull_all = np.asarray(xe @ beta_a.T) - gamma
    err0 = max(float(np.max(np.abs(z0c - saved["z0c"]))),
               float(np.max(np.abs(z0e - saved["z0e"]))))
    erra = float(np.max(np.abs(zfull_all - saved["zaudit"])))
    if max(err0, erra) > 1e-6:
        raise AuditError(f"saved prediction reproduction mismatch {scale}/{bank}: D0={err0}, AUDIT={erra}")
    alert_c = z0c >= 0.0; alert_e = z0e >= 0.0
    if int(alert_e.sum()) != int((saved["z0e"] >= 0).sum()):
        raise AuditError("EVAL alert queue changed")
    xe_a = xe[alert_e].tocsr(); xc_a = xc[alert_c].tocsr()
    z0e_a = z0e[alert_e]
    zfull = zfull_all[alert_e, :]

    # Stage 0: the same score covariance and ridge/intercept convention as the saved IF code.
    if_cpu0 = time.process_time()
    eta0 = np.asarray(xtr @ beta0).reshape(-1)
    p0 = expit(np.clip(eta0, -40, 40))
    resid = p0 - y
    score_rows = xtr.multiply(resid[:, None]).tocsr()
    score_mean = np.asarray(score_rows.mean(axis=0)).reshape(-1)
    jmat = (score_rows.T @ score_rows).toarray() - n * np.outer(score_mean, score_mean)
    pen = _penalty(d)
    h0 = _hessian(xtr, np.maximum(p0 * (1.0 - p0), 1e-12), pen)
    fac0 = cho_factor(h0, lower=True, check_finite=False)
    hinv = cho_solve(fac0, np.eye(d), check_finite=False)
    middle = hinv @ jmat @ hinv
    middle = (middle + middle.T) * 0.5
    xe_a_dense = xe_a.toarray()
    xc_a_dense = xc_a.toarray()
    v_if = np.einsum("ij,jk,ik->i", xe_a_dense, middle, xe_a_dense, optimize=True)
    v_if_cal = np.einsum("ij,jk,ik->i", xc_a_dense, middle, xc_a_dense, optimize=True)
    v_if = np.maximum(v_if, 0.0)
    v_if_cal = np.maximum(v_if_cal, 0.0)
    h0_grad = np.asarray(xtr.T @ resid).reshape(-1) + pen * beta0
    if_cpu = time.process_time() - if_cpu0

    # Stages 1 and 2 use each AUDIT model's exact saved multinomial multiplicities.
    stage1_deltas = []; stage2_deltas = []; hessian_diagnostics = []
    grad_norms = []
    curvature0 = np.maximum(p0 * (1.0 - p0), 1e-12)
    for j in range(1, 12):
        w, _seed = _bootstrap_weights(n, bank, j)
        grad = np.asarray(xtr.T @ (w * resid)).reshape(-1) + pen * beta0
        grad_norms.append(float(np.max(np.abs(grad))))
        delta1 = -cho_solve(fac0, grad, check_finite=False)
        stage1_deltas.append(delta1)
        hb = _hessian(xtr, w * curvature0, pen)
        try:
            facb = cho_factor(hb, lower=True, check_finite=False)
            delta2 = -cho_solve(facb, grad, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise AuditError(f"Stage 2 weighted Hessian failed: {scale}/{bank}/AUDIT-{j:02d}") from exc
        stage2_deltas.append(delta2)
        hessian_diagnostics.append({"audit_index": j,
            "weighted_hessian_trace_ratio_to_H0": float(np.trace(hb) / np.trace(h0)),
            "min_cholesky_diagonal": float(np.min(np.diag(facb[0]))),
            "max_cholesky_diagonal": float(np.max(np.diag(facb[0])))})
    stage1_deltas = np.vstack(stage1_deltas)
    stage2_deltas = np.vstack(stage2_deltas)
    zlin = z0e_a[:, None] + np.asarray(xe_a @ stage1_deltas.T)
    z1step = z0e_a[:, None] + np.asarray(xe_a @ stage2_deltas.T)
    s1 = np.var(zlin, axis=1, ddof=1)
    s2 = np.var(z1step, axis=1, ddof=1)
    s3 = np.var(zfull, axis=1, ddof=1)
    mse1_full = np.mean((zlin - zfull) ** 2, axis=1)
    mse2_full = np.mean((z1step - zfull) ** 2, axis=1)
    mse0_full = np.mean((z0e_a[:, None] - zfull) ** 2, axis=1)

    # Feature attribution: all numerical columns and OHE columns with support >=50 are common.
    num_n = len(pub.DIABETES_NUMERIC)
    support = np.asarray((xtr[:, 1 + num_n:] != 0).sum(axis=0)).reshape(-1)
    rare_local = np.flatnonzero(support < 50)
    rare_idx = rare_local + 1 + num_n
    rare_mask = np.zeros(d, dtype=bool); rare_mask[rare_idx] = True
    common_idx = np.flatnonzero(~rare_mask)
    delta_full = beta_a - beta0[None, :]
    norm_total = np.sum(delta_full ** 2, axis=1)
    norm_rare = np.sum(delta_full[:, rare_idx] ** 2, axis=1)
    norm_common = np.sum(delta_full[:, common_idx] ** 2, axis=1)
    zrare = np.asarray(xe_a[:, rare_idx] @ delta_full[:, rare_idx].T)
    zcommon = np.asarray(xe_a[:, common_idx] @ delta_full[:, common_idx].T)
    v_rare = np.var(zrare, axis=1, ddof=1)
    v_common = np.var(zcommon, axis=1, ddof=1)
    v_cross = s3 - v_rare - v_common
    attribution = {
        "feature_dimension": d, "numeric_feature_count": num_n,
        "onehot_feature_count": int(len(support)), "rare_onehot_support_lt50_count": int(len(rare_idx)),
        "rare_onehot_support_lt50_fraction": float(len(rare_idx) / max(1, len(support))),
        "full_delta_theta_norm2_mean": float(norm_total.mean()),
        "full_delta_theta_norm2_median": float(np.median(norm_total)),
        "rare_onehot_delta_theta_norm2_share_mean_over_models": float(np.mean(norm_rare / np.maximum(norm_total, 1e-300))),
        "common_and_continuous_delta_theta_norm2_share_mean_over_models": float(np.mean(norm_common / np.maximum(norm_total, 1e-300))),
        "rare_onehot_logit_variance_mean": float(v_rare.mean()),
        "common_and_continuous_logit_variance_mean": float(v_common.mean()),
        "cross_covariance_logit_variance_mean": float(v_cross.mean()),
        "rare_onehot_logit_variance_share_of_full": float(v_rare.mean() / max(float(s3.mean()), 1e-300)),
        "common_and_continuous_logit_variance_share_of_full": float(v_common.mean() / max(float(s3.mean()), 1e-300)),
        "cross_covariance_logit_variance_share_of_full": float(v_cross.mean() / max(float(s3.mean()), 1e-300)),
        "largest_abs_full_delta_beta_rare": float(np.max(np.abs(delta_full[:, rare_idx]))) if len(rare_idx) else 0.0,
        "largest_abs_full_delta_beta_common": float(np.max(np.abs(delta_full[:, common_idx]))) if len(common_idx) else 0.0,
    }
    stage_vectors = {"Stage0_v_IF": v_if, "Stage1_var_z_lin": s1,
                     "Stage2_var_z_1step": s2, "Stage3_var_z_full": s3}
    base_mean = float(v_if.mean())
    summary = {"scale": scale, "bank": bank, "train_n": n,
        "eval_alert_n": int(alert_e.sum()), "D0_gradient_inf_norm": float(np.max(np.abs(h0_grad))),
        "max_saved_prediction_reproduction_abs_error": max(err0, erra),
        "stage_means_medians": {k: _summarize_vector(v) for k, v in stage_vectors.items()},
        "stage_mean_ratio_to_stage0": {k: float(v.mean() / base_mean) for k, v in stage_vectors.items()},
        "stage_pairwise_instance_correlations": {
            "stage0_vs_stage1": _cor(v_if, s1), "stage1_vs_stage2": _cor(s1, s2),
            "stage2_vs_stage3": _cor(s2, s3), "stage0_vs_stage3": _cor(v_if, s3),
            "stage1_vs_stage3": _cor(s1, s3)},
        "stage1_vs_full_per_instance_mse": _summarize_vector(mse1_full),
        "stage2_vs_full_per_instance_mse": _summarize_vector(mse2_full),
        "D0_center_vs_full_per_instance_mse": _summarize_vector(mse0_full),
        "stage1_to_stage0_instance_ratio": _summarize_vector(s1 / np.maximum(v_if, 1e-300)),
        "stage2_to_stage1_mean_increment": float(np.mean(s2 - s1)),
        "stage3_to_stage2_mean_increment": float(np.mean(s3 - s2)),
        "gradient_inf_norm_at_theta0_by_audit": grad_norms,
        "audit_weight_reconstruction": weight_summaries,
        "weighted_hessian_diagnostics": hessian_diagnostics,
        "feature_attribution": attribution,
        "shared_preprocessor": True,
        "preprocessor_fit_rows": int(prep["fit_rows"]),
        "no_prediction_model_fits": True,
        "IF_and_stage_CPU_seconds": time.process_time() - t_cpu,
        "wall_seconds": time.monotonic() - t_wall,
        "peak_rss_bytes_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}
    out = outroot / scale / f"bank{bank}"
    out.mkdir(parents=True, exist_ok=False)
    _write_npz(out / "stage_arrays.npz", v_if=v_if, v_if_cal=v_if_cal,
        z0_cal_alert=z0c[alert_c], z0_eval_alert=z0e_a, z_int_cal_alert=saved["zic"][alert_c],
        z_int_eval_alert=saved["zie"][alert_e], z_calref_cal_alert=saved["zcalref"][alert_c],
        z_audit_eval_alert=zfull, z_lin=zlin, z_1step=z1step,
        var_stage1=s1, var_stage2=s2, var_stage3=s3)
    _write_json(out / "report.json", summary)
    meta = {"saved_prediction_sha256": _sha_file(saved_path),
        "D0_model_sha256": _sha_file(model_dir / "D0/model.json"),
        "AUDIT_model_sha256": {f"AUDIT-{j:02d}": _sha_file(model_dir / f"AUDIT-{j:02d}/model.json") for j in range(1,12)},
        "training_patient_sha256": json.loads((bdir / "bank_manifest.json").read_text())["train_patient_sha256"],
        "sample_weight_reconstruction": "seed formula from saved fit records; exact sum and nonzero counts match each AUDIT manifest"}
    _write_json(out / "provenance.json", meta)
    row = {"scale": scale, "bank": bank, "train_n": n, "eval_alert_n": int(alert_e.sum())}
    for key, vec in stage_vectors.items():
        row[key + "_mean"] = float(vec.mean())
        row[key + "_median"] = float(np.median(vec))
        row[key + "_ratio_to_stage0"] = float(vec.mean() / base_mean)
    row.update({"corr_stage0_stage3": summary["stage_pairwise_instance_correlations"]["stage0_vs_stage3"],
        "corr_stage2_stage3": summary["stage_pairwise_instance_correlations"]["stage2_vs_stage3"],
        "stage1_full_mse_mean": float(mse1_full.mean()), "stage2_full_mse_mean": float(mse2_full.mean()),
        "rare_delta_theta_norm2_share": attribution["rare_onehot_delta_theta_norm2_share_mean_over_models"],
        "rare_logit_var_share": attribution["rare_onehot_logit_variance_share_of_full"],
        "if_cpu_seconds": if_cpu, "total_cpu_seconds": summary["IF_and_stage_CPU_seconds"]})
    return summary, row


def s1_bank0():
    outroot = OUT / "s1_four_stage"
    outroot.mkdir(parents=True, exist_ok=False)
    reports = []; rows = []
    for scale in ("14k", "33k"):
        summary, row = _s1_one(scale, 0, outroot)
        reports.append(summary); rows.append(row)
    _write_csv(outroot / "bank0_four_stage.csv", rows)
    _write_json(outroot / "bank0_report.json", {"status":"S1_BANK0_COMPLETE",
        "scope":"Bank 0 only; both Diabetes Logistic training scales; saved models and exact bootstrap multiplicities reused",
        "banks":reports,
        "interpretation_rule":"Stage1/Stage0 near one supports covariance/normalization agreement; Stage2-Stage1 isolates random-Hessian effects; Stage3-Stage2 isolates full-fit nonlinear remainder. B=11 is a finite AUDIT sample and ratios have sampling variability."})


def s1_other_banks():
    outroot = OUT / "s1_four_stage"
    if not (outroot / "bank0_report.json").is_file():
        raise AuditError("run S1 Bank 0 first and inspect its decomposition")
    reports = []; rows = []
    for scale in ("14k", "33k"):
        for bank in (1, 2):
            summary, row = _s1_one(scale, bank, outroot)
            reports.append(summary); rows.append(row)
    _write_csv(outroot / "banks1_2_four_stage.csv", rows)
    bank0 = json.loads((outroot / "bank0_report.json").read_text())["banks"]
    _write_csv(outroot / "all_banks_four_stage.csv", [
        {"scale":x["scale"], "bank":x["bank"], "train_n":x["train_n"],
         **{f"{k}_{stat}":x["stage_means_medians"][k][stat]
            for k in x["stage_means_medians"] for stat in ("mean","median")},
         **{f"{k}_ratio_stage0":x["stage_mean_ratio_to_stage0"][k]
            for k in x["stage_mean_ratio_to_stage0"]},
         "corr_stage0_stage3":x["stage_pairwise_instance_correlations"]["stage0_vs_stage3"],
         "corr_stage2_stage3":x["stage_pairwise_instance_correlations"]["stage2_vs_stage3"],
         "rare_delta_theta_norm2_share":x["feature_attribution"]["rare_onehot_delta_theta_norm2_share_mean_over_models"],
         "rare_logit_var_share":x["feature_attribution"]["rare_onehot_logit_variance_share_of_full"]}
        for x in bank0 + reports])
    _write_json(outroot / "all_banks_report.json", {"status":"S1_ALL_BANKS_COMPLETE",
        "banks":bank0 + reports, "no_new_prediction_model_fits":True})


def _fit_learned_brier(xcal: np.ndarray, target: np.ndarray, signs: list[int], names: list[str]):
    """Fit weakly regularized Brier-logit with sign bounds on standardized features."""
    xcal = np.asarray(xcal, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if xcal.ndim != 2 or target.shape != (xcal.shape[0],) or not np.isfinite(xcal).all():
        raise AuditError("invalid learned CALREF design")
    mu = xcal.mean(axis=0)
    sd = xcal.std(axis=0)
    sd[sd < 1e-10] = 1.0
    xs = (xcal - mu) / sd
    design = np.column_stack((np.ones(len(xs)), xs))
    mean_target = float(np.clip(target.mean(), 1e-6, 1 - 1e-6))
    init = np.r_[logit(mean_target), np.zeros(xcal.shape[1])]
    bounds = [(None, None)]
    for direction in signs:
        bounds.append((None, 0.0) if direction < 0 else (0.0, None) if direction > 0 else (None, None))

    def objective(w):
        q = expit(np.clip(design @ w, -40, 40))
        delta = q - target
        loss = float(np.mean(delta * delta) + .5 * LEARNED_L2 * np.dot(w[1:], w[1:]))
        grad = (2.0 / len(target)) * (design.T @ (delta * q * (1.0 - q)))
        grad[1:] += LEARNED_L2 * w[1:]
        return loss, grad

    fitted = minimize(objective, init, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-9, "maxls": 50})
    if not fitted.success and np.linalg.norm(fitted.jac, ord=np.inf) > 1e-5:
        raise AuditError(f"CALREF learned baseline optimizer failed: {fitted.message}")
    raw_coef = fitted.x[1:] / sd
    raw_intercept = float(fitted.x[0] - np.dot(mu, raw_coef))
    return {"mu": mu, "sd": sd, "weights_standardized": fitted.x,
            "weights_raw": np.r_[raw_intercept, raw_coef],
            "optimizer_success": bool(fitted.success), "optimizer_message": str(fitted.message),
            "cal_brier_objective": float(fitted.fun), "l2": LEARNED_L2,
            "features": names, "direction_bounds": signs,
            "fit_rows": int(len(target)), "fit_target_mean": float(target.mean())}


def _predict_learned(model: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return expit(np.clip(model["weights_standardized"][0] +
                         ((x - model["mu"]) / model["sd"]) @ model["weights_standardized"][1:],
                         -40, 40))


def _eval_metrics_row(dataset: str, bank: int, operation: str, method: str,
                      q: np.ndarray, audit: np.ndarray, qmargin: np.ndarray) -> dict:
    q = np.asarray(q, float); audit = np.asarray(audit, np.int8)
    t = audit.mean(axis=1)
    return {"dataset":dataset, "bank":bank, "operation":operation, "method":method,
        "n_alerts":int(len(q)), "R_070":base._risk(q,t,.70),
        "AURC100":float(np.mean([base._risk(q,t,k/100) for k in range(1,101)])),
        "AUC":base._auc_finite(q,audit),
        "Brier":float(np.mean((q[:,None]-audit)**2)),
        "ECE10":float(pub._ece10(q,t)),
        "delta_Brier_vs_Margin":float(np.mean(q*q-qmargin*qmargin-2*(q-qmargin)*t))}


def _load_diabetes_case(family: str, bank: int):
    if family == "LGB_33k":
        run = RUNS / "w2a"; map_dir = run / f"bank{bank}"
        s1_path = OUT / "s1_four_stage/33k" / f"bank{bank}/stage_arrays.npz"
    elif family == "Logistic_33k":
        run = RUNS / "w2b_fulltrain"; map_dir = run / f"bank{bank}"
        s1_path = OUT / "s1_four_stage/33k" / f"bank{bank}/stage_arrays.npz"
    else:
        raise AuditError(f"unsupported diabetes case family: {family}")
    with np.load(map_dir / "predictions.npz", allow_pickle=False) as saved:
        z0c=saved["z0c"].copy();z0e=saved["z0e"].copy();zic=saved["zic"].copy();zie=saved["zie"].copy()
        zcalref=saved["zcalref"].copy();zaudit=saved["zaudit"].copy()
    ac=z0c>=0;ae=z0e>=0
    if family == "Logistic_33k":
        with np.load(s1_path,allow_pickle=False) as stage:
            vcal=stage["v_if_cal"].copy();veval=stage["v_if"].copy()
            z0ca=stage["z0_cal_alert"].copy();z0ea=stage["z0_eval_alert"].copy()
        if (int(ac.sum())!=len(z0ca) or int(ae.sum())!=len(z0ea) or
            vcal.shape!=(int(ac.sum()),) or veval.shape!=(int(ae.sum()),)):
            raise AuditError(f"CAL/EVAL alert alignment failure for {family}/{bank}")
    else:
        # LightGBM has no analytic Logistic IF object; use its own frozen alert masks.
        z0ca=z0c[ac].copy();z0ea=z0e[ae].copy()
        vcal=veval=None
    tcal=(zcalref[ac]<0).mean(axis=1);audit=(zaudit[ae]<0).astype(np.int8)
    sigma2=float(np.var(zic[ac],axis=1,ddof=1).mean())
    zbarc=zic[ac].mean(axis=1);zbare=zie[ae].mean(axis=1)
    s2c=np.var(zic[ac],axis=1,ddof=1);s2e=np.var(zie[ae],axis=1,ddof=1)
    vtildec=.5*s2c+.5*sigma2;vtildee=.5*s2e+.5*sigma2
    mcal=.5*(z0ca+zbarc);meval=.5*(z0ea+zbare)
    if family == "LGB_33k":
        maps=json.loads((map_dir/"evaluation.json").read_text())["calref_maps"]
    else:
        maps=json.loads((map_dir/"calibrators.json").read_text())
    def apply(method, raw):
        if method not in maps: raise AuditError(f"missing CALREF map {method} for {family}/{bank}")
        return base._apply_cal_map(maps[method],raw)
    qmargin=apply("margin_logistic",z0ea)
    qfusion=apply("fusion_mean_only",mcal) # CAL-side value unused; map is common
    qfusion_eval=apply("fusion_mean_only",meval)
    qqmv_eval=apply("qMV",meval/np.sqrt(vtildee))
    x2c=np.column_stack((z0ca,zbarc));x2e=np.column_stack((z0ea,zbare))
    x3c=np.column_stack((z0ca,zbarc,np.log(vtildec)))
    x3e=np.column_stack((z0ea,zbare,np.log(vtildee)))
    fit2=_fit_learned_brier(x2c,tcal,[-1,-1],["z0","z_bar_INT"])
    fit3=_fit_learned_brier(x3c,tcal,[-1,-1,1],["z0","z_bar_INT","log_v_INT"])
    pred2=_predict_learned(fit2,x2e);pred3=_predict_learned(fit3,x3e)
    methods={"Margin_CALREF":qmargin,"FusionMean_CALREF":qfusion_eval,
             "qMV_CALREF":qqmv_eval,"Learned_2D_z0_zbar":pred2,"Learned_3D_z0_zbar_logv":pred3}
    params={"Learned_2D_z0_zbar":fit2,"Learned_3D_z0_zbar_logv":fit3}
    if family=="Logistic_33k":
        if np.any(vcal<=0) or np.any(veval<=0):raise AuditError("nonpositive conditional IF variance")
        if_score_c=z0ca/np.sqrt(vcal);if_score_e=z0ea/np.sqrt(veval)
        if_map=base._cal_stability(if_score_c,tcal,if_score_e)[1]
        qif=base._apply_cal_map(if_map,if_score_e)
        xifc=np.column_stack((z0ca,np.log(vcal)));xife=np.column_stack((z0ea,np.log(veval)))
        fitif=_fit_learned_brier(xifc,tcal,[-1,1],["z_center","log_v_IF"])
        qif2=_predict_learned(fitif,xife)
        methods.update({"ConditionIF_CALREF":qif,"Learned_IF_2D_zcenter_logv":qif2})
        params.update({"ConditionIF_CALREF_map":if_map,"Learned_IF_2D_zcenter_logv":fitif})
    rowdata={"tcal":tcal,"audit":audit,"qmargin":qmargin,"sigma2":sigma2,
             "eval_alert_n":int(ae.sum()),"cal_alert_n":int(ac.sum()),
             "v_if_cal_mean":float(vcal.mean()) if family=="Logistic_33k" else None,
             "v_if_eval_mean":float(veval.mean()) if family=="Logistic_33k" else None,
             "new_map_fits":1 if family=="Logistic_33k" else 0,
             "learned_fits":sum(str(k).startswith("Learned_") for k in params)}
    return methods,params,rowdata


def _adult_s2_case(bank: int, operation: str):
    a,b,fit,xa,xb,xc,xe,parts=base._adult_p2_design(bank)
    ncal=len(xc);beta0,_=pub._p2_load_beta(bank,"shared","D0")
    xall=np.vstack((xc,xe));z0all=xall@beta0;z0c_all=z0all[:ncal];z0e_all=z0all[ncal:]
    ac=z0c_all>=0;ae=z0e_all>=0
    bundle_path=base.P2_FOLLOW/f"bank{bank}/{operation}.npz"
    with np.load(bundle_path,allow_pickle=False) as saved:
        logits=saved["logits"].copy()
    if logits.shape!=(len(z0all),23):raise AuditError("Adult P2 reference logit shape mismatch")
    intc=logits[:ncal,:6][ac];inte=logits[ncal:,:6][ae]
    calref=logits[:ncal,6:12][ac];audit_logits=logits[ncal:,12:][ae]
    tc=(calref<0).mean(axis=1);audit=(audit_logits<0).astype(np.int8)
    z0c=z0c_all[ac];z0e=z0e_all[ae]
    zbarc=intc.mean(axis=1);zbare=inte.mean(axis=1)
    sigma2=float(np.mean(np.var(intc,axis=1,ddof=1)))
    s2c=np.var(intc,axis=1,ddof=1);s2e=np.var(inte,axis=1,ddof=1)
    vt_c=.5*s2c+.5*sigma2;vt_e=.5*s2e+.5*sigma2
    mc=.5*(z0c+zbarc);me=.5*(z0e+zbare)
    qmaps=json.loads((base.OUT/"w1_p2"/f"bank{bank}_{operation}"/"calibrators.json").read_text())
    def amap(name,raw):return base._apply_cal_map(qmaps[name],raw)
    methods={"Margin_CALREF":amap("margin_logistic",z0e),
        "FusionMean_CALREF":amap("fusion_mean_only",me),
        "qMV_CALREF":amap("qMV",me/np.sqrt(vt_e))}
    x2c=np.column_stack((z0c,zbarc));x2e=np.column_stack((z0e,zbare))
    x3c=np.column_stack((z0c,zbarc,np.log(vt_c)));x3e=np.column_stack((z0e,zbare,np.log(vt_e)))
    fit2=_fit_learned_brier(x2c,tc,[-1,-1],["z0","z_bar_INT"])
    fit3=_fit_learned_brier(x3c,tc,[-1,-1,1],["z0","z_bar_INT","log_v_INT"])
    methods["Learned_2D_z0_zbar"]=_predict_learned(fit2,x2e)
    methods["Learned_3D_z0_zbar_logv"]=_predict_learned(fit3,x3e)
    beta_r=beta0.copy() if operation=="bootstrap" else np.asarray(pub.read_json(base.P2_FOLLOW/f"bank{bank}/{operation}_anchor.json")["beta"],float)
    alpha=0.0 if operation in ("bootstrap","alpha_0") else .5
    operator_score,vr_all=base.follow.anchor_if(xa,pub._group_soft_labels(a),xb,
        pub._group_soft_labels(b),beta_r,xall,alpha,operation=="bootstrap")
    zrc=(xall[:ncal]@beta_r)[ac];zre=(xall[ncal:]@beta_r)[ae]
    vrc=np.maximum(vr_all[:ncal][ac],1e-12);vre=np.maximum(vr_all[ncal:][ae],1e-12)
    q_if_map=qmaps["operator_center_IF"]
    methods["OperatorCenter_CALREF"]=amap("operator_center_mean",zre)
    methods["ConditionIF_CALREF"]=base._apply_cal_map(q_if_map,operator_score[ncal:][ae])
    fitif=_fit_learned_brier(np.column_stack((zrc,np.log(vrc))),tc,[-1,1],
        ["z_center","log_v_IF"])
    methods["Learned_IF_2D_zcenter_logv"]=_predict_learned(fitif,np.column_stack((zre,np.log(vre))))
    params={"Learned_2D_z0_zbar":fit2,"Learned_3D_z0_zbar_logv":fit3,
        "Learned_IF_2D_zcenter_logv":fitif,"ConditionIF_CALREF_map_reused":q_if_map}
    return methods,params,{"tcal":tc,"audit":audit,"qmargin":methods["Margin_CALREF"],
        "sigma2":sigma2,"eval_alert_n":int(ae.sum()),"cal_alert_n":int(ac.sum()),
        "new_map_fits":0,"learned_fits":3,"if_variance_eval_mean":float(vre.mean()),
        "operator_center_eval_mean":float(zre.mean())}


def s2_learned_baselines():
    out=OUT/"s2_learned_baselines"
    out.mkdir(parents=True,exist_ok=False)
    metrics=[];params_rows=[];case_rows=[];learned_fits=0;new_if_maps=0
    # Adult P2: 3 banks x 3 saved operations.
    for bank in range(3):
        for operation in ("alpha_0","alpha_0_5","bootstrap"):
            methods,params,meta=_adult_s2_case(bank,operation)
            for method,q in methods.items():
                metrics.append(_eval_metrics_row("Adult_P2_Logistic",bank,operation,method,q,meta["audit"],meta["qmargin"]))
            for method,fitrow in params.items():
                params_rows.append({"dataset":"Adult_P2_Logistic","bank":bank,"operation":operation,
                                    "method":method,"fit":fitrow})
            learned_fits+=meta["learned_fits"]
            case_rows.append({"dataset":"Adult_P2_Logistic","bank":bank,"operation":operation,
                "CAL_alert_n":meta["cal_alert_n"],"EVAL_alert_n":meta["eval_alert_n"],
                "sigma_g_sq_CAL":meta["sigma2"],"CALREF_mean":float(meta["tcal"].mean()),
                "new_condition_IF_maps":0,"learned_Brier_fits":meta["learned_fits"],
                "CALREF_parameter_fit_artifacts":meta["learned_fits"]})
    # Diabetes LGB 33.6k: 6-INT learned baselines.
    for family in ("LGB_33k","Logistic_33k"):
        for bank in range(3):
            methods,params,meta=_load_diabetes_case(family,bank)
            dataset="Diabetes_LGB_33k" if family=="LGB_33k" else "Diabetes_Logistic_33k"
            for method,q in methods.items():
                metrics.append(_eval_metrics_row(dataset,bank,"bootstrap",method,q,meta["audit"],meta["qmargin"]))
            for method,fitrow in params.items():
                params_rows.append({"dataset":dataset,"bank":bank,"operation":"bootstrap",
                                    "method":method,"fit":fitrow})
            learned_fits+=meta["learned_fits"]
            new_if_maps+=meta["new_map_fits"]
            case_rows.append({"dataset":dataset,"bank":bank,"operation":"bootstrap",
                "CAL_alert_n":meta["cal_alert_n"],"EVAL_alert_n":meta["eval_alert_n"],
                "sigma_g_sq_CAL":meta["sigma2"],"CALREF_mean":float(meta["tcal"].mean()),
                "IF_variance_CAL_mean":meta["v_if_cal_mean"],"IF_variance_EVAL_mean":meta["v_if_eval_mean"],
                "new_condition_IF_maps":meta["new_map_fits"],"learned_Brier_fits":meta["learned_fits"],
                "CALREF_parameter_fit_artifacts":meta["learned_fits"]+meta["new_map_fits"]})
    _write_csv(out/"metrics_by_bank.csv",metrics)
    _write_json(out/"learned_calibrator_parameters.json",{"ridge_l2":LEARNED_L2,
        "fit_target":"CALREF model flip frequency on CAL alert rows only",
        "feature_scaling":"mean and standard deviation fitted on CAL alert rows only",
        "direction_constraints":"risk logit nonincreasing in z0/zbar/zcenter and nondecreasing in log variance",
        "fits":params_rows})
    _write_csv(out/"case_counts.csv",case_rows)
    valcols=["R_070","AURC100","AUC","Brier","ECE10","delta_Brier_vs_Margin"]
    agg=[]
    frame=pd.DataFrame(metrics)
    for key,grp in frame.groupby(["dataset","operation","method"],dropna=False,sort=True):
        row={"dataset":key[0],"operation":key[1],"method":key[2],"banks":len(grp)}
        for c in valcols:
            x=grp[c].astype(float).to_numpy()
            row[c+"_mean"]=float(np.mean(x));row[c+"_median"]=float(np.median(x));row[c+"_min"]=float(np.min(x));row[c+"_max"]=float(np.max(x))
        agg.append(row)
    _write_csv(out/"mean_median_range.csv",agg)
    _write_json(out/"report.json",{"status":"S2_CALREF_LEARNED_BASELINES_COMPLETE",
        "case_count":len(case_rows),"metric_rows":len(metrics),"learned_Brier_fits":learned_fits,
        "new_condition_IF_1D_CALREF_maps":new_if_maps,"model_fits":0,
        "retained_run_model_score_matrix_batches":36,
        "retained_CALREF_parameter_fit_artifacts":learned_fits+new_if_maps,
        "CALREF_only_fit":True,"AUDIT_EVAL_labels_used_for_fitting":False,
        "ridge_l2":LEARNED_L2,"directions":"z margin coefficients <=0; log variance coefficient >=0",
        "limitations":["P1 LightGBM banks were not included; user-requested low-dimensional S2 cases are Adult P2 and Diabetes 33.6k LGB/Logistic.",
            "Banks share evaluation cohorts; bank ranges are training-bank sensitivity, not independent-cohort intervals."]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["s1-bank0", "s1-other-banks", "s2-learned"])
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        aff = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(aff)})
    if args.stage == "s1-bank0":
        s1_bank0()
    elif args.stage == "s1-other-banks":
        s1_other_banks()
    elif args.stage == "s2-learned":
        s2_learned_baselines()


if __name__ == "__main__":
    main()
