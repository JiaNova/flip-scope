#!/usr/bin/env python3
"""Cache-only C1/C2 analysis and V2.8 paper package.

No prediction model fits are performed.  C1 adds only the two requested
low-dimensional CALREF score maps.  Credit alert identities are reconstructed
from public IDs/features and saved D0 coefficients; EVAL outcomes are never
selected or read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "reports/core"
OUT = BASE / "paper_evidence"
U1 = BASE / "S1_S4_20261111/U1_POOL_UPDATE_20261112"
D1 = BASE / "S1_S4_20261111/D1_D2_D3_20261111"
DIAB_SPLIT = ROOT / "reports/public_benchmarks/diabetes_data_audit/selected_patient_split_manifest.csv"
CREDIT_ROOT = BASE / "R1_R4_20261112/R4/R4_credit_single_shot_v1"
CREDIT_DATA = ROOT / "data/public/credit/default_credit_data.csv"
CREDIT_REVIEW = BASE / "R1_R4_20261112/review"
OPS = ("Bootstrap", "Op_Half_Replace", "Op_Full_Replace")
PAIR_SPECS = {
    "Pair_1_IF_minus_Margin": ("OpCenter_IF", "Margin"),
    "Pair_2_IF_minus_OpCenter_Mean": ("OpCenter_IF", "OpCenter_Mean"),
    "Pair_3_IF_minus_Tuned_INT6_3D": ("OpCenter_IF", "Tuned_INT6_3D"),
}
METRICS = ("R_070", "AURC100", "Brier", "ECE10")
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 2026111301


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing/unsafe input: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing/unsafe JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing/unsafe NPZ: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _json_write(path: Path, value) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        raise FileExistsError(f"temporary output exists: {tmp}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _csv_write(path: Path, rows: list[dict]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status"]
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        raise FileExistsError(f"temporary output exists: {tmp}")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _npz_write(path: Path, **arrays) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    if tmp.exists() or tmp.is_symlink():
        raise FileExistsError(f"temporary output exists: {tmp}")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _finite_arrays(**named) -> None:
    for name, value in named.items():
        arr = np.asarray(value)
        if not np.isfinite(arr).all():
            raise ValueError(f"nonfinite cached values: {name}")


def _ece10(q: np.ndarray, t: np.ndarray, weights: np.ndarray | None = None) -> float:
    q = np.asarray(q, float); t = np.asarray(t, float)
    w = np.ones(len(q), float) if weights is None else np.asarray(weights, float)
    if q.ndim != 1 or t.shape != q.shape or w.shape != q.shape or np.any(w < 0):
        raise ValueError("ECE inputs must be aligned with nonnegative row weights")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("ECE requires positive total row weight")
    bins = np.minimum((q * 10.0).astype(np.int64), 9)
    qw = np.bincount(bins, weights=w * q, minlength=10)
    tw = np.bincount(bins, weights=w * t, minlength=10)
    nw = np.bincount(bins, weights=w, minlength=10)
    occupied = nw > 0
    return float(np.sum(nw[occupied] * np.abs(qw[occupied] / nw[occupied] - tw[occupied] / nw[occupied])) / total)


def _fit_score_map(score: np.ndarray, target: np.ndarray) -> dict:
    """Fit q=expit(a+b*s_std), with the required nonincreasing risk direction."""
    score = np.asarray(score, float); target = np.asarray(target, float)
    if score.ndim != 1 or target.shape != score.shape or len(score) < 2:
        raise ValueError("score-map inputs must be aligned vectors")
    _finite_arrays(score=score, target=target)
    if np.any((target < 0) | (target > 1)):
        raise ValueError("CALREF target must lie in [0,1]")
    center = float(np.mean(score)); scale = float(np.std(score, ddof=0))
    if scale <= 1e-15:
        scale = 1.0
    x = (score - center) / scale
    mean_target = float(np.clip(np.mean(target), 1e-8, 1 - 1e-8))
    w0 = np.asarray([math.log(mean_target / (1 - mean_target)), 0.0])

    def objective(w):
        p = expit(np.clip(w[0] + w[1] * x, -40, 40))
        e = p - target
        g = 2.0 * e * p * (1 - p)
        return float(np.mean(e * e)), np.asarray([g.mean(), np.mean(g * x)])

    result = minimize(objective, w0, jac=True, method="L-BFGS-B",
                      bounds=[(None, None), (None, 0.0)],
                      options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 50})
    loss, grad = objective(result.x)
    if not np.isfinite(result.x).all() or not np.isfinite(loss) or np.max(np.abs(grad)) > 1e-5:
        raise RuntimeError(f"CAL score mapping did not converge: {result.message}; grad={np.max(np.abs(grad))}")
    return {"center": center, "scale": scale, "intercept": float(result.x[0]),
            "slope": float(result.x[1]), "brier_loss": loss,
            "gradient_inf": float(np.max(np.abs(grad))), "iterations": int(result.nit),
            "converged": bool(result.success), "slope_constraint": "b <= 0"}


def _apply_score_map(mapping: dict, score: np.ndarray) -> np.ndarray:
    x = (np.asarray(score, float) - mapping["center"]) / mapping["scale"]
    return expit(np.clip(mapping["intercept"] + mapping["slope"] * x, -40, 40))


def _apply_saved_map(mapping: dict, score: np.ndarray) -> np.ndarray:
    co = np.asarray(mapping["coefficients"], float)
    return expit(np.clip(co[0] + co[1] * (np.asarray(score, float) - mapping["center"]) / mapping["scale"], -40, 40))


def _metric_row(q: np.ndarray, audit: np.ndarray) -> dict:
    q = np.asarray(q, float); audit = np.asarray(audit)
    if (q.ndim != 1 or audit.ndim != 2 or audit.shape[0] != len(q) or not len(q)
            or audit.shape[1] == 0 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1))):
        raise ValueError("metric inputs have incompatible shapes")
    if not np.isin(audit, [0, 1]).all():
        raise ValueError("AUDIT columns must be binary")
    audit = audit.astype(np.int8)
    t = audit.mean(axis=1)
    from experiments.adult_support.run_p1_instance_transfer import compute_tie_aware_risk
    r70 = compute_tie_aware_risk(q, t, 0.70)
    aurc = float(np.mean([compute_tie_aware_risk(q, t, k / 100.0) for k in range(1, 101)]))
    auc = float(roc_auc_score(audit.ravel(), np.repeat(q, audit.shape[1])))
    return {"n_alerts": int(len(q)), "R_070": r70, "AURC100": aurc, "AUC": auc,
            "Brier_finite_AUDIT": float(np.mean((q[:, None] - audit) ** 2)), "ECE10": _ece10(q, t),
            "mean_flip_rate": float(t.mean())}


def _risk_vector_from_groups(cum_w: np.ndarray, cum_t: np.ndarray, group_w: np.ndarray,
                             group_t: np.ndarray, total: float, coverages: np.ndarray) -> np.ndarray:
    out = np.empty(len(coverages), float)
    for ix, c in enumerate(coverages):
        mass = float(c) * total
        j = int(np.searchsorted(cum_w, mass, side="left"))
        prev_w = 0.0 if j == 0 else float(cum_w[j - 1])
        prev_t = 0.0 if j == 0 else float(cum_t[j - 1])
        if j >= len(group_w) or group_w[j] <= 0:
            raise RuntimeError("weighted tie boundary is invalid")
        frac = (mass - prev_w) / float(group_w[j])
        out[ix] = (prev_t + frac * float(group_t[j])) / mass
    return out


class _WeightedRisk:
    def __init__(self, q: np.ndarray):
        self.q = np.asarray(q, float)
        self.unique, self.group = np.unique(self.q, return_inverse=True)

    def risks(self, t: np.ndarray, weights: np.ndarray, coverages: np.ndarray) -> np.ndarray:
        t = np.asarray(t, float); weights = np.asarray(weights, float)
        if t.shape != self.q.shape or weights.shape != self.q.shape or np.any(weights < 0):
            raise ValueError("weighted risk vectors are misaligned")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("weighted risk requires positive mass")
        gw = np.bincount(self.group, weights=weights, minlength=len(self.unique))
        gt = np.bincount(self.group, weights=weights * t, minlength=len(self.unique))
        return _risk_vector_from_groups(np.cumsum(gw), np.cumsum(gt), gw, gt, total, coverages)

    def metrics(self, t: np.ndarray, weights: np.ndarray) -> dict:
        coverages = np.arange(1, 101, dtype=float) / 100.0
        risks = self.risks(t, weights, coverages)
        return {"R_070": float(risks[69]), "AURC100": float(np.mean(risks)),
                "ECE10": _ece10(self.q, np.asarray(t, float), weights)}


def _paired_delta(method_metrics: dict, baseline_metrics: dict, q_m: np.ndarray,
                  q_b: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    total = float(np.sum(w))
    brier_delta = float(np.sum(w * ((q_m - t) ** 2 - (q_b - t) ** 2)) / total)
    return {"R_070": method_metrics["R_070"] - baseline_metrics["R_070"],
            "AURC100": method_metrics["AURC100"] - baseline_metrics["AURC100"],
            "Brier": brier_delta,
            "ECE10": method_metrics["ECE10"] - baseline_metrics["ECE10"]}


def _read_diabetes_eval_ids() -> np.ndarray:
    if DIAB_SPLIT.is_symlink() or not DIAB_SPLIT.is_file():
        raise RuntimeError("Diabetes identity manifest missing/unsafe")
    # Column selection avoids materializing outcome strings in DictReader rows.
    identity = pd.read_csv(DIAB_SPLIT, usecols=["patient_key_sha256", "split"], dtype=str)
    keys = identity.loc[identity["split"] == "EVAL", "patient_key_sha256"].tolist()
    if len(keys) != 13949 or len(set(keys)) != len(keys):
        raise RuntimeError("Diabetes EVAL identity sequence is incomplete or duplicated")
    return np.asarray(keys, dtype="U64")


def _diabetes_bootstrap_cases(eval_ids: np.ndarray) -> dict:
    scheme = D1 / "Scheme_D_Merged_Rare50_Unscaled"
    extension = D1 / "D2_lambda_boundary_extension"
    d2fit_path = extension / "Scheme_D_Merged_Rare50_Unscaled_bank{bank}_fits.json"
    cases = {}
    for bank in range(6):
        bdir = scheme / f"bank{bank}"
        pred = _read_npz(bdir / "predictions.npz")
        ifa = _read_npz(bdir / "if_arrays.npz")
        maps = _read_json(bdir / "d1_calref_maps.json")
        z0 = pred["z0e"]; zi = pred["zie"]; audit_z = pred["zaudit"]
        if (len(z0) != len(eval_ids) or zi.shape != (len(eval_ids), 6) or
                audit_z.shape[0] != len(eval_ids) or audit_z.shape[1] < 11):
            raise RuntimeError(f"D1 bootstrap prediction shape mismatch, bank={bank}")
        mask = ifa["eval_alert_mask"].astype(bool)
        if not np.array_equal(mask, z0 >= 0.0):
            raise RuntimeError(f"D1 saved alert mask differs from z0, bank={bank}")
        idx = np.flatnonzero(mask)
        audit = (audit_z[idx, :11] < 0).astype(np.int8)
        zbar = zi.mean(axis=1)
        svar = zi.var(axis=1, ddof=1)
        perf = _read_json(bdir / "d1_performance.json")
        rows = [r for r in perf["rows"] if r["method"] == "Margin"]
        sigma2_values = [r["sigma_g_sq"] for r in perf["rows"] if r["method"] == "Margin"]
        if not sigma2_values:
            raise RuntimeError("D1 performance cache lacks sigma_g_sq")
        sigma2 = float(sigma2_values[0])
        vtilde = 0.5 * svar + 0.5 * sigma2
        fusion = 0.5 * z0 + 0.5 * zbar
        v_if = ifa["v_if_eval_alert"]
        if len(v_if) != len(idx):
            raise RuntimeError(f"D1 IF cache is not alert-aligned, bank={bank}")
        q = {
            "Margin": _apply_saved_map(maps["Margin"], z0[idx]),
            "OpCenter_Mean": _apply_saved_map(maps["Margin"], z0[idx]),
            "OpCenter_IF": _apply_saved_map(maps["ConditionalIF"], z0[idx] / np.sqrt(np.maximum(v_if, 1e-300))),
            "qMV": _apply_saved_map(maps["qMV"], fusion[idx] / np.sqrt(np.maximum(vtilde[idx], 1e-300))),
        }
        fit_items = _read_json(Path(d2fit_path.as_posix().format(bank=bank)))
        tuned = next(r["fit"] for r in fit_items if r["learner"] == "Learned3D_MV_Unconstrained_CV")
        x3 = np.column_stack([z0[idx], zbar[idx], np.log(np.maximum(vtilde[idx], 1e-300))])
        xs = (x3 - np.asarray(tuned["scaler_mean"])) / np.asarray(tuned["scaler_scale"])
        q["Tuned_INT6_3D"] = expit(np.clip(tuned["intercept"] + xs @ np.asarray(tuned["coefficients"]), -40, 40))
        for name, vals in q.items():
            if vals.shape != (len(idx),):
                raise RuntimeError(f"D1 q cache reconstruction misaligned: bank={bank} {name}")
            _finite_arrays(**{name: vals})
        cases[bank] = {"alert_idx": idx, "audit": audit, "q": q,
                       "source": "D1 Scheme D full EVAL arrays + persisted D2 selected 3D CAL fit",
                       "audit_columns_used": 11, "n_eval": len(eval_ids)}
    return cases


def _diabetes_update_cases(eval_ids: np.ndarray, operation: str) -> dict:
    cases = {}
    for bank in range(6):
        bdir = U1 / f"bank{bank}"
        pred_path = bdir / operation / "prediction_arrays.npz"
        pred = _read_npz(pred_path)
        prospective = _read_npz(bdir / "prospective/prospective_scores.npz")
        mask = prospective["eval_alert_mask"].astype(bool)
        idx = np.flatnonzero(mask)
        if len(idx) != len(pred["audit_eval_alert"]) or len(eval_ids) != len(mask):
            raise RuntimeError(f"U1 EVAL alert identity alignment failed bank={bank} op={operation}")
        q = {
            "Margin": pred["q_Margin_D0_1D"],
            "OpCenter_Mean": pred["q_OpCenter_Mean_1D"],
            "OpCenter_IF": pred["q_OpCenter_IF_1D"],
            "Tuned_INT6_3D": pred["q_Update_INT6_Tuned3D_MV"],
            "qMV": pred["q_Update_INT6_qMV"],
        }
        for name, vals in q.items():
            if vals.shape != (len(idx),):
                raise RuntimeError(f"U1 q prediction alignment failed bank={bank} {name}")
            _finite_arrays(**{name: vals})
        cases[bank] = {"alert_idx": idx, "audit": pred["audit_eval_alert"].astype(np.int8),
                       "q": q, "source": str(pred_path.relative_to(ROOT)),
                       "audit_columns_used": int(pred["audit_eval_alert"].shape[1]),
                       "n_eval": len(eval_ids)}
    return cases


def _credit_eval_ids_and_alert_indices() -> tuple[np.ndarray, dict, dict]:
    if CREDIT_DATA.is_symlink() or not CREDIT_DATA.is_file():
        raise RuntimeError("Credit public feature input needed for identity reconstruction is missing")
    # Select only ID and predictors; do not load Y at all.
    numeric = ["X1", "X5"] + [f"X{i}" for i in range(12, 24)]
    categorical = ["X2", "X3", "X4"] + [f"X{i}" for i in range(6, 12)]
    usecols = ["ID", *numeric, *categorical]
    frame = pd.read_csv(CREDIT_DATA, usecols=usecols)
    ids = frame["ID"].astype(int).to_numpy()
    if len(ids) != 30000 or len(np.unique(ids)) != len(ids):
        raise RuntimeError("Credit public IDs do not match the locked 30,000-row input")
    ordered = sorted(ids.tolist(), key=lambda x: (hashlib.sha256(f"R4_SPLIT_V1:{x}".encode()).digest(), x))
    eval_ids = set(ordered[24000:])
    eval_frame = frame[frame.ID.isin(eval_ids)].reset_index(drop=True)
    ids_eval = eval_frame.ID.astype(int).to_numpy()
    if len(ids_eval) != 6000 or len(np.unique(ids_eval)) != 6000:
        raise RuntimeError("Credit EVAL row-key recovery failed")
    positions = {int(v): i for i, v in enumerate(ids_eval)}
    alert_indices = {}; prediction_check = {}
    feature_inference_batches = 0
    for bank in range(6):
        bdir = CREDIT_ROOT / f"bank{bank}"
        prep = _read_json(bdir / "preprocessor.json")
        d0 = _read_json(bdir / "D0.json")
        manifest = _read_json(bdir / "bank_manifest.json")
        names = prep["feature_names"]
        colidx = {name: j for j, name in enumerate(names)}
        x = np.zeros((len(eval_frame), len(names)), dtype=np.float64)
        x[:, colidx["__INTERCEPT__"]] = 1.0
        for col in numeric:
            values = eval_frame[col].to_numpy(float)
            x[:, colidx[col]] = (values - float(prep["numeric_means"][col])) / float(prep["numeric_scales"][col])
        for col, vocab in zip(categorical, prep["vocabularies"]):
            mapping = prep["categorical_maps"][col]["mapping"]
            values = eval_frame[col].astype(int).astype(str).map(lambda v: mapping.get(v, "<RARE>")).to_numpy()
            for level in vocab:
                x[values == level, colidx[f"{col}={level}"]] = 1.0
        beta = np.asarray(d0["beta"], float)
        if x.shape[1] != len(beta) or len(beta) != int(manifest["feature_dimension"]):
            raise RuntimeError(f"Credit feature dimension mismatch bank={bank}")
        z0 = x @ beta - float(manifest["gamma0"])
        mask = z0 >= 0.0
        idx = np.flatnonzero(mask)
        # Exact identity check: the selected rows must reproduce the saved alert logits.
        cache = _read_npz(bdir / "Bootstrap/prediction_arrays.npz")
        saved = cache["z0_eval_alert"]
        if len(idx) != len(saved) or not np.allclose(z0[idx], saved, rtol=0, atol=2e-10):
            raise RuntimeError(f"Credit D0 alert identity reconstruction differs from cache bank={bank}")
        alert_indices[bank] = idx
        prediction_check[bank] = {"alert_count": int(len(idx)),
                                  "max_abs_z0_cache_difference": float(np.max(np.abs(z0[idx] - saved))),
                                  "D0_EVAL_identity_inference_batches": 1}
        feature_inference_batches += 1
    return ids_eval, alert_indices, {"D0_EVAL_identity_inference_batches": feature_inference_batches,
                                    "EVAL_label_columns_loaded": 0,
                                    "credit_source_sha256": _sha_file(CREDIT_DATA),
                                    "bank_checks": prediction_check}


def _credit_cases(eval_ids: np.ndarray, alert_indices: dict) -> dict:
    cases = {op: {} for op in OPS}
    for bank in range(6):
        for op in OPS:
            folder = CREDIT_ROOT / f"bank{bank}" / op
            arr = _read_npz(folder / "prediction_arrays.npz")
            names = [str(v) for v in arr["scored_method_names"].tolist()]
            scored = arr["scored"]
            if scored.shape != (len(arr["audit_eval_alert"]), len(names)):
                raise RuntimeError(f"Credit scored matrix/name mismatch bank={bank} op={op}")
            qcol = {name: scored[:, j] for j, name in enumerate(names)}
            required = {"Margin_D0_1D", "OpCenter_IF_1D", "OpCenter_Mean_1D", "Update_INT6_Tuned3D_MV"}
            if not required.issubset(qcol):
                raise RuntimeError(f"Credit saved q predictions missing: bank={bank} op={op} missing={required-set(qcol)}")
            idx = alert_indices[bank]
            if len(idx) != len(arr["audit_eval_alert"]):
                raise RuntimeError(f"Credit alert ID order/length mismatch bank={bank} op={op}")
            q = {"Margin": qcol["Margin_D0_1D"], "OpCenter_Mean": qcol["OpCenter_Mean_1D"],
                 "OpCenter_IF": qcol["OpCenter_IF_1D"], "Tuned_INT6_3D": qcol["Update_INT6_Tuned3D_MV"],
                 "qMV": qcol["Update_INT6_qMV"] if "Update_INT6_qMV" in qcol else np.full(len(idx), np.nan)}
            for name in ("Margin", "OpCenter_Mean", "OpCenter_IF", "Tuned_INT6_3D"):
                if q[name].shape != (len(idx),):
                    raise RuntimeError(f"Credit q row order mismatch bank={bank} op={op} method={name}")
                _finite_arrays(**{name: q[name]})
            cases[op][bank] = {"alert_idx": idx, "audit": arr["audit_eval_alert"].astype(np.int8),
                               "q": q, "source": str((folder / "prediction_arrays.npz").relative_to(ROOT)),
                               "audit_columns_used": int(arr["audit_eval_alert"].shape[1]),
                               "n_eval": len(eval_ids)}
    return cases


def _c1_run() -> tuple[list[dict], list[dict], list[dict], dict]:
    rows = []; contrasts = []; target_effects = []; maps_used = {}
    fits = 0; q_files = 0; max_rank_target_diff = 0.0
    eval_ids = _read_diabetes_eval_ids()
    for bank in range(6):
        bdir = U1 / f"bank{bank}"
        prospective = _read_npz(bdir / "prospective/prospective_scores.npz")
        for op in ("Op_Half_Replace", "Op_Full_Replace"):
            arr = _read_npz(bdir / op / "prediction_arrays.npz")
            z0c = arr["z0_cal_alert"]; zRc = arr["zR_cal_alert"]
            z0e = arr["z0_eval_alert"]; zRe = arr["zR_eval_alert"]
            v0c = prospective["v0_cal"]; v0e = prospective["v0_eval_alert"]
            vRc = arr["vR_cal_alert"]; vRe = arr["vR_eval_alert"]
            t_old = prospective["tcal_poolA"]; t_new = arr["tcal_update"]
            audit = arr["audit_eval_alert"].astype(np.int8)
            if not (len(z0c) == len(t_old) == len(t_new) == len(v0c) == len(vRc)):
                raise RuntimeError(f"C1 CAL cohort alignment failed bank={bank} op={op}")
            if not (len(z0e) == len(zRe) == len(v0e) == len(vRe) == len(audit)):
                raise RuntimeError(f"C1 EVAL alert alignment failed bank={bank} op={op}")
            if np.any(v0c < 0) or np.any(v0e < 0) or np.any(vRc < 0) or np.any(vRe < 0):
                raise RuntimeError("C1 variance cache contains negative values")
            cal_scores = {
                "z0_div_1": z0c,
                "z0_div_sqrt_v0": z0c / np.sqrt(np.maximum(v0c, 1e-300)),
                "z0_div_sqrt_vR": z0c / np.sqrt(np.maximum(vRc, 1e-300)),
                "zR_div_1": zRc,
                "zR_div_sqrt_v0": zRc / np.sqrt(np.maximum(v0c, 1e-300)),
                "zR_div_sqrt_vR": zRc / np.sqrt(np.maximum(vRc, 1e-300)),
            }
            eval_scores = {
                "z0_div_1": z0e,
                "z0_div_sqrt_v0": z0e / np.sqrt(np.maximum(v0e, 1e-300)),
                "z0_div_sqrt_vR": z0e / np.sqrt(np.maximum(vRe, 1e-300)),
                "zR_div_1": zRe,
                "zR_div_sqrt_v0": zRe / np.sqrt(np.maximum(v0e, 1e-300)),
                "zR_div_sqrt_vR": zRe / np.sqrt(np.maximum(vRe, 1e-300)),
            }
            q_payload = {}
            score_metrics = {}
            for score_name in cal_scores:
                q_payload[score_name] = {}
                score_metrics[score_name] = {}
                for target_name, target in (("Old_Target_Calibrated", t_old), ("New_Target_Calibrated", t_new)):
                    mapping = _fit_score_map(cal_scores[score_name], target)
                    q = _apply_score_map(mapping, eval_scores[score_name])
                    fits += 1
                    m = _metric_row(q, audit)
                    q_payload[score_name][target_name] = q
                    score_metrics[score_name][target_name] = m
                    rows.append({"dataset": "Diabetes_Scheme_D", "bank": bank, "operation": op,
                                 "score": score_name, "calibration_target": target_name,
                                 "cal_n": int(len(target)), "eval_alert_n": int(len(audit)),
                                 "score_CAL_mean": mapping["center"], "score_CAL_sd": mapping["scale"],
                                 "map_intercept": mapping["intercept"], "map_slope": mapping["slope"],
                                 "map_brier_loss_CAL": mapping["brier_loss"],
                                 "map_gradient_inf": mapping["gradient_inf"],
                                 "map_iterations": mapping["iterations"], "map_converged": mapping["converged"], **m})
                    maps_used[f"bank{bank}/{op}/{score_name}/{target_name}"] = mapping
                for key in ("R_070", "AURC100", "AUC"):
                    old_value = score_metrics[score_name]["Old_Target_Calibrated"][key]
                    new_value = score_metrics[score_name]["New_Target_Calibrated"][key]
                    diff = float(new_value - old_value)
                    max_rank_target_diff = max(max_rank_target_diff, abs(diff))
                    target_effects.append({"bank": bank, "operation": op, "score": score_name,
                                           "metric": key, "old_target": old_value,
                                           "new_target": new_value, "new_minus_old": diff,
                                           "interpretation": "should be invariant when both fitted slopes are strictly negative"})
                for key in ("Brier_finite_AUDIT", "ECE10"):
                    old_value = score_metrics[score_name]["Old_Target_Calibrated"][key]
                    new_value = score_metrics[score_name]["New_Target_Calibrated"][key]
                    target_effects.append({"bank": bank, "operation": op, "score": score_name,
                                           "metric": key, "old_target": old_value,
                                           "new_target": new_value, "new_minus_old": float(new_value - old_value),
                                           "interpretation": "matched score and CAL standardization; target-source calibration contrast"})
            # New-target-calibrated pure contrasts: center changes hold the denominator fixed;
            # variance changes hold the center fixed.
            contrast_pairs = [
                ("Center_at_Div1_zR_minus_z0", "z0_div_1", "zR_div_1"),
                ("Center_at_Div_vR_zR_minus_z0", "z0_div_sqrt_vR", "zR_div_sqrt_vR"),
                ("Variance_at_zR_vR_minus_1", "zR_div_1", "zR_div_sqrt_vR"),
                ("Variance_update_at_zR_vR_minus_v0", "zR_div_sqrt_v0", "zR_div_sqrt_vR"),
            ]
            for label, base_score, changed_score in contrast_pairs:
                for metric in ("R_070", "AURC100", "AUC", "Brier_finite_AUDIT", "ECE10"):
                    lhs = score_metrics[base_score]["New_Target_Calibrated"][metric]
                    rhs = score_metrics[changed_score]["New_Target_Calibrated"][metric]
                    contrasts.append({"bank": bank, "operation": op, "contrast": label,
                                      "metric": metric, "base_score": base_score,
                                      "changed_score": changed_score, "base_value": lhs,
                                      "changed_value": rhs, "changed_minus_base": float(rhs - lhs),
                                      "calibration_target": "New_Target_Calibrated"})
            q_path = OUT / "c1_recalibrated" / f"bank{bank}_{op}.npz"
            arrays = {f"{score}__{target}": q for score, targets in q_payload.items() for target, q in targets.items()}
            _npz_write(q_path, **arrays); q_files += 1
    if fits != 144:
        raise RuntimeError(f"C1 expected exactly 144 low-dimensional mapping fits; got {fits}")
    return rows, contrasts, target_effects, {"C1_mapping_fit_calls": fits,
            "C1_prediction_model_fit_calls": 0, "C1_q_npz_files": q_files,
            "max_abs_target_invariance_diff_R_AURC_AUC": max_rank_target_diff,
            "maps": maps_used}


def _make_datasets() -> tuple[dict, dict, dict]:
    diabetes_ids = _read_diabetes_eval_ids()
    diab_cases = {"Bootstrap": _diabetes_bootstrap_cases(diabetes_ids)}
    for op in ("Op_Half_Replace", "Op_Full_Replace"):
        diab_cases[op] = _diabetes_update_cases(diabetes_ids, op)
    credit_ids, credit_idx, credit_identity_audit = _credit_eval_ids_and_alert_indices()
    credit_cases = _credit_cases(credit_ids, credit_idx)
    return ({"Diabetes_Scheme_D": {"ids": diabetes_ids, "operations": diab_cases},
             "UCI_Credit": {"ids": credit_ids, "operations": credit_cases}},
            {"Diabetes_Scheme_D": _sha_bytes("\n".join(diabetes_ids.tolist()).encode()),
             "UCI_Credit": _sha_bytes("\n".join(map(str, credit_ids.tolist())).encode())},
            {"credit_identity_reconstruction": credit_identity_audit,
             "diabetes_identity_manifest_sha256": _sha_file(DIAB_SPLIT)})


def _point_tables(datasets: dict) -> tuple[list[dict], list[dict], list[dict]]:
    perbank = []; paired = []; events = []
    for dataset, data in datasets.items():
        for op, cases in data["operations"].items():
            flip_n = 0; denom = 0
            for bank, case in cases.items():
                idx = case["alert_idx"]; audit = case["audit"]
                if case["n_eval"] != len(data["ids"]):
                    raise RuntimeError(f"point table EVAL count mismatch {dataset}/{op}/bank{bank}")
                if len(idx) != audit.shape[0]:
                    raise RuntimeError(f"point table alert/audit row mismatch {dataset}/{op}/bank{bank}")
                flip_n += int(audit.sum()); denom += int(audit.size)
                for method, q in case["q"].items():
                    if not np.isfinite(q).all():
                        continue
                    metrics = _metric_row(q, audit)
                    perbank.append({"dataset": dataset, "operation": op, "bank": bank,
                                    "method": method, **metrics,
                                    "D0_alert_n": int(len(idx)), "audit_B": int(audit.shape[1]),
                                    "source": case["source"]})
                # Event-count rows are repeated by method to quantify what the selected cutoff retains.
                t = audit.mean(axis=1)
                from experiments.adult_support.run_p1_instance_transfer import compute_tie_aware_risk
                for method, q in case["q"].items():
                    if not np.isfinite(q).all():
                        continue
                    r = compute_tie_aware_risk(q, t, 0.70)
                    events.append({"dataset": dataset, "operation": op, "bank": bank,
                                   "method": method, "audit_flip_events_all_alerts": int(audit.sum()),
                                   "audit_flip_trials_all_alerts": int(audit.size),
                                   "full_alert_mean_flip_rate": float(t.mean()),
                                   "R70_expected_flip_mass": float(0.70 * len(q) * r),
                                   "R70_retained_alert_mass": float(0.70 * len(q)),
                                   "sparse_total_flip_events_lt20": bool(audit.sum() < 20)})
            events.append({"dataset": dataset, "operation": op, "bank": "ALL6",
                           "method": "ALL_METHODS_SHARED_AUDIT_TARGET", "audit_flip_events_all_alerts": flip_n,
                           "audit_flip_trials_all_alerts": denom,
                           "full_alert_mean_flip_rate": (flip_n / denom if denom else None),
                           "R70_expected_flip_mass": None, "R70_retained_alert_mass": None,
                           "sparse_total_flip_events_lt20": bool(flip_n < 20)})
            for bank in cases:
                case = cases[bank]
                qdict = case["q"]; audit = case["audit"]
                t = audit.mean(axis=1); w = np.ones(len(t), float)
                method_metrics = {}
                for method in set(sum(([a, b] for a, b in PAIR_SPECS.values()), [])):
                    if method not in qdict or not np.isfinite(qdict[method]).all():
                        continue
                    scorer = _WeightedRisk(qdict[method])
                    method_metrics[method] = scorer.metrics(t, w)
                for pair, (lhs, rhs) in PAIR_SPECS.items():
                    if lhs not in method_metrics or rhs not in method_metrics:
                        paired.append({"dataset": dataset, "operation": op, "bank": bank,
                                       "pair": pair, "status": "NOT_COMPUTABLE_MISSING_MATCHED_PREDICTION_CACHE"})
                        continue
                    delta = _paired_delta(method_metrics[lhs], method_metrics[rhs], qdict[lhs], qdict[rhs], t, w)
                    for metric, val in delta.items():
                        paired.append({"dataset": dataset, "operation": op, "bank": bank,
                                       "pair": pair, "method": lhs, "baseline": rhs,
                                       "metric": metric, "delta": val, "status": "COMPUTED_FROM_CACHED_ROWS"})
    # six-bank summaries use equal bank weights; the intervals are not population CIs.
    point_by_key = {}
    for row in paired:
        if "metric" not in row:
            continue
        point_by_key.setdefault((row["dataset"], row["operation"], row["pair"], row["metric"]), []).append(row["delta"])
    summary_rows = []
    for key, vals in point_by_key.items():
        arr = np.asarray(vals, float)
        summary_rows.append({"dataset": key[0], "operation": key[1], "pair": key[2], "metric": key[3],
                             "paired_mean": float(arr.mean()), "paired_min": float(arr.min()),
                             "paired_max": float(arr.max()), "sd_across_banks": float(arr.std(ddof=1)),
                             "wins_negative_out_of_n": int(np.sum(arr < 0)), "banks_n": int(len(arr)),
                             "interpretation": "descriptive training-bank variation; shared EVAL patients"})
    return perbank, paired, events, summary_rows


def _bootstrap_and_jackknife(datasets: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    replicate_rows = []; weight_digests = []; jackknife_rows = []; jackknife_summary = []
    bootstrap_summary = []
    coverages = np.arange(1, 101, dtype=float) / 100.0
    for d_ix, (dataset, data) in enumerate(datasets.items()):
        n_eval = len(data["ids"])
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, d_ix]))
        keys = [(op, pair, metric) for op in OPS for pair in PAIR_SPECS for metric in METRICS]
        # Missing pairs must remain missing, never uninitialized finite memory.
        samples = {key: np.full(BOOTSTRAP_REPS, np.nan) for key in keys}
        for m in range(BOOTSTRAP_REPS):
            counts = rng.multinomial(n_eval, np.full(n_eval, 1.0 / n_eval)).astype(np.uint32)
            weight_digests.append({"dataset": dataset, "replicate": m + 1,
                                   "patient_weight_sha256": _sha_bytes(counts.astype("<u4").tobytes()),
                                   "eval_patient_n": n_eval, "resampled_total": int(counts.sum()),
                                   "seed": BOOTSTRAP_SEED, "bit_generator": "PCG64 via numpy default_rng/SeedSequence"})
            for op in OPS:
                cases = data["operations"][op]
                bank_pair_deltas = {pair: {metric: [] for metric in METRICS} for pair in PAIR_SPECS}
                for bank, case in cases.items():
                    alert_idx = case["alert_idx"]
                    w = counts[alert_idx].astype(float)
                    if w.sum() <= 0:
                        raise RuntimeError("a patient bootstrap draw yielded zero current alerts")
                    audit = case["audit"]
                    t = audit.mean(axis=1)
                    metrics_by_method = {}
                    for method in set(sum(([a, b] for a, b in PAIR_SPECS.values()), [])):
                        q = case["q"].get(method)
                        if q is None or not np.isfinite(q).all():
                            continue
                        scorer = _WeightedRisk(q)
                        metrics_by_method[method] = scorer.metrics(t, w)
                    for pair, (lhs, rhs) in PAIR_SPECS.items():
                        if lhs not in metrics_by_method or rhs not in metrics_by_method:
                            continue
                        delta = _paired_delta(metrics_by_method[lhs], metrics_by_method[rhs],
                                              case["q"][lhs], case["q"][rhs], t, w)
                        for metric in METRICS:
                            bank_pair_deltas[pair][metric].append(delta[metric])
                for pair in PAIR_SPECS:
                    for metric in METRICS:
                        values = bank_pair_deltas[pair][metric]
                        if len(values) != 6:
                            raise RuntimeError(f"missing paired predictions in {dataset}/{op}/{pair}/{metric}")
                        avg = float(np.mean(values))
                        samples[(op, pair, metric)][m] = avg
                        replicate_rows.append({"dataset": dataset, "operation": op, "replicate": m + 1,
                                               "pair": pair, "metric": metric, "mean_delta_across_6_banks": avg,
                                               "banks_n": 6, "weight_sha256": weight_digests[-1]["patient_weight_sha256"]})
        for key, sample in samples.items():
            if not np.isfinite(sample).all() or len(sample) != BOOTSTRAP_REPS:
                continue
            bootstrap_summary.append({"dataset": dataset, "operation": key[0], "pair": key[1], "metric": key[2],
                                      "replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
                                      "bootstrap_mean": float(sample.mean()), "row_bootstrap_SE": float(sample.std(ddof=1)),
                                      "percentile_2_5": float(np.quantile(sample, .025)),
                                      "percentile_97_5": float(np.quantile(sample, .975)),
                                      "resampling_unit": "shared unique EVAL patient ID; reweight alert rows and recut ties each draw"})
        # AUDIT draws are independent across banks. Delete one bank's column
        # at a time and add within-bank variance contributions, never pair
        # unrelated columns merely because their ordinal indices match.
        for op in OPS:
            summaries, deletes = _stratified_column_jackknife(data["operations"][op], dataset, op)
            jackknife_summary.extend(summaries)
            jackknife_rows.extend(deletes)
    return bootstrap_summary, replicate_rows, weight_digests, jackknife_summary, jackknife_rows


def _stratified_jackknife_se(deleted_grand_means: np.ndarray) -> float:
    """Independent bank strata; input[k,b] is the grand mean deleting (k,b)."""
    values = np.asarray(deleted_grand_means, float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("finite bank-by-column deletion values required")
    b = values.shape[1]
    centered = values - values.mean(axis=1, keepdims=True)
    return float(np.sqrt((b-1)/b * np.sum(centered**2)))


def _stratified_column_jackknife(cases: dict, dataset: str, operation: str):
    banks = sorted(cases)
    k = len(banks)
    sizes = {cases[bank]["audit"].shape[1] for bank in banks}
    if k < 2 or len(sizes) != 1 or next(iter(sizes)) != 11:
        raise ValueError("this analysis requires >=2 banks each with exactly 11 AUDIT draws")
    b = 11
    pairs = list(PAIR_SPECS)
    points = np.empty((k, len(pairs), len(METRICS)))
    local_deleted = np.empty((k, b, len(pairs), len(METRICS)))
    for ix, bank in enumerate(banks):
        case = cases[bank]; q = case["q"]
        for drop in range(-1, b):
            audit = case["audit"] if drop == -1 else np.delete(case["audit"], drop, axis=1)
            target = audit.mean(axis=1); weights = np.ones(len(target))
            met = {name: _WeightedRisk(q[name]).metrics(target, weights)
                   for name in {v for pair in PAIR_SPECS.values() for v in pair}}
            for pi, pair in enumerate(pairs):
                lhs, rhs = PAIR_SPECS[pair]
                delta = _paired_delta(met[lhs], met[rhs], q[lhs], q[rhs], target, weights)
                values = [delta[metric] for metric in METRICS]
                if drop == -1: points[ix,pi] = values
                else: local_deleted[ix,drop,pi] = values
    grand_point = points.mean(axis=0)
    grand_deleted = grand_point[None,None,:,:] + (local_deleted-points[:,None,:,:])/k
    rows = []; summaries = []
    for pi, pair in enumerate(pairs):
        for mi, metric in enumerate(METRICS):
            values = grand_deleted[:,:,pi,mi]
            for ix, bank in enumerate(banks):
                for drop in range(b):
                    rows.append({"dataset": dataset, "operation": operation, "pair": pair, "metric": metric,
                                 "omitted_bank": bank, "omitted_AUDIT_column": drop+1,
                                 "mean_delta_across_6_banks": float(values[ix,drop]), "banks_n": k,
                                 "remaining_AUDIT_B_in_omitted_bank": b-1,
                                 "other_banks_AUDIT_B": b})
            summaries.append({"dataset": dataset, "operation": operation, "pair": pair, "metric": metric,
                              "LOAO_mean": float(values.mean()), "LOAO_min": float(values.min()),
                              "LOAO_max": float(values.max()), "jackknife_SE": _stratified_jackknife_se(values),
                              "B": b, "banks_n": k, "deletions_n": k*b,
                              "sensitivity_scope": "independent-bank stratified AUDIT delete-one; fixed predictions/EVAL; ECE SE is diagnostic"})
    return summaries, rows


def _table1(datasets: dict, perbank: list[dict], paired_summary: list[dict], boot_summary: list[dict],
            jack_summary: list[dict]) -> tuple[list[dict], list[dict]]:
    metrics = ("R_070", "AURC100", "AUC", "Brier_finite_AUDIT", "ECE10")
    summary = []
    for dataset, data in datasets.items():
        for op in OPS:
            rows = [r for r in perbank if r["dataset"] == dataset and r["operation"] == op]
            methods = sorted(set(r["method"] for r in rows))
            for method in methods:
                rr = [r for r in rows if r["method"] == method]
                row = {"dataset": dataset, "operation": op, "method": method,
                       "banks_n": len(rr), "n_alerts_mean": float(np.mean([x["n_alerts"] for x in rr]))}
                for metric in metrics:
                    vals = np.asarray([x[metric] for x in rr], float)
                    row[f"{metric}_mean"] = float(vals.mean())
                    row[f"{metric}_min"] = float(vals.min())
                    row[f"{metric}_max"] = float(vals.max())
                summary.append(row)
    # Preserve exact unadapted review outputs for Adult; no re-analysis or metric replacement.
    adult_rows = []
    adult_root = ROOT / "reports/adult_support/P1_EXECUTION_20261110"
    for cohort, filename in (("Adult_P1_development", "development_metrics.csv"),
                             ("Adult_P1_locked", "locked_validation_metrics.csv")):
        frame = pd.read_csv(adult_root / filename)
        for record in frame.to_dict("records"):
            adult_rows.append({"dataset": "Adult", "cohort": cohort, **record})
    p2 = pd.read_csv(BASE / "w1_p2/mean_range_metrics.csv")
    for record in p2.to_dict("records"):
        adult_rows.append({"dataset": "Adult", "cohort": record["task"], **record})
    # C2 sensitivities join onto the paired table; missing rows stay explicit.
    paired_final = []
    for row in paired_summary:
        ds, op, pair, metric = row["dataset"], row["operation"], row["pair"], row["metric"]
        boot = next((x for x in boot_summary if (x["dataset"], x["operation"], x["pair"], x["metric"]) == (ds, op, pair, metric)), None)
        jack = next((x for x in jack_summary if (x.get("dataset"), x.get("operation"), x.get("pair"), x.get("metric")) == (ds, op, pair, metric)), None)
        paired_final.append({**row,
                             "row_bootstrap_CI_2_5": boot.get("percentile_2_5") if boot else None,
                             "row_bootstrap_CI_97_5": boot.get("percentile_97_5") if boot else None,
                             "audit_LOAO_min": jack.get("LOAO_min") if jack else None,
                             "audit_LOAO_max": jack.get("LOAO_max") if jack else None,
                             "audit_LOAO_SE": jack.get("jackknife_SE") if jack else None})
    return summary, adult_rows, paired_final


def _copy_tables(out: Path) -> tuple[list[dict], list[dict]]:
    # Table 2: exact recorded D1 stage decomposition and independently reviewed Credit closure.
    d1var = pd.read_csv(D1 / "D1_variance_mean_range.csv").to_dict("records")
    credit_var = pd.read_csv(CREDIT_REVIEW / "R4_variance_replayed.csv")
    credit_rows = []
    for op, g in credit_var.groupby("operation", sort=False):
        for col in ("mean_IF_variance", "mean_AUDIT_variance", "AUDIT_over_IF_mean_ratio", "instance_correlation"):
            vals = g[col].to_numpy(float)
            credit_rows.append({"dataset": "UCI_Credit", "operation": op, "quantity": col,
                                "mean_across_6_banks": float(vals.mean()), "min_across_6_banks": float(vals.min()),
                                "max_across_6_banks": float(vals.max()), "banks_n": int(len(vals)),
                                "scope": "finite AUDIT variance closure; not a full representation decomposition"})
    # Table 4A: independent review-corrected clinical denominators.
    clinical = pd.read_csv(BASE / "S1_S4_20261111/review/s4_clinical_corrected.csv")
    clinical_summary = []
    group_cols = [c for c in ("B", "method", "rho", "coverage") if c in clinical.columns]
    for keys, g in clinical.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(group_cols, keys)); row["banks_n"] = int(g["bank"].nunique())
        for col in ("flip_R", "D0_true_positive_retention", "population_sensitivity", "PPV"):
            vals = g[col].to_numpy(float)
            row[f"{col}_mean"] = float(vals.mean()); row[f"{col}_min"] = float(vals.min()); row[f"{col}_max"] = float(vals.max())
        clinical_summary.append(row)
    # Table 4B: reviewed, deduplicated partial route timings.
    cost = pd.read_csv(CREDIT_REVIEW / "R3_partial_cost_scope.csv")
    cost_rows = []
    for record in cost.to_dict("records"):
        record["table_name"] = "Partial Scoring-Path & Setup Cost"
        record["route"] = "Update_INT6_qMV" if record["route"] == "Update_INT6" else record["route"]
        cost_rows.append(record)
    for op in ("Op_Half_Replace", "Op_Full_Replace"):
        for cohort in ("all_eval", "d0_alert_only"):
            qmv = next(r for r in cost_rows if r["operation"] == op and r["cohort"] == cohort and r["route"] == "Update_INT6_qMV")
            cost_rows.append({"operation": op, "cohort": cohort, "route": "Update_INT6_Tuned3D",
                              "banks_n": 6, "setup_CPU_seconds": None,
                              "online_CPU_ms_per_input_row": None,
                              "scope": "Tuned3D total and shared component not timed; related qMV full path includes its own ratio/mapping work; CALREF/CV excluded",
                              "INT_method": "6 INT + 3D logistic combination; measured qMV projection component is not the complete tuned3D time",
                              "raw_preprocess": "shared raw preprocessing reference ~544 ms/all EVAL; borrowed from D3, not measured on this path",
                              "table_name": "Partial Scoring-Path & Setup Cost",
                              "qMV_fullpath_reference_CPU_ms_per_input_row": qmv["online_CPU_ms_per_input_row"]})
    return d1var, credit_rows, clinical_summary, cost_rows


def _write_manuscript(path: Path, table1_summary: list[dict], paired_final: list[dict],
                      c1_rows: list[dict], d1var: list[dict], credit_var: list[dict],
                      clinical: list[dict], costs: list[dict], adult_rows: list[dict]) -> None:
    d_summary = pd.DataFrame(d1var)
    a_scheme = d_summary[(d_summary.scheme == "Scheme_A_Scaled_OHE") & (d_summary.criterion == "total_lt50")]
    d_scheme = d_summary[(d_summary.scheme == "Scheme_D_Merged_Rare50_Unscaled") & (d_summary.criterion == "total_lt50")]
    def stage_value(frame, stage, col):
        hit = frame[frame.stage == stage]
        return float(hit.iloc[0][col]) if len(hit) else float("nan")
    a_s3 = stage_value(a_scheme, "Stage3_FullRefit", "variance_amplification_vs_IF")
    d_s3 = stage_value(d_scheme, "Stage3_FullRefit", "variance_amplification_vs_IF")
    # Descriptive top lines are computed from the cached aggregate, never re-fit here.
    credit_boot_if = next(r for r in paired_final if r["dataset"] == "UCI_Credit" and r["operation"] == "Bootstrap" and r["pair"] == "Pair_1_IF_minus_Margin" and r["metric"] == "R_070")
    d_full_if = next(r for r in paired_final if r["dataset"] == "Diabetes_Scheme_D" and r["operation"] == "Op_Full_Replace" and r["pair"] == "Pair_1_IF_minus_Margin" and r["metric"] == "R_070")
    lines = [
        "# Manuscript core draft (V2.8 locked cache analysis)", "",
        "> Analysis package created 2026-11-13 from saved public-data prediction caches. This local execution is not an external public preregistration. No prediction model was fitted and no ICU/confirmation data were read.", "",
        "## 1. Introduction and related work positioning", "",
        "| Work family | Estimand and information condition | Assumptions / geometry | Evidence added here |", "|---|---|---|---|",
        "| Empirical Bayes / Tweedie (Robbins, 1956; Efron, 2011) | Posterior mean under an independent superpopulation observation model | Fixed latent mean and independent noise; the classical `sqrt(2) sigma` construction does not encode a concrete pool update | We compare matched-pool bootstrap and pool replacement and use the three-covariance conditional moments; the results bound when independent-noise shrinkage transports poorly |",
        "| Influence functions / resampling (Koh & Liang, 2017; Giordano et al., 2019; Basu et al., 2021) | Local response to training-weight perturbations | First-order expansion around a fitted objective; curvature and penalty geometry matter | The representation audit shows rare scaled OHE directions can inflate full-refit variance relative to IF; unscaled OHE plus rare merging reduces this gap in the tested setup |",
        "| Prediction multiplicity / Rashomon and self-consistency | Disagreement over candidate models or a model set | Often conditional on the hypothesis class or a shared sampling mechanism | We condition stability on bootstrap vs. partial/full pool replacement and on whether Pool B is available |",
        "| Calibration / selective prediction (Guo et al., 2017; Geifman & El-Yaniv, 2017) | Calibration to outcomes or abstention by one-model confidence | Usually targets outcome uncertainty or ranks by `|z0|` | We target finite-reference retraining flips and compare center and variance at a common D0 alarm queue; one-dimensional monotone calibration cannot change the ordering |",
        "",
        "## 2. Methods", "",
        "### 2.1 Estimand and finite reference library", "",
        "For fixed deployed model `D0`, pools, threshold and retraining protocol `R`, the population target is `q_R(x) = P_R(D_R(x) != D0(x) | D0, Pools, Protocol)`. The empirical target from `B=11` saved AUDIT models is `qhat_11(x) = (1/11) sum_b 1[D_b(x) != D0(x)]`. The reported finite-AUDIT risks describe this saved reference library; they are not a guarantee for the unknown retraining distribution.",
        "",
        "Under a conditional Gaussian approximation with `Z0` variance `V0`, retrained output variance `VR`, covariance `C`, and marginal log-density `l(z)=log m(z)`, the conditional moments are `MR(z)=z+(V0-C)l'(z)` and `WR(z)=V0+VR-2C+(V0-C)^2 l''(z)`. For current decision sign `s0=2I[z>=0]-1`, the signed crossing probability is `Phi(-s0*MR/sqrt(WR))` when `WR>0`; the zero-variance limit is the deterministic sign disagreement. Gaussian conditioning, Tweedie identities, influence functions and finite-population SRSWOR corrections are established tools; our empirical contribution is to test protocol mismatch, sparse-OHE geometry and the division of labor between a shifted center and local variance. The expected operator center `beta_R` minimizes the expected weighted empirical objective and is a first-order anchor; it is not generally identical to `E[beta_refit]`.",
        "",
        "### 2.2 Cache reanalysis and uncertainty", "",
        "C1 standardizes each of six raw scores with that score's own CAL mean and population SD, then fits the same two-parameter nonincreasing logistic-Brier mapping to either the Pool-A Bootstrap CALREF6 target or the operation-specific Update CALREF6 target. Each score and target pair is fitted separately; the 144 low-dimensional map fits are not prediction-model fits. C2 uses 1,000 shared patient-level EVAL row-bootstrap weight vectors, re-cuts each bank's alert queue with exact score ties split fractionally at every coverage, and separately deletes each of 11 AUDIT columns. The row-bootstrap intervals condition on saved models, calibrators, and observed EVAL covariates; the jackknife measures finite AUDIT target sensitivity. Neither is an independent-population confidence interval.",
        "",
        "## 3. Results", "",
        "### 3.1 Cross-dataset and protocol results (Table 1)", "",
        f"Across the six Diabetes full-replacement banks, the cached IF-minus-Margin paired R70 difference is {d_full_if['paired_mean']:.6f} (bank wins {d_full_if['wins_negative_out_of_n']}/6); this is descriptive because EVAL patients are shared across banks. In Credit bootstrap, the corresponding difference is {credit_boot_if['paired_mean']:.6f} (wins {credit_boot_if['wins_negative_out_of_n']}/6). Table 1 and its CSV companions report per-bank means/ranges, paired differences, row-bootstrap intervals and AUDIT leave-one-column ranges. Bootstrap sample size and pool protocol differ between datasets; comparisons across datasets are not causal contrasts of replacement fraction.",
        "",
        "Adult P1/P2 remain development context, not independent external validation. The saved P1 locked-validation cohort contains 1,688 alert groups; the development cohort contains 2,938. P2 is an exploratory three-bank Logistic experiment. Its results are included separately in the Adult context file, without pooling banks as independent patients.",
        "",
        "### 3.2 Representation and regularization geometry (Table 2)", "",
        f"In the saved Diabetes D1 decomposition, Stage-3/IF variance amplification for Scheme A scaled OHE is {a_s3:.3f}; for Scheme D rare-category merging plus unscaled OHE it is {d_s3:.3f}. These are results from the earlier D1 execution, carried forward without model refitting. The component table reports Common, Rare and twice-Cross-covariance shares at every stage; negative cross shares are possible and the three terms sum to 100% up to stored precision. Credit closure is reported by operation rather than compressed into a single universal ratio.",
        "",
        "### 3.3 Matched-score center × variance × calibration target (Table 3)", "",
        "Table 3 reports all six combinations `(z0,1)`, `(z0,sqrt(v0))`, `(z0,sqrt(vR))`, `(zR,1)`, `(zR,sqrt(v0))`, and `(zR,sqrt(vR))`. The center contrasts hold a denominator fixed; the variance contrasts hold the center fixed. Old and new CALREF targets are applied to the same standardized score. R70/AURC/AUC are invariant to target source whenever both fitted slopes are strictly negative; Brier/ECE differences then isolate target-source calibration on that matched score. The data table includes the fitted slopes and any numerical rank-invariance residual.",
        "",
        "### 3.4 Clinical outcome trade-off and partial scoring-path cost (Table 4)", "",
        "The clinical table keeps both denominators: retained alert true positives over D0-alert true positives, and population sensitivity over all EVAL positives. PPV, flip risk, and the guardrail policies are shown together. Cost rows are deliberately labeled partial: D0 scoring, operator-center/Sigma setup, six-INT setup, online linear/quadratic paths and the mapping call are included where recorded; D0 fitting, CALREF generation, CV search/final fits and fitted preprocessing are excluded. The approximately 544 ms raw preprocessing figure is a separate D3 reference, not a U1 end-to-end measurement. Tuned3D's additional combination is not separately timed.",
        "",
        "## 4. Discussion and limitations", "",
        "The matched comparisons support a conditional conclusion: the value of local variance depends on whether a usable operator center is available and on the training representation. The Diabetes full-replacement result is not evidence that variance alone caused the improvement over Margin; Table 3 separates the two axes. Credit provides a one-shot public external dataset check but uses a different sample-size design (12,000 bootstrap vs. 6,000 replacement fits) and is not a replication of the Diabetes causal contrast. Diabetes underwent multiple rounds of development; Credit was designated as a one-shot external validation after method choices, so the two datasets have different selection histories. Small calibration changes, including Credit full-replacement ECE, must remain visible.",
        "",
        "This analysis does not convert local execution-time freezing into public preregistration. Patient row bootstrap intervals condition on a fixed observed EVAL cohort; leave-one-AUDIT intervals describe reference-column sensitivity only. Six bank ranges/standard deviations are training-draw sensitivity, not population confidence intervals. ECE is computed using prediction-probability bins, including q=1 in the last bin. Finite AUDIT zero-flip findings are resolution-limited and do not imply zero unknown retraining risk.",
        "",
        "The evidence is limited to public Adult, Diabetes, and Credit experiments and the saved finite model libraries. MIMIC/eICU confirmation data remain sealed, GM3 remains PARTIAL, and no clinical confirmation or final registration is asserted. This manuscript draft should not describe clinical readiness or universal coverage guarantees.",
        "",
        "## 5. Table and artifact map", "",
        "- Table 1: `Table1_main_summary.csv`, `Table1_paired_effects.csv`, `Table1_adult_context.csv`.",
        "- Table 2: `Table2_variance_geometry.csv`, `Table2_credit_closure.csv`.",
        "- Table 3: `C1_score_matched_metrics.csv`, `C1_orthogonal_contrasts.csv`, `C1_calibration_target_effects.csv`.",
        "- Table 4: `Table4_clinical_utility.csv`, `Table4_partial_scoring_cost.csv`.",
        "- C2 detail: `C2_row_bootstrap_summary.csv`, `C2_row_bootstrap_replicates.csv`, `C2_audit_loao_summary.csv`, `C2_audit_loao_replicates.csv`, `C2_event_counts.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    global OUT
    started_wall = time.monotonic(); started_cpu = time.process_time()
    final_out = OUT
    stage_out = BASE / "paper_evidence.pending_20261113_attempt2"
    if final_out.exists() or final_out.is_symlink():
        raise FileExistsError(f"final locked output is create-only: {final_out}")
    resumed_c1 = False
    if stage_out.exists() or stage_out.is_symlink():
        c1_required = [stage_out / name for name in ("C1_score_matched_metrics.csv",
                        "C1_orthogonal_contrasts.csv", "C1_calibration_target_effects.csv")]
        q_files = list((stage_out / "c1_recalibrated").glob("*.npz")) if (stage_out / "c1_recalibrated").is_dir() else []
        if (stage_out.is_symlink() or not stage_out.is_dir() or
                not all(p.is_file() and not p.is_symlink() for p in c1_required) or len(q_files) != 12):
            raise FileExistsError(f"pending output exists without a complete reusable C1 checkpoint: {stage_out}")
        resumed_c1 = True
    needed = [DIAB_SPLIT, CREDIT_DATA, U1, D1, CREDIT_ROOT, CREDIT_REVIEW]
    for p in needed:
        if p.is_symlink() or not p.exists():
            raise RuntimeError(f"required source missing/unsafe: {p}")
    OUT = stage_out
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"package": "paper_evidence", "created_date": "2026-11-13",
                "authorization": "current user C1-C4 instruction", "status": "CACHE_REANALYSIS_COMPLETE_PENDING_REVIEW",
                "models": {"prediction_model_fit_calls": 0, "prediction_model_inference_calls": 6,
                           "inference_scope": "Credit D0-only EVAL alert identity reconstruction using IDs/features; no EVAL outcome; all evaluated scores from cached predictions"},
                "C1": {"scope": "Diabetes Scheme D U1 Half/Full caches", "raw_score_grid": ["z0/1", "z0/sqrt(v0)", "z0/sqrt(vR)", "zR/1", "zR/sqrt(v0)", "zR/sqrt(vR)"],
                       "CAL_standardization": "score-specific CAL mean and population SD (ddof=0), shared across old/new target maps",
                       "mapping": "two-parameter logistic Brier, slope constrained nonpositive because score is stability-oriented", "fit_calls": 144},
                "C2": {"row_bootstrap_replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
                       "weights": "one multinomial patient-frequency vector shared over all six banks and operations within dataset; alert subsets are reweighted by ID row position",
                       "tie_rule": "exact score ties share fractional boundary mass; recut at each replicate and coverage",
                       "AUDIT_LOAO_columns": 11},
                "labels": {"Diabetes_EVAL_outcome_reads": 0, "Credit_EVAL_outcome_reads": 0,
                           "ICU_or_confirmation_reads": 0},
                "calibration_fits": {"C1_new_score_maps": 144, "prediction_model_fits": 0},
                "execution_attempts": {"first_attempt_failed_after_one_unsaved_C1_mapping_call": 1,
                                       "completed_attempt_persisted_C1_maps": 144,
                                       "total_C1_mapping_optimizer_calls_this_task": 145},
                "limits": ["six bank SD/range are training draw sensitivity, not patient population confidence intervals",
                           "row bootstrap conditions on fixed models and saved EVAL cohort", "LOAO measures finite AUDIT column sensitivity"]}
    if not resumed_c1:
        _json_write(OUT / "protocol.json", protocol)
        c1_metrics, c1_contrasts, c1_target_effects, c1_audit = _c1_run()
        _csv_write(OUT / "C1_score_matched_metrics.csv", c1_metrics)
        _csv_write(OUT / "C1_orthogonal_contrasts.csv", c1_contrasts)
        _csv_write(OUT / "C1_calibration_target_effects.csv", c1_target_effects)
    else:
        c1_metrics = pd.read_csv(OUT / "C1_score_matched_metrics.csv").to_dict("records")
        c1_contrasts = pd.read_csv(OUT / "C1_orthogonal_contrasts.csv").to_dict("records")
        c1_target_effects = pd.read_csv(OUT / "C1_calibration_target_effects.csv").to_dict("records")
        rank_diffs = [abs(float(row["new_minus_old"])) for row in c1_target_effects
                      if row.get("metric") in ("R_070", "AURC100", "AUC")]
        c1_audit = {"C1_mapping_fit_calls": len(c1_metrics), "C1_prediction_model_fit_calls": 0,
                    "C1_q_npz_files": len(q_files),
                    "max_abs_target_invariance_diff_R_AURC_AUC": max(rank_diffs, default=0.0)}
        if len(c1_metrics) != 144:
            raise RuntimeError(f"reused C1 checkpoint has wrong row count: {len(c1_metrics)}")

    datasets, id_hashes, identity_audit = _make_datasets()
    perbank, paired_by_bank, events, paired_summary = _point_tables(datasets)
    boot_summary, boot_reps, weight_digests, jack_summary, jack_reps = _bootstrap_and_jackknife(datasets)
    _csv_write(OUT / "Table1_main_per_bank.csv", perbank)
    _csv_write(OUT / "C2_point_paired_by_bank.csv", paired_by_bank)
    _csv_write(OUT / "C2_point_paired_summary.csv", paired_summary)
    _csv_write(OUT / "C2_event_counts.csv", events)
    _csv_write(OUT / "C2_row_bootstrap_summary.csv", boot_summary)
    _csv_write(OUT / "C2_row_bootstrap_replicates.csv", boot_reps)
    _csv_write(OUT / "C2_patient_weight_digests.csv", weight_digests)
    _csv_write(OUT / "C2_audit_loao_summary.csv", jack_summary)
    _csv_write(OUT / "C2_audit_loao_replicates.csv", jack_reps)
    t1_summary, adult_context, paired_final = _table1(datasets, perbank, paired_summary, boot_summary, jack_summary)
    _csv_write(OUT / "Table1_main_summary.csv", t1_summary)
    _csv_write(OUT / "Table1_adult_context.csv", adult_context)
    _csv_write(OUT / "Table1_paired_effects.csv", paired_final)

    d1var, credit_var, clinical, costs = _copy_tables(OUT)
    _csv_write(OUT / "Table2_variance_geometry.csv", d1var)
    _csv_write(OUT / "Table2_credit_closure.csv", credit_var)
    _csv_write(OUT / "Table4_clinical_utility.csv", clinical)
    _csv_write(OUT / "Table4_partial_scoring_cost.csv", costs)

    _write_manuscript(OUT / "MANUSCRIPT_CORE_DRAFT.md", t1_summary, paired_final, c1_metrics,
                      d1var, credit_var, clinical, costs, adult_context)
    if len(boot_summary) != 72 or len(jack_summary) != 72 or len(jack_reps) != 4752:
        raise RuntimeError("incomplete paired row/AUDIT uncertainty outputs")
    validation = {"status": "PASS", "assertions": {
        "C1_fits_exactly_144": c1_audit["C1_mapping_fit_calls"] == 144,
        "C1_model_training_zero": c1_audit["C1_prediction_model_fit_calls"] == 0,
        "C1_rank_metric_target_invariance_max_abs": c1_audit["max_abs_target_invariance_diff_R_AURC_AUC"],
        "Diabetes_EVAL_ids_n": int(len(datasets["Diabetes_Scheme_D"]["ids"])),
        "Credit_EVAL_ids_n": int(len(datasets["UCI_Credit"]["ids"])),
        "C2_row_bootstrap_replicates_each_dataset": BOOTSTRAP_REPS,
        "C2_patient_weight_digest_rows": len(weight_digests),
        "C2_LOAO_replicates_rows": len(jack_reps),
        "C2_all_row_bootstrap_summaries_present": len(boot_summary) == 2 * 3 * 3 * 4,
        "C2_all_LOAO_summaries_present_or_explicit_na": len(jack_summary) == 2 * 3 * 3 * 4,
        "Credit_D0_identity_reconstruction": identity_audit["credit_identity_reconstruction"],
        "EVAL_labels_read": 0, "prediction_model_fits": 0},
        "row_bootstrap_seed": BOOTSTRAP_SEED, "elapsed_wall_seconds": time.monotonic() - started_wall,
        "elapsed_cpu_seconds": time.process_time() - started_cpu,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "input_hashes": {"Diabetes_split_manifest": identity_audit["diabetes_identity_manifest_sha256"],
                         "Credit_public_feature_csv": identity_audit["credit_identity_reconstruction"]["credit_source_sha256"],
                         "Diabetes_U1_bank0_half_prediction_arrays": _sha_file(U1 / "bank0/Op_Half_Replace/prediction_arrays.npz"),
                         "Diabetes_D1_schemeD_bank0_predictions": _sha_file(D1 / "Scheme_D_Merged_Rare50_Unscaled/bank0/predictions.npz"),
                         "Credit_bank0_bootstrap_prediction_arrays": _sha_file(CREDIT_ROOT / "bank0/Bootstrap/prediction_arrays.npz")},
        "row_identity_hashes": id_hashes,
        "C1_counts": {k: v for k, v in c1_audit.items() if k != "maps"},
        "C1_unsaved_prior_failed_attempt_calls": 1,
        "resumed_from_complete_C1_checkpoint": resumed_c1,
        "C1_total_mapping_optimizer_calls_this_task": c1_audit["C1_mapping_fit_calls"] + 1,
        "files": []}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "execution_audit.json":
            validation["files"].append({"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size,
                                        "sha256": _sha_file(p)})
    readme = """# V2.8 cache-reanalysis work package\n\nThis package contains C1 score-matched calibration and orthogonal center/variance analysis, C2 paired patient-row bootstrap and AUDIT leave-one-column sensitivity, C3 scoped partial path cost accounting, and a structured manuscript draft. It is create-only and does not alter historical runs. No prediction models were fitted. Credit D0 inference was used only to restore alert-row identity from public IDs/features, checked against the saved alert logits; no EVAL target was loaded.\n\nSee `MANUSCRIPT_CORE_DRAFT.md`, `protocol.json`, and `execution_audit.json` for scope and validation.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    _json_write(OUT / "execution_audit.json", validation)
    os.replace(OUT, final_out)
    OUT = final_out
    print(json.dumps({"status": "PASS", "output": str(OUT),
                      "C1_rows": len(c1_metrics), "C1_map_fits": c1_audit["C1_mapping_fit_calls"],
                      "paired_point_rows": len(paired_by_bank), "row_bootstrap_summary_rows": len(boot_summary),
                      "row_bootstrap_replicate_rows": len(boot_reps), "LOAO_summary_rows": len(jack_summary),
                      "wall_seconds": validation["elapsed_wall_seconds"], "cpu_seconds": validation["elapsed_cpu_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
