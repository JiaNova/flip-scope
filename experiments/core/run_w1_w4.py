#!/usr/bin/env python3
"""Authorized W1-W4 public benchmark follow-up.

Stages reuse the fixed Adult/Diabetes protocols and CALREF utilities already
present in the workspace.  All new outputs are create-only.  The script does
not open ICU/confirmation sources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit, ndtr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.public_benchmarks import run_public_experiments as pub
from experiments.public_benchmarks import run_followup as follow
from experiments.adult_support import run_p1_instance_transfer as p1
from inflight import public_adult as adult

OUT = ROOT / "reports/core"
RUNS = ROOT / "runs/diabetes_credit"
P1_OUT = ROOT / "reports/adult_support/P1_EXECUTION_20261110"
P2_FOLLOW = ROOT / "reports/public_benchmarks/followup/p2_variable_reference_sets"
P3_FOLLOW = ROOT / "reports/public_benchmarks/followup/p3_independent_calibration"
P3_ORIGINAL = ROOT / "reports/public_benchmarks/p3_three_bank_results"
P2_ORIGINAL = ROOT / "reports/public_benchmarks/p2_three_bank_results"
DIABETES_CSV = pub.DATA_DIR / "diabetic_data.csv"
RISK_COVERAGES = (0.50, 0.70, 0.80, 0.85, 0.90, 1.00)
LGB_FULL_N = 33646
SEED_BASE = 2026112000


class WorkError(RuntimeError):
    pass


def _json_safe(x: Any):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _canonical(x: Any) -> bytes:
    return (json.dumps(_json_safe(x), sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise WorkError(f"missing/unsafe required file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_new(path: Path, raw: bytes) -> None:
    """Atomically publish one new regular file and refuse all replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise WorkError(f"create-only output already exists: {path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path, follow_symlinks=False)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _write_json(path: Path, value: Any) -> None:
    _write_new(path, _canonical(value))


def _csv_bytes(rows: list[dict], fields: list[str] | None = None) -> bytes:
    if not rows:
        return b""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _risk(score: np.ndarray, t: np.ndarray, c: float) -> float:
    return pub.compute_tie_aware_risk(np.asarray(score, float), np.asarray(t, float), c)


def _auc_finite(score: np.ndarray, audit: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(audit, dtype=np.int8)
    if y.ndim != 2 or y.shape[0] != len(score):
        raise WorkError("AUDIT labels and score dimensions do not agree")
    try:
        return float(roc_auc_score(y.reshape(-1), np.repeat(np.asarray(score), y.shape[1])))
    except ValueError:
        return None


def _metrics(q: np.ndarray, audit: np.ndarray, margin: np.ndarray,
             qsqrt2: np.ndarray) -> dict:
    t = audit.mean(axis=1).astype(float)
    return {
        "n_alerts": int(len(q)),
        **{f"R_{int(c*100):03d}": _risk(q, t, c) for c in RISK_COVERAGES},
        "AURC100": float(np.mean([_risk(q, t, k/100) for k in range(1, 101)])),
        "AUC_flip": _auc_finite(q, audit),
        "Brier_finite_AUDIT": float(np.mean((np.asarray(q)[:, None] - audit) ** 2)),
        "ECE10": pub._ece10(np.asarray(q), t),
        "delta_Brier_vs_margin": float(np.mean(q*q - margin*margin - 2*(q-margin)*t)),
        "delta_Brier_vs_q_sqrt2": float(np.mean(q*q - qsqrt2*qsqrt2 - 2*(q-qsqrt2)*t)),
        "mean_flip_rate": float(t.mean()),
    }


def _aggregate(rows: list[dict], group_keys: list[str], value_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    out = []
    for key, vals in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        item = dict(zip(group_keys, key))
        for name in value_keys:
            xs = [float(v[name]) for v in vals if v.get(name) is not None and np.isfinite(v[name])]
            item[name + "_mean"] = float(np.mean(xs)) if xs else None
            item[name + "_min"] = float(np.min(xs)) if xs else None
            item[name + "_max"] = float(np.max(xs)) if xs else None
        out.append(item)
    return out


def _cal_stability(stab_cal: np.ndarray, t_cal: np.ndarray,
                   stab_eval: np.ndarray) -> tuple[np.ndarray, dict]:
    """Shared two-parameter nonincreasing logistic-Brier CALREF map."""
    return follow.calibrate(stab_cal, t_cal, stab_eval)


def _apply_cal_map(mapping: dict, x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float)
    center=float(mapping["center"]);scale=float(mapping["scale"])
    a,b=np.asarray(mapping["coefficients"],float)
    return expit(a+b*(x-center)/scale)


def _pool_a_if_variance(xa: np.ndarray,ya: np.ndarray,beta: np.ndarray,
                        x_eval: np.ndarray) -> np.ndarray:
    """Bootstrap score sandwich using A only; B is not referenced."""
    pa=expit(xa@beta);ua=(pa-np.asarray(ya,float))[:,None]*xa
    ua=ua-ua.mean(axis=0,keepdims=True)
    h=xa.T@((pa*(1-pa))[:,None]*xa)
    penalty=np.ones(xa.shape[1]);penalty[0]=0.
    h+=np.diag(penalty);hinv=np.linalg.inv(h);j=ua.T@ua
    projected=x_eval@hinv
    v=np.einsum("ij,jk,ik->i",projected,j,projected,optimize=True)
    return np.maximum(v,0.)


def w1_p1() -> dict:
    """Ranking-only mean-center ablation; P1 has no independent CALREF cache."""
    out = OUT / "w1_p1"
    rows = []
    for cohort, file in (("Adult_P1_dev_2938", "development_instance_scores.csv"),
                         ("Adult_P1_locked_1688", "locked_validation_instance_scores.csv")):
        data = pd.read_csv(P1_OUT / file)
        z0 = data.z0.to_numpy(float)
        zi = data[[f"z_INT_{i:02d}" for i in range(1, 7)]].to_numpy(float)
        t = data.audit_flip_rate.to_numpy(float)
        if len(data) not in (2938, 1688) or not np.isfinite(zi).all():
            raise WorkError(f"unexpected P1 score file cohort shape: {file}")
        audit_n = np.rint(t * 11).astype(int)
        if np.max(np.abs(t * 11 - audit_n)) > 1e-9:
            raise WorkError("P1 finite-11 audit target is not an integer count fraction")
        audit = np.concatenate([np.r_[np.ones(k, dtype=np.int8),
                                      np.zeros(11-k, dtype=np.int8)][None, :] for k in audit_n])
        fused = 0.5*z0 + 0.5*zi.mean(axis=1)
        risk_scores = {
            "fusion_mean_only": -fused,
            "q_V": data.q_V.to_numpy(float),
            "q_MV_shrunk": data.q_MV_shrunk.to_numpy(float),
        }
        for method, q in risk_scores.items():
            rows.append({"cohort": cohort, "method": method, "n_alerts": len(q),
                         "R_070": _risk(q, t, .70), "R_085": _risk(q, t, .85),
                         "AURC100": float(np.mean([_risk(q, t, k/100) for k in range(1, 101)])),
                         "AUC_flip": _auc_finite(q, audit),
                         "calibration": "ranking-only; any strictly monotone CALREF map preserves these four metrics",
                         "CALREF_available": False})
    _write_new(out / "ranking_ablation.csv", _csv_bytes(rows))
    report = {"status": "P1_RANKING_ABLATION_COMPLETE",
              "calref_note": "No CALREF model predictions are saved for P1. These requested endpoints are ranking-only and are invariant to a shared strictly monotone map; no probability calibration metrics are reported.",
              "rows": rows,
              "source_hashes": {f: _sha_file(P1_OUT/f) for f in
                                ("development_instance_scores.csv", "locked_validation_instance_scores.csv")}}
    _write_json(out / "report.json", report)
    return report


def _adult_p2_design(bank: int):
    _, _, parts = pub._adult_groups_for_p2()
    pools = pub._p2_bank_pools(parts["train"])
    a, b0 = pools[bank]
    occupied = {g["group_id"] for aa, bb in pools for g in aa+bb}
    reserve = sorted((g for g in parts["train"] if g["group_id"] not in occupied),
                     key=lambda g: g["group_id"])
    b = b0 + reserve[bank*2000:(bank+1)*2000]
    design = pub.read_json(pub.RUN_ROOT/f"adult_p2/bank-{bank}/pool_manifest/fit_design.json")
    fit = pub._adult_lr_transform_fit(a, numeric_feature_indices=design["numeric_feature_indices"])
    xa, xb = pub._adult_lr_features(a, fit), pub._adult_lr_features(b, fit)
    cal = pub._adult_lr_features(parts["calibration"], fit)
    eva = pub._adult_lr_features(parts["evaluation"], fit)
    return a, b, fit, xa, xb, cal, eva, parts


def w1_p2() -> dict:
    """Add center-only and split-IF baselines using already saved P2 models."""
    out = OUT / "w1_p2"
    all_rows, calibration_records, if_diag = [], [], []
    for bank in range(3):
        a, b, fit, xa, xb, xc0, xe0, parts = _adult_p2_design(bank)
        ncal = len(xc0)
        cache_path = pub.RUN_ROOT/f"adult_p2/bank-{bank}/logit_prediction_cache/arrays.npz"
        with np.load(cache_path, allow_pickle=False) as cached:
            z_d0_all = cached["z_shared_D0"].copy()
        if z_d0_all.shape[0] != ncal + len(xe0):
            raise WorkError("P2 D0 prediction cache has unexpected CAL/EVAL row layout")
        beta0, _ = pub._p2_load_beta(bank, "shared", "D0")
        z0_all = np.vstack((xc0, xe0)) @ beta0
        # The old cache is known to have been generated under process-dependent
        # categorical-column order.  It is measured for audit only; never used.
        legacy_cache_difference=float(np.max(np.abs(z0_all-z_d0_all)))
        cal_alert, eval_alert = z0_all[:ncal] >= 0, z0_all[ncal:] >= 0
        for op in ("alpha_0", "alpha_0_5", "bootstrap"):
            model_path = P2_FOLLOW/f"bank{bank}/{op}.npz"
            if model_path.is_symlink() or not model_path.is_file():
                raise WorkError(f"missing saved P2 model prediction bundle: {model_path}")
            with np.load(model_path, allow_pickle=False) as bundle:
                betas = bundle["betas"].copy()
                logits = bundle["logits"].copy()
            if betas.shape[0] != 23 or logits.shape != (len(z0_all), 23):
                raise WorkError(f"P2 bank {bank}/{op} saved role bundle shape mismatch")
            ic = logits[:ncal, :6][cal_alert]
            ie = logits[ncal:, :6][eval_alert]
            cal_roles = logits[:ncal, 6:12][cal_alert]
            audit = (logits[ncal:, 12:][eval_alert] < 0).astype(np.int8)
            tc = (cal_roles < 0).mean(axis=1)
            t = audit.mean(axis=1)
            sigma2 = float(np.mean(np.var(ic, axis=1, ddof=1)))
            z0c, z0e = z0_all[:ncal][cal_alert], z0_all[ncal:][eval_alert]
            mc, me = ic.mean(axis=1), ie.mean(axis=1)
            s2c, s2e = np.var(ic, axis=1, ddof=1), np.var(ie, axis=1, ddof=1)
            vt_c, vt_e = .5*s2c+.5*sigma2, .5*s2e+.5*sigma2
            mtilde_c, mtilde_e = .5*z0c+.5*mc, .5*z0e+.5*me
            qlex_c = (np.sum(ic < 0, axis=1)+.25+.5*expit(-z0c))/7.0
            qlex_e = (np.sum(ie < 0, axis=1)+.25+.5*expit(-z0e))/7.0
            # D0-centered variance uses only pool A; operator variance is at beta_R.
            v0all = _pool_a_if_variance(xa,pub._group_soft_labels(a),beta0,
                                        np.vstack((xc0,xe0)))
            if op == "bootstrap":
                beta_r = beta0.copy()
            else:
                beta_r = np.asarray(pub.read_json(P2_FOLLOW/f"bank{bank}/{op}_anchor.json")["beta"], float)
            operator_score, vrall = follow.anchor_if(
                xa, pub._group_soft_labels(a), xb, pub._group_soft_labels(b), beta_r,
                np.vstack((xc0, xe0)), 0.0 if op == "bootstrap" else (0.0 if op == "alpha_0" else .5),
                op == "bootstrap")
            zrc = (np.vstack((xc0, xe0))[:ncal] @ beta_r)[cal_alert]
            zre = (np.vstack((xc0, xe0))[ncal:] @ beta_r)[eval_alert]
            d0ifc = z0c / np.sqrt(np.maximum(v0all[:ncal][cal_alert], 1e-12))
            d0ife = z0e / np.sqrt(np.maximum(v0all[ncal:][eval_alert], 1e-12))
            opifc = operator_score[:ncal][cal_alert]
            opife = operator_score[ncal:][eval_alert]
            raw_features = {
                "margin_logistic": (z0c, z0e),
                "int_mean_logistic": (mc, me),
                "fusion_mean_only": (mtilde_c, mtilde_e),
                "qMV": (mtilde_c/np.sqrt(vt_c), mtilde_e/np.sqrt(vt_e)),
                "qV": (z0c/np.sqrt(vt_c), z0e/np.sqrt(vt_e)),
                "D0_center_IF": (d0ifc, d0ife),
                "operator_center_mean": (zrc, zre),
                "operator_center_IF": (opifc, opife),
                "int_freq": (-np.sum(ic < 0, axis=1), -np.sum(ie < 0, axis=1)),
                "int_freq_margin_tiebreak": (-qlex_c, -qlex_e),
                "q_global_1": (-ndtr(-z0c/np.sqrt(sigma2)), -ndtr(-z0e/np.sqrt(sigma2))),
                "q_sqrt2": (-ndtr(-z0c/np.sqrt(2*sigma2)), -ndtr(-z0e/np.sqrt(2*sigma2))),
            }
            saved_maps_path=P2_FOLLOW/f"bank{bank}/{op}_calibrators.json"
            saved_maps=json.loads(saved_maps_path.read_text())
            existing_map_names={"margin_logistic":"margin_calibrated",
                "int_mean_logistic":"int_mean_calibrated","qV":"qV_calibrated",
                "qMV":"qMV_calibrated","operator_center_mean":"anchor_mean_calibrated",
                "operator_center_IF":"IF_anchor_calibrated","int_freq":"int_frequency_calibrated"}
            calibrated, maps = {}, {}
            for method, (sc, se) in raw_features.items():
                prior_name=existing_map_names.get(method)
                if prior_name is not None and prior_name in saved_maps:
                    maps[method]=saved_maps[prior_name]
                    calibrated[method]=_apply_cal_map(maps[method],se)
                else:
                    calibrated[method], maps[method] = _cal_stability(sc, tc, se)
            qmargin = calibrated["margin_logistic"]
            qsqrt2 = calibrated["q_sqrt2"]
            for method, q in calibrated.items():
                metric = _metrics(q, audit, qmargin, qsqrt2)
                all_rows.append({"task": "Adult_P2_Logistic", "bank": bank,
                                 "operation": op, "method": method, **metric})
            if_diag.append({"task":"P2","bank":bank,"operation":op,
                "V0_mean":float(np.mean(v0all[ncal:][eval_alert])),
                "VR_mean":float(np.mean(vrall[ncal:][eval_alert])),
                "AUDIT_variance_mean":float(np.mean(np.var(logits[ncal:,12:][eval_alert],axis=1,ddof=1))),
                "D0_IF_operator_IF_corr":float(np.corrcoef(v0all[ncal:][eval_alert],vrall[ncal:][eval_alert])[0,1]),
                "CAL_alerts":int(cal_alert.sum()),"EVAL_alerts":int(eval_alert.sum())})
            new_map_count=sum(m not in existing_map_names for m in raw_features)
            calibration_records.append({"bank":bank,"operation":op,"map_count":len(maps),
                "new_map_count":new_map_count,
                "calref_rows":int(len(tc)),"calref_target_mean":float(np.mean(tc)),
                "saved_bundle_sha256":_sha_file(model_path),"sigma_g_sq":sigma2,
                "ignored_legacy_D0_cache_max_abs_difference":legacy_cache_difference})
            caldir = out/f"bank{bank}_{op}"
            _write_json(caldir/"calibrators.json", maps)
            _write_new(caldir/"metrics.json", _canonical({r["method"]:r for r in all_rows
                if r["bank"]==bank and r["operation"]==op}))
    metric_fields = [k for k in all_rows[0] if k not in ("task","bank","operation","method")]
    _write_new(out/"per_bank_metrics.csv", _csv_bytes(all_rows))
    agg = _aggregate(all_rows,["task","operation","method"],metric_fields)
    _write_new(out/"mean_range_metrics.csv",_csv_bytes(agg))
    _write_new(out/"if_diagnostics.csv",_csv_bytes(if_diag))
    report={"status":"W1_P2_COMPLETED","new_prediction_model_fits":0,
        "additional_CALREF_map_fits":sum(x["new_map_count"] for x in calibration_records),
        "maps_per_bank_operation":len(raw_features),"calibration_records":calibration_records,
        "column_order":"saved corrected per-run beta bundles; D0 recomputed in persisted numeric feature order",
        "method_orientation":"all maps consume a stability score (larger means more stable) and use the shared nonincreasing two-parameter CALREF logistic-Brier map",
        "IF":"D0-center variance uses pool A only; operator IF uses the saved expected-objective center and finite-pool design covariance",
        "diagnostics":if_diag,"aggregate":agg}
    _write_json(out/"report.json",report)
    return report


def _full_lgb_fit_params(seed: int) -> dict:
    return {"objective":"binary","metric":"binary_logloss","learning_rate":.05,
        "num_leaves":15,"max_depth":5,"min_data_in_leaf":40,"lambda_l2":1.,
        "feature_fraction":1.,"bagging_fraction":1.,"bagging_freq":0,
        "deterministic":True,"force_col_wise":True,"num_threads":1,
        "device_type":"cpu","verbosity":-1,"seed":seed,"data_random_seed":seed,
        "feature_fraction_seed":seed,"bagging_seed":seed,"drop_seed":seed}


def _lgb_new(path: Path, train_x: pd.DataFrame, y: np.ndarray, weights: np.ndarray,
             seed: int, model_meta: dict) -> tuple[lgb.Booster, dict]:
    if path.exists() or path.is_symlink():
        raise WorkError(f"refuse existing W2-A model path: {path}")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    dataset = lgb.Dataset(train_x, label=y, weight=weights,
                          categorical_feature=list(pub.DIABETES_CATEGORICAL),
                          free_raw_data=False)
    booster = lgb.train(_full_lgb_fit_params(seed), dataset, num_boost_round=100)
    raw = booster.model_to_string().encode()
    manifest = {**model_meta,"seed":int(seed),"fit_wall_seconds":time.monotonic()-start_wall,
        "fit_cpu_seconds":time.process_time()-start_cpu,"model_sha256":_sha(raw),
        "lightgbm_version":lgb.__version__,"boosting_rounds":100,"device":"cpu","threads":1}
    _write_new(path/"booster.txt",raw)
    _write_json(path/"model_manifest.json",manifest)
    return booster, manifest


def _predict_centered(model: lgb.Booster, xcal: pd.DataFrame, xeva: pd.DataFrame,
                      gamma: float) -> tuple[np.ndarray,np.ndarray]:
    pc=model.predict(xcal,num_threads=1); pe=model.predict(xeva,num_threads=1)
    return pub.safe_logit(pc)-gamma, pub.safe_logit(pe)-gamma


def _feature_rows_diabetes(cal: pd.DataFrame, eva: pd.DataFrame, z0c: np.ndarray,
                           z0e: np.ndarray, p0c: np.ndarray, p0e: np.ndarray,
                           prep: dict, train: pd.DataFrame,
                           zic: np.ndarray, sigma2: float) -> tuple[dict, dict]:
    mc, me = zic.mean(axis=1), None
    # The proxy is fitted only on CAL with D0+INT teachers. Evaluation uses D0 and X only.
    xprep, delta_model, eta_model = pub._p3_proxy_fit(cal, train, prep, z0c,p0c,zic,sigma2)
    all_df = pd.concat([cal,eva],ignore_index=True)
    xall = pub.transform_sparse_proxy(all_df,prep,xprep,np.r_[z0c,z0e],np.r_[p0c,p0e])
    d = delta_model.predict(xall); h = eta_model.predict(xall)
    vhat=np.clip(np.exp(h),.1*sigma2,10*sigma2)
    usur=(np.r_[z0c,z0e]+d)/np.sqrt(vhat)
    return {"proxy":(usur[:len(cal)],usur[len(cal):])},{"proxy_ridge_fits":2}


def _make_calibrated_metrics(bank: int, source: str, z0c: np.ndarray, z0e: np.ndarray,
                             zic: np.ndarray, zie: np.ndarray, zcalref: np.ndarray,
                             zaudit: np.ndarray, gamma: float, p0c: np.ndarray, p0e: np.ndarray,
                             cal_features: pd.DataFrame, eval_features: pd.DataFrame,
                             train_features: pd.DataFrame, prep: dict) -> tuple[list[dict],dict]:
    cal_alert, eval_alert = z0c>=0, z0e>=0
    audit=(zaudit[eval_alert]<0).astype(np.int8)
    tcal=(zcalref[cal_alert]<0).mean(axis=1)
    sigma2=float(np.var(zic[cal_alert],axis=1,ddof=1).mean())
    if not np.isfinite(sigma2) or sigma2<=0: raise WorkError("CAL INT variance prior is nonpositive")
    ic,ie=zic[cal_alert],zie[eval_alert]
    zcc,zee=z0c[cal_alert],z0e[eval_alert]
    zbarc,zbare=ic.mean(axis=1),ie.mean(axis=1)
    s2c,s2e=np.var(ic,axis=1,ddof=1),np.var(ie,axis=1,ddof=1)
    vt_c,vt_e=.5*s2c+.5*sigma2,.5*s2e+.5*sigma2
    m_c,m_e=.5*(zcc+zbarc),.5*(zee+zbare)
    k_c,k_e=np.sum(ic<0,axis=1),np.sum(ie<0,axis=1)
    qlexc=(k_c+.25+.5*expit(-zcc))/7
    qlexe=(k_e+.25+.5*expit(-zee))/7
    stable={"margin_logistic":(zcc,zee),"int_mean_logistic":(zbarc,zbare),
        "fusion_mean_only":(m_c,m_e),"qMV":(m_c/np.sqrt(vt_c),m_e/np.sqrt(vt_e)),
        "qV":(zcc/np.sqrt(vt_c),zee/np.sqrt(vt_e)),"int_freq":(-k_c,-k_e),
        "int_freq_margin_tiebreak":(-qlexc,-qlexe)}
    if "proxy" in source:
        pass
    # Add the test-time-zero-INT proxy, whose teacher targets are CAL-only.
    xprep,delta,eta=pub._p3_proxy_fit(cal_features,train_features,prep,z0c,p0c,zic,sigma2)
    all_features=pd.concat([cal_features,eval_features],ignore_index=True)
    proxy_x=pub.transform_sparse_proxy(all_features,prep,xprep,np.r_[z0c,z0e],np.r_[p0c,p0e])
    hu=eta.predict(proxy_x); dd=delta.predict(proxy_x)
    vhat=np.clip(np.exp(hu),.1*sigma2,10*sigma2)
    u=(np.r_[z0c,z0e]+dd)/np.sqrt(vhat)
    stable["q_MV_surrogate"]=(u[:len(z0c)][cal_alert],u[len(z0c):][eval_alert])
    # Raw one-scale controls receive the same CALREF map as the fitted methods.
    stable["q_global_1"]=( -ndtr(-zcc/np.sqrt(sigma2)), -ndtr(-zee/np.sqrt(sigma2)))
    stable["q_sqrt2"]=( -ndtr(-zcc/np.sqrt(2*sigma2)), -ndtr(-zee/np.sqrt(2*sigma2)))
    qeval,maps={},{}
    for method,(sc,se) in stable.items():
        qeval[method],maps[method]=_cal_stability(sc,tcal,se)
    qm=qeval["margin_logistic"]; qs=qeval["q_sqrt2"]
    rows=[]
    for method,q in qeval.items():
        rows.append({"task":source,"bank":bank,"method":method,
                     **_metrics(q,audit,qm,qs)})
    meta={"sigma_g_sq_CAL":sigma2,"CAL_alerts":int(cal_alert.sum()),
          "EVAL_alerts":int(eval_alert.sum()),"CALREF_count":int(zcalref.shape[1]),
          "AUDIT_count":int(zaudit.shape[1]),"calibrator_fits":len(maps),
          "proxy_ridge_fits":2,"calref_target_mean":float(tcal.mean()),
          "calref_maps":maps,"test_time_INT_inference_for_proxy":0,
          "audit_evaluation_only":True}
    return rows,meta


def w2a_fulltrain() -> dict:
    out=OUT/"w2a_diabetes_lgb_fulltrain"; run=RUNS/"w2a"
    if out.exists() or run.exists(): raise WorkError("W2-A create-only output path exists")
    process_wall,process_cpu=time.monotonic(),time.process_time()
    selected,audit=pub.load_selected_diabetes()
    train_all=selected[selected.split=="TRAIN"].reset_index(drop=True)
    cal=selected[selected.split=="CAL"].reset_index(drop=True)
    eva=selected[selected.split=="EVAL"].reset_index(drop=True)
    if len(train_all)!=42058 or len(cal)!=13983 or len(eva)!=13949:
        raise WorkError("frozen Diabetes split size mismatch")
    plan={"authorization":"current user W1-W4 message","status":"fixed before W2-A model fits",
        "train_sampling":"3 independent 33,646-patient SRS-without-replacement draws from fixed 42,058 TRAIN patients; banks may overlap",
        "seed_base":SEED_BASE,"models_per_bank":{"D0":1,"INT":6,"CALREF":6,"AUDIT":11},
        "CAL":"all 13,983 CAL patients, tau at D0 80th percentile","EVAL":"all 13,949 EVAL patients, audit only",
        "lgbm":_full_lgb_fit_params(SEED_BASE),"rounds":100,"calibration":"two-parameter nonincreasing Brier logistic; six CALREF flips on CAL alert rows",
        "expected_prediction_models":72,"native_threads":1,"gpu":False}
    _write_json(out/"protocol.json",plan)
    fit_summary=[]; metrics=[]; bank_reports=[]
    for bank in range(3):
        bseed=SEED_BASE+bank*10000
        rng=np.random.default_rng(bseed)
        take=np.sort(rng.choice(len(train_all),size=LGB_FULL_N,replace=False))
        tr=train_all.iloc[take].reset_index(drop=True)
        ids=tr.patient_nbr.astype(str).tolist()
        if len(set(ids))!=len(ids):raise WorkError("W2-A TRAIN bank has duplicate patients")
        prep=pub.fit_diabetes_preprocessor(tr)
        tx=pub.transform_diabetes_lgb(tr,prep)
        cx=pub.transform_diabetes_lgb(cal,prep); ex=pub.transform_diabetes_lgb(eva,prep)
        y=tr.y_readmit30.to_numpy(np.int8)
        bdir=run/f"bank{bank}"
        _write_json(bdir/"preprocessor.json",prep)
        _write_json(bdir/"bank_manifest.json",{"bank":bank,"seed":bseed,"train_n":len(tr),
            "train_patient_sha256":_sha("\n".join(ids).encode()),"fit_rows_hash":_sha("\n".join(map(str,take)).encode()),
            "train_positive_rate":float(y.mean()),"sample_without_replacement":True})
        specs=[("D0",0)]+[(f"INT-{j:02d}",100+j) for j in range(1,7)]+[(f"CALREF-{j:02d}",200+j) for j in range(1,7)]+[(f"AUDIT-{j:02d}",300+j) for j in range(1,12)]
        models={}
        for model_id,code in specs:
            seed=SEED_BASE+bank*10000+code
            if model_id=="D0": weights=np.ones(len(tr),float)
            else:
                rngm=np.random.default_rng(seed+50000)
                weights=np.bincount(rngm.integers(0,len(tr),size=len(tr)),minlength=len(tr)).astype(float)
            meta={"bank":bank,"model_id":model_id,"role":model_id.split("-")[0],
                "train_n":len(tr),"fit_weight_sum":float(weights.sum()),
                "unique_weight_patients":int(np.count_nonzero(weights)),
                "train_patient_sha256":_sha("\n".join(ids).encode()),
                "categorical_features":list(pub.DIABETES_CATEGORICAL)}
            model,record=_lgb_new(bdir/"models"/model_id,tx,y,weights,seed,meta)
            models[model_id]=model; fit_summary.append(record)
        # Cache logits only after each role's model has been durably saved.
        inference_start=time.monotonic();inference_cpu=time.process_time();inference_batches=0
        p0c=models["D0"].predict(cx,num_threads=1);p0e=models["D0"].predict(ex,num_threads=1);inference_batches+=2
        gamma=float(logit(np.quantile(p0c,.8)))
        z0c=pub.safe_logit(p0c)-gamma;z0e=pub.safe_logit(p0e)-gamma
        zic_cols=[];zie_cols=[]
        for j in range(1,7):
            zic_cols.append(pub.safe_logit(models[f"INT-{j:02d}"].predict(cx,num_threads=1))-gamma);inference_batches+=1
            zie_cols.append(pub.safe_logit(models[f"INT-{j:02d}"].predict(ex,num_threads=1))-gamma);inference_batches+=1
        zic=np.column_stack(zic_cols);zie=np.column_stack(zie_cols)
        zcalref=np.column_stack([pub.safe_logit(models[f"CALREF-{j:02d}"].predict(cx,num_threads=1))-gamma for j in range(1,7)])
        inference_batches+=6
        zaudit=np.column_stack([pub.safe_logit(models[f"AUDIT-{j:02d}"].predict(ex,num_threads=1))-gamma for j in range(1,12)])
        inference_batches+=11
        inference_wall=time.monotonic()-inference_start;inference_cpu=time.process_time()-inference_cpu
        _write_new(bdir/"predictions.npz",_npz_bytes(z0c=z0c,z0e=z0e,zic=zic,zie=zie,
            zcalref=zcalref,zaudit=zaudit,p0c=p0c,p0e=p0e))
        rows,meta=_make_calibrated_metrics(bank,"W2A_Diabetes_LGB_42k",z0c,z0e,zic,zie,
            zcalref,zaudit,gamma,p0c,p0e,cal,eva,tr,prep)
        meta.update({"model_inference_batches":inference_batches,"model_inference_wall_seconds":inference_wall,
                     "model_inference_CPU_seconds":inference_cpu})
        metrics.extend(rows);_write_json(bdir/"evaluation.json",{"bank":bank,**meta,
            "threshold_tau":float(expit(gamma)),"model_count":24})
        bank_reports.append({"bank":bank,"train_n":len(tr),"threshold_tau":float(expit(gamma)),**meta})
        print(f"W2-A bank {bank} complete; fits={len(specs)}; alerts={meta['EVAL_alerts']}",flush=True)
    fields=[k for k in metrics[0] if k not in ("task","bank","method")]
    _write_new(out/"per_bank_metrics.csv",_csv_bytes(metrics))
    aggregate=_aggregate(metrics,["task","method"],fields)
    _write_new(out/"mean_range_metrics.csv",_csv_bytes(aggregate))
    report={"status":"W2A_FULLTRAIN_COMPLETED","split_audit":audit,"banks":bank_reports,
        "new_LightGBM_fits":len(fit_summary),"model_fit_records":fit_summary,
        "model_fit_cpu_seconds":sum(r["fit_cpu_seconds"] for r in fit_summary),
        "model_fit_wall_seconds_sum":sum(r["fit_wall_seconds"] for r in fit_summary),
        "process_wall_seconds":time.monotonic()-process_wall,"process_CPU_seconds":time.process_time()-process_cpu,
        "model_inference_batches":sum(x["model_inference_batches"] for x in bank_reports),
        "model_inference_CPU_seconds":sum(x["model_inference_CPU_seconds"] for x in bank_reports),
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "mean_range_metrics":aggregate,"limitations":["Diabetes public benchmark only; not ICU evidence","3 banks reuse the same CAL/EVAL patients; bank range is not an independent-cohort confidence interval"]}
    _write_json(out/"report.json",report)
    return report


def _npz_bytes(**arrays) -> bytes:
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);return stream.getvalue()


def _diabetes_sparse_features(frame: pd.DataFrame, prep: dict, fitted: dict,
                              intercept: bool=True) -> sparse.csr_matrix:
    num=pub.transform_diabetes_lgb(frame,prep)[list(pub.DIABETES_NUMERIC)].to_numpy(float)
    ns=fitted["numeric_scaler"].transform(num)
    cat_values=np.empty((len(frame),len(pub.DIABETES_CATEGORICAL)),dtype=object)
    for j,col in enumerate(pub.DIABETES_CATEGORICAL):
        allowed=set(prep["categorical_vocabularies"][col]); vals=[]
        for v in frame[col].astype(str):
            v="__MISSING__" if v in ("","?") else v
            vals.append(v if v in allowed else "__UNK__")
        cat_values[:,j]=vals
    cats=fitted["cat_scaler"].transform(fitted["encoder"].transform(cat_values))
    blocks=[sparse.csr_matrix(ns),sparse.csr_matrix(cats)]
    if intercept: blocks.insert(0,sparse.csr_matrix(np.ones((len(frame),1),float)))
    return sparse.hstack(blocks,format="csr")


def _logloss_obj(beta: np.ndarray, x: sparse.csr_matrix, y: np.ndarray,
                 sample_w: np.ndarray, penalty: np.ndarray) -> float:
    eta=np.asarray(x@beta).reshape(-1)
    return float(np.dot(sample_w,np.logaddexp(0,eta)-y*eta)+.5*np.dot(penalty*beta,beta))


def _fit_irls(x: sparse.csr_matrix, y: np.ndarray, weights: np.ndarray | None=None,
              ridge: float=1.0, max_iter: int=100, tol: float=1e-8) -> tuple[np.ndarray,dict]:
    """Damped exact Newton/IRLS for sum-logloss + ridge (unpenalized intercept)."""
    x=x.tocsr();y=np.asarray(y,float);n,d=x.shape
    sw=np.ones(n,float) if weights is None else np.asarray(weights,float)
    if y.shape!=(n,) or sw.shape!=(n,) or np.any(sw<0) or sw.sum()<=0:
        raise WorkError("invalid IRLS data/weights")
    pen=np.full(d,float(ridge));pen[0]=0.0
    beta=np.zeros(d,float);obj=_logloss_obj(beta,x,y,sw,pen)
    for it in range(1,max_iter+1):
        eta=np.asarray(x@beta).reshape(-1);p=expit(np.clip(eta,-40,40))
        g=np.asarray(x.T@(sw*(p-y))).reshape(-1)+pen*beta
        w=sw*np.maximum(p*(1-p),1e-12)
        h=(x.T@sparse.diags(w)@x).toarray()
        h.flat[::d+1]+=pen
        try: step=np.linalg.solve(h,g)
        except np.linalg.LinAlgError as exc: raise WorkError("Diabetes Logistic IRLS Hessian singular") from exc
        if np.max(np.abs(g))<tol: break
        rate=1.0
        while rate>1e-8:
            cand=beta-rate*step;new_obj=_logloss_obj(cand,x,y,sw,pen)
            if new_obj<=obj+1e-12:break
            rate*=.5
        if rate<=1e-8: raise WorkError("Diabetes Logistic IRLS line search failed")
        beta=cand;delta=float(np.max(np.abs(rate*step)));obj=new_obj
        if delta<tol:break
    grad=np.asarray(x.T@(sw*(expit(np.clip(np.asarray(x@beta).reshape(-1),-40,40))-y))).reshape(-1)+pen*beta
    if not np.isfinite(beta).all() or np.max(np.abs(grad))>1e-4:
        raise WorkError(f"Diabetes Logistic IRLS did not converge; gradient_inf={np.max(np.abs(grad))}")
    return beta,{"iterations":it,"objective":obj,"gradient_inf":float(np.max(np.abs(grad))),
                 "ridge":ridge,"solver":"damped Newton/IRLS exact Hessian"}


def _logit_l2(x: sparse.csr_matrix,beta: np.ndarray) -> np.ndarray:
    return np.asarray(x@beta).reshape(-1)


def _fit_if_variance(xtrain: sparse.csr_matrix,ytrain: np.ndarray,beta: np.ndarray,
                     xeval: sparse.csr_matrix) -> tuple[np.ndarray,np.ndarray,float]:
    """Conditional bootstrap sandwich variance at the fixed D0 objective.

    The two centers coincide here, so both scores use the same conditional
    variance. An unconditional VR/V0=2 heuristic must not double this value.
    """
    t0=time.process_time();eta=_logit_l2(xtrain,beta);p=expit(np.clip(eta,-40,40))
    resid=p-ytrain
    u=xtrain.multiply(resid[:,None]).tocsr()
    mean_u=np.asarray(u.mean(axis=0)).reshape(-1)
    ww=np.maximum(p*(1-p),1e-12)
    h=(xtrain.T@sparse.diags(ww)@xtrain).toarray()
    h.flat[::h.shape[0]+1]+=1.0
    h[0,0]-=1.0
    hinv=np.linalg.inv(h)
    # Covariance of multinomial bootstrap score sums is the centered score cross-product.
    j=(u.T@u).toarray()-xtrain.shape[0]*np.outer(mean_u,mean_u)
    middle=hinv@j@hinv
    xe=np.asarray(xeval.toarray())
    v0=np.einsum("ij,jk,ik->i",xe,middle,xe,optimize=True)
    v0=np.maximum(v0,0.0)
    return v0,v0.copy(),time.process_time()-t0


def w2b_diabetes_logistic(fulltrain: bool=False) -> dict:
    suffix="_fulltrain" if fulltrain else ""
    out=OUT/f"w2b_diabetes_logistic{suffix}";run=RUNS/f"w2b{suffix}"
    if out.exists() or run.exists():raise WorkError("W2-B create-only output path exists")
    process_wall,process_cpu=time.monotonic(),time.process_time()
    selected,audit=pub.load_selected_diabetes()
    train=selected[selected.split=="TRAIN"].reset_index(drop=True)
    cal=selected[selected.split=="CAL"].reset_index(drop=True)
    eva=selected[selected.split=="EVAL"].reset_index(drop=True)
    method_rows=[];bank_rows=[];fit_records=[];cov_rows=[]
    protocol={"authorization":"current user W1-W4 message","status":"fixed before W2-B fits",
        "bank_assignment":("same three 33,646-patient SRS-without-replacement draws as W2-A, from fixed 42,058 TRAIN patients" if fulltrain else "reuse fixed, disjoint 20261110 P3 TRAIN bank assignment (about 14k each)"),
        "bank_seed_base":SEED_BASE if fulltrain else None,
        "models_per_bank":{"D0":1,"INT":6,"CALREF":6,"AUDIT":11},"expected_prediction_models":72,
        "solver":"sum binary log loss + 0.5*||beta_slopes||^2, exact damped Newton/IRLS; intercept unpenalized",
        "bootstrap":"patient multiplicity weights, independent seeds for each role/model",
        "roles":"CALREF targets fit maps on CAL only; AUDIT used on EVAL only",
        "threshold":"each D0 CAL probability 80th percentile; logit-centered margins for every model",
        "native_threads":1,"gpu":False}
    _write_json(out/"protocol.json",protocol)
    for bank in range(3):
        if fulltrain:
            rng=np.random.default_rng(SEED_BASE+bank*10000)
            take=np.sort(rng.choice(len(train),size=LGB_FULL_N,replace=False))
            tr=train.iloc[take].reset_index(drop=True)
        else:
            tr=train[train.bank_id==bank].sort_values("patient_nbr",kind="mergesort").reset_index(drop=True)
        prep=pub.fit_diabetes_preprocessor(tr)
        fitted=pub.fit_sparse_proxy_preprocessor(tr,prep)
        xt=_diabetes_sparse_features(tr,prep,fitted);xc=_diabetes_sparse_features(cal,prep,fitted);xe=_diabetes_sparse_features(eva,prep,fitted)
        y=tr.y_readmit30.to_numpy(np.int8);yc=cal.y_readmit30.to_numpy(np.int8)
        ids=tr.patient_nbr.astype(str).tolist()
        bdir=run/f"bank{bank}";_write_json(bdir/"preprocessor.json",prep)
        _write_json(bdir/"bank_manifest.json",{"bank":bank,"train_n":len(tr),
            "train_patient_sha256":_sha("\n".join(ids).encode()),"split_audit_sha256":_sha_file(pub.OUT_ROOT/"diabetes_data_audit/audit.json"),
            "feature_shape":list(xt.shape),"positive_rate":float(y.mean()),
            "training_sample":"W2-A matching 33,646-row without-replacement draw" if fulltrain else "fixed disjoint P3 14k bank"})
        specs=[("D0",0)]+[(f"INT-{j:02d}",100+j) for j in range(1,7)]+[(f"CALREF-{j:02d}",200+j) for j in range(1,7)]+[(f"AUDIT-{j:02d}",300+j) for j in range(1,12)]
        betas={}
        for model_id,code in specs:
            seed=SEED_BASE+500000+bank*10000+code
            if model_id=="D0":w=np.ones(len(tr),float)
            else:w=np.bincount(np.random.default_rng(seed+50000).integers(0,len(tr),size=len(tr)),minlength=len(tr)).astype(float)
            t0=time.monotonic();c0=time.process_time()
            beta,info=_fit_irls(xt,y,w,ridge=1.0)
            fitrow={"bank":bank,"model_id":model_id,"seed":seed,"train_n":len(tr),
                "fit_wall_seconds":time.monotonic()-t0,"fit_cpu_seconds":time.process_time()-c0,
                "fit_weight_sum":float(w.sum()),"nonzero_weight_patients":int(np.count_nonzero(w)),**info}
            _write_json(bdir/"models"/model_id/"model.json",{**fitrow,"beta":beta.tolist(),"beta_sha256":_sha(beta.astype("<f8").tobytes())})
            fit_records.append(fitrow);betas[model_id]=beta
        # Use D0 CAL probabilities to freeze tau, then center every model's logits.
        infer_start=time.process_time()
        pc=expit(np.clip(_logit_l2(xc,betas["D0"]),-40,40));pe=expit(np.clip(_logit_l2(xe,betas["D0"]),-40,40))
        gamma=float(logit(np.quantile(pc,.8)))
        z0c=_logit_l2(xc,betas["D0"])-gamma;z0e=_logit_l2(xe,betas["D0"])-gamma
        int_infer_start=time.process_time()
        zic=np.column_stack([_logit_l2(xc,betas[f"INT-{j:02d}"])-gamma for j in range(1,7)])
        zie=np.column_stack([_logit_l2(xe,betas[f"INT-{j:02d}"])-gamma for j in range(1,7)])
        int_infer_cpu=time.process_time()-int_infer_start
        zcalref=np.column_stack([_logit_l2(xc,betas[f"CALREF-{j:02d}"])-gamma for j in range(1,7)])
        zaudit=np.column_stack([_logit_l2(xe,betas[f"AUDIT-{j:02d}"])-gamma for j in range(1,12)])
        inference_cpu=time.process_time()-infer_start
        # One D0 Hessian inverse and one sandwich pass over CAL+EVAL covariates.
        v0all,vRall,if_cpu=_fit_if_variance(xt,y,betas["D0"],sparse.vstack((xc,xe),format="csr"))
        v0c,v0e=v0all[:len(cal)],v0all[len(cal):]
        vr_c,vr_e=vRall[:len(cal)],vRall[len(cal):]
        ac,ae=z0c>=0,z0e>=0
        sigma2=float(np.var(zic[ac],axis=1,ddof=1).mean())
        # CAL-only surrogate uses D0/INT teachers and TRAIN-fitted input scaling.
        pxcal=pub.transform_sparse_proxy(cal,prep,fitted,z0c,pc)
        # CAL proxy targets are defined on all CAL rows; use all rows, not alert-only.
        mean_all=zic.mean(axis=1);s2_all=np.var(zic,axis=1,ddof=1)
        ridge_delta=Ridge(alpha=1.0,fit_intercept=True,solver="lsqr").fit(pxcal,.5*(mean_all-z0c))
        ridge_eta=Ridge(alpha=1.0,fit_intercept=True,solver="lsqr").fit(pxcal,np.log(.5*s2_all+.5*sigma2))
        pxeval=pub.transform_sparse_proxy(eva,prep,fitted,z0e,pe)
        dhat=np.r_[ridge_delta.predict(pxcal),ridge_delta.predict(pxeval)]
        etahat=np.r_[ridge_eta.predict(pxcal),ridge_eta.predict(pxeval)]
        vh=np.clip(np.exp(etahat),.1*sigma2,10*sigma2)
        proxy_u=(np.r_[z0c,z0e]+dhat)/np.sqrt(vh)
        ac,ae=z0c>=0,z0e>=0
        qlexc=(np.sum(zic[ac]<0,axis=1)+.25+.5*expit(-z0c[ac]))/7
        qlexe=(np.sum(zie[ae]<0,axis=1)+.25+.5*expit(-z0e[ae]))/7
        mc,me=zic[ac].mean(axis=1),zie[ae].mean(axis=1)
        s2c,s2e=np.var(zic[ac],axis=1,ddof=1),np.var(zie[ae],axis=1,ddof=1)
        m_c,m_e=.5*(z0c[ac]+mc),.5*(z0e[ae]+me)
        stable={"margin_logistic":(z0c[ac],z0e[ae]),"D0_center_IF":(z0c[ac]/np.sqrt(np.maximum(v0c[ac],1e-12)),z0e[ae]/np.sqrt(np.maximum(v0e[ae],1e-12))),
            "operator_center_mean":(z0c[ac],z0e[ae]),"operator_center_IF":(z0c[ac]/np.sqrt(np.maximum(vr_c[ac],1e-12)),z0e[ae]/np.sqrt(np.maximum(vr_e[ae],1e-12))),
            "ridge_surrogate":(proxy_u[:len(cal)][ac],proxy_u[len(cal):][ae]),
            "int_mean_logistic":(mc,me),"fusion_mean_only":(m_c,m_e),
            "qMV":(m_c/np.sqrt(.5*s2c+.5*sigma2),m_e/np.sqrt(.5*s2e+.5*sigma2)),
            "qV":(z0c[ac]/np.sqrt(.5*s2c+.5*sigma2),z0e[ae]/np.sqrt(.5*s2e+.5*sigma2)),
            "int_freq":(-np.sum(zic[ac]<0,axis=1),-np.sum(zie[ae]<0,axis=1)),
            "int_freq_margin_tiebreak":(-qlexc,-qlexe),
            "q_global_1":(-ndtr(-z0c[ac]/np.sqrt(sigma2)),-ndtr(-z0e[ae]/np.sqrt(sigma2))),
            "q_sqrt2":(-ndtr(-z0c[ac]/np.sqrt(2*sigma2)),-ndtr(-z0e[ae]/np.sqrt(2*sigma2)))}
        tc=(zcalref[ac]<0).mean(axis=1);audit_eval=(zaudit[ae]<0).astype(np.int8)
        qev,maps={},{}
        for m,(sc,se) in stable.items():qev[m],maps[m]=_cal_stability(sc,tc,se)
        qm,qs=qev["margin_logistic"],qev["q_sqrt2"]
        task_name="W2B_Diabetes_Logistic_42k" if fulltrain else "W2B_Diabetes_Logistic_14k"
        for m,q in qev.items():method_rows.append({"task":task_name,"bank":bank,"method":m,**_metrics(q,audit_eval,qm,qs)})
        audit_var=np.var(zaudit[ae],axis=1,ddof=1)
        for name,v in (("D0_center_IF",v0e[ae]),("operator_center_IF",vr_e[ae])):
            cov_rows.append({"task":task_name,"bank":bank,"method":name,
                "predicted_variance_mean":float(v.mean()),"AUDIT_empirical_variance_mean":float(audit_var.mean()),
                "correlation_with_AUDIT_variance":float(np.corrcoef(v,audit_var)[0,1]) if np.std(v)>0 and np.std(audit_var)>0 else None,
                "variance_CPU_seconds":if_cpu,"D0_hessian_inverse_CPU_seconds_included":if_cpu,
                "C_over_V0_design":None,"VR_over_V0_design":None,
                "variance_target":"conditional_multinomial_bootstrap"})
        _write_new(bdir/"predictions.npz",_npz_bytes(z0c=z0c,z0e=z0e,zic=zic,zie=zie,zcalref=zcalref,zaudit=zaudit,v0c=v0c,v0e=v0e,vr_c=vr_c,vr_e=vr_e))
        _write_json(bdir/"calibrators.json",maps)
        bank_rows.append({"bank":bank,"train_n":len(tr),"feature_dimension":int(xt.shape[1]),
            "threshold_tau":float(expit(gamma)),"CAL_alerts":int(ac.sum()),"EVAL_alerts":int(ae.sum()),
            "sigma_g_sq":sigma2,"calibrator_fits":len(maps),"ridge_fits":2,
            "hessian_and_if_CPU_seconds":if_cpu,"model_inference_CPU_seconds":inference_cpu,
            "six_INT_inference_CPU_seconds":int_infer_cpu,
            "six_INT_fit_CPU_seconds":sum(r["fit_cpu_seconds"] for r in fit_records if r["bank"]==bank and r["model_id"].startswith("INT-")),
            "audit_variance_mean":float(audit_var.mean()),
            "C_over_V0_design":None,"VR_over_V0_design":None,
            "variance_target":"conditional_multinomial_bootstrap",
            "operator_equals_D0_for_bootstrap":True})
        print(f"W2-B bank {bank} complete; fits={len(specs)}; IF CPU={if_cpu:.3f}s",flush=True)
    _write_new(out/"per_bank_metrics.csv",_csv_bytes(method_rows))
    fields=[k for k in method_rows[0] if k not in ("task","bank","method")]
    aggregate=_aggregate(method_rows,["task","method"],fields)
    _write_new(out/"mean_range_metrics.csv",_csv_bytes(aggregate))
    _write_new(out/"if_variance_diagnostics.csv",_csv_bytes(cov_rows))
    report={"status":"W2B_DIABETES_LOGISTIC_IF_COMPLETED" if not fulltrain else "W2B_DIABETES_LOGISTIC_FULLTRAIN_IF_COMPLETED","split_audit":audit,
        "new_Logistic_model_fits":len(fit_records),"fit_records":fit_records,
        "new_Ridge_fits":6,"new_CALREF_map_fits":sum(x["calibrator_fits"] for x in bank_rows),
        "banks":bank_rows,"mean_range_metrics":aggregate,"variance_diagnostics":cov_rows,
        "model_fit_CPU_seconds":sum(r["fit_cpu_seconds"] for r in fit_records),
        "model_fit_wall_seconds_sum":sum(r["fit_wall_seconds"] for r in fit_records),
        "analysis_wall_seconds":time.monotonic()-process_wall,
        "analysis_CPU_seconds":time.process_time()-process_cpu,
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "limits":["all estimates are for public Diabetes and this finite bootstrap reference set","three banks share CAL/EVAL patients","analytic IF is first-order bootstrap covariance, not a distribution-free guarantee"]}
    _write_json(out/"report.json",report)
    return report


def _p3_saved_inputs(bank: int):
    cache_path=pub._p3_bank_dir(bank)/"prediction_cache/arrays.npz"
    with np.load(cache_path,allow_pickle=False) as cache:
        ncal=int(cache["cal_count"][0]);p0=cache["p_D0"].copy()
        ints=np.column_stack([cache[f"p_INT-{j:02d}"] for j in range(1,7)])
    if ncal!=13983 or p0.size!=27932:raise WorkError("P3 prediction cache does not match frozen split")
    gamma=float(logit(np.quantile(p0[:ncal],.8)))
    z0=pub.safe_logit(p0)-gamma;zi=pub.safe_logit(ints)-gamma
    c=z0[:ncal]>=0;e=z0[ncal:]>=0
    with np.load(P3_FOLLOW/f"bank{bank}/scores.npz",allow_pickle=False) as saved:
        tcal=saved["tcal"].copy();audit_eval=saved["audit"].copy()
        saved_scores={k:saved[k].copy() for k in saved.files if k not in ("tcal","audit")}
    if len(tcal)!=int(c.sum()) or audit_eval.shape!=(int(e.sum()),11):
        raise WorkError("P3 CALREF/AUDIT outputs mismatch D0 queues")
    return ncal,p0,zi,z0,gamma,c,e,tcal,audit_eval,saved_scores


def p3_calref_upgrade() -> dict:
    """Add prior-locked tie-break and calibrated controls to the 14k P3 bank."""
    out=OUT/"p3_calref_upgrade"
    rows=[];reports=[]
    for bank in range(3):
        ncal,p0,zi,z0,gamma,c,e,tcal,audit,saved=_p3_saved_inputs(bank)
        z0c,z0e=z0[:ncal][c],z0[ncal:][e]
        zic,zie=zi[:ncal][c],zi[ncal:][e]
        zbc,zbe=zic.mean(axis=1),zie.mean(axis=1)
        sigma2=float(np.var(zic,axis=1,ddof=1).mean())
        vtilde=.5*np.var(zic,axis=1,ddof=1)+.5*sigma2
        qlexc=(np.sum(zic<0,axis=1)+.25+.5*expit(-z0c))/7
        qlexe=(np.sum(zie<0,axis=1)+.25+.5*expit(-z0e))/7
        lex_eval,lex_map=_cal_stability(-qlexc,tcal,-qlexe)
        qg_c=ndtr(-z0c/np.sqrt(sigma2));qg_e=ndtr(-z0e/np.sqrt(sigma2))
        qs_c=ndtr(-z0c/np.sqrt(2*sigma2));qs_e=ndtr(-z0e/np.sqrt(2*sigma2))
        qg_eval,gmap=_cal_stability(-qg_c,tcal,-qg_e)
        qs_eval,smap=_cal_stability(-qs_c,tcal,-qs_e)
        existing_maps=json.loads((P3_FOLLOW/f"bank{bank}/calibrators.json").read_text())
        mean_map=json.loads((P3_FOLLOW/f"bank{bank}/shrunk_mean_ablation.json").read_text())["calibrator"]
        mean_qe=_apply_cal_map(mean_map,.5*z0e+.5*zbe)
        mean_qc=_apply_cal_map(mean_map,.5*z0c+.5*zbc)
        # Existing report uses equivalent CALREF maps for these methods.
        qmethod={"margin_logistic":saved["margin_calibrated"],
            "int_mean_logistic":saved["int_mean_calibrated"],
            "fusion_mean_only":mean_qe,"qV":saved["qV_calibrated"],
            "qMV":saved["qMV_calibrated"],"q_MV_surrogate":saved["surrogate_calibrated"],
            "int_freq":saved["int_frequency_calibrated"],
            "int_freq_margin_tiebreak":lex_eval,"q_global_1_CALREF":qg_eval,
            "q_sqrt2_CALREF":qs_eval}
        for method,q in qmethod.items():
            rows.append({"task":"P3_Diabetes_14k_CALREF","bank":bank,"method":method,
                         **_metrics(q,audit,qmethod["margin_logistic"],qmethod["q_sqrt2_CALREF"])})
        _write_json(out/f"bank{bank}_new_maps.json",{"int_freq_margin_tiebreak":lex_map,
            "q_global_1":gmap,"q_sqrt2":smap})
        reports.append({"bank":bank,"CAL_alerts":int(c.sum()),"EVAL_alerts":int(e.sum()),
            "sigma_g_sq_CAL":sigma2,"new_CALREF_maps":3,
            "CALREF_targets":"six separately saved CALREF model flips; CAL only"})
    fields=[k for k in rows[0] if k not in ("task","bank","method")]
    _write_new(out/"per_bank_metrics.csv",_csv_bytes(rows))
    agg=_aggregate(rows,["task","method"],fields)
    _write_new(out/"mean_range_metrics.csv",_csv_bytes(agg))
    report={"status":"P3_CALREF_BASELINE_UPGRADE_COMPLETED","banks":reports,
        "new_prediction_model_fits":0,"new_inference_batches":0,"new_CALREF_map_fits":9,
        "mean_range_metrics":agg,
        "calibration":"all reported methods use CALREF-only two-parameter monotone Brier maps; existing maps reused where already present",
        "limitations":["same public CAL/EVAL patients and original 14k training-bank models","three banks are not independent patient cohorts"]}
    _write_json(out/"report.json",report)
    return report


def _fractional_pick(score: np.ndarray, mass: float,
                     available: np.ndarray | None=None) -> np.ndarray:
    """Select exact mass by score, splitting exact ties proportionally."""
    s=np.asarray(score,float);a=np.ones(len(s),float) if available is None else np.asarray(available,float)
    if s.ndim!=1 or a.shape!=s.shape or not np.isfinite(s).all() or np.any(a< -1e-12):
        raise WorkError("invalid fractional-selection inputs")
    mass=float(mass)
    if mass< -1e-9 or mass>a.sum()+1e-7:raise WorkError(f"selection mass {mass} outside available {a.sum()}")
    mass=min(max(mass,0.0),float(a.sum()))
    out=np.zeros(len(s),float)
    if mass==0:return out
    vals=np.unique(s)
    left=mass
    for val in vals:
        idx=np.flatnonzero(s==val);cap=float(a[idx].sum())
        if cap<=left+1e-12:
            out[idx]=a[idx];left-=cap
        else:
            frac=left/cap
            out[idx]=a[idx]*frac;left=0.;break
    if abs(out.sum()-mass)>1e-7:raise WorkError("fractional tie selection failed mass conservation")
    return out


def _banded_weights(z0: np.ndarray,q: np.ndarray,rho: float,c: float=.70) -> np.ndarray:
    n=len(z0)
    if rho>=c:raise WorkError("guardrail rho must be less than total coverage")
    protected=_fractional_pick(-np.asarray(z0),rho*n)
    remaining=np.maximum(1-protected,0)
    boundary=_fractional_pick(q,(c-rho)*n,remaining)
    return protected+boundary


def _utility_weights(q: np.ndarray,poutcome: np.ndarray,lam: float,c: float) -> np.ndarray:
    score=(1-lam)*np.asarray(q)-lam*np.asarray(poutcome)
    return _fractional_pick(score,c*len(score))


def _outcome_metric(w: np.ndarray,y: np.ndarray,t: np.ndarray,
                    total_population_positives: float | None=None) -> dict:
    w=np.asarray(w,float);y=np.asarray(y,float);t=np.asarray(t,float)
    mass=float(w.sum());total=float(len(w))
    if mass<=0:raise WorkError("empty retained alert set")
    tp=float(np.dot(w,y));yp=float(y.sum())
    if total_population_positives is not None and total_population_positives<yp:
        raise WorkError("population positive count is smaller than D0-alert positives")
    return {"retained_fraction":mass/total,"R":float(np.dot(w,t)/mass),
        "PPV":tp/mass,"D0_true_positive_retention":tp/yp if yp>0 else None,
        "population_sensitivity":tp/total_population_positives if total_population_positives else None,
        "retained_positive_mass":tp,"total_positive_alerts":yp}


def _load_split_y_only(split: str) -> tuple[list[str],np.ndarray]:
    """Recover outcomes only for selected patients in one requested split.

    First pass reads IDs/disposition only and deterministically selects each
    patient's encounter.  The second pass accesses `readmitted` only inside
    the requested split branch, so W3 selection cannot inspect EVAL labels.
    """
    if split not in ("CAL","EVAL","TRAIN"):raise WorkError("unknown frozen patient split")
    choices={}
    with DIABETES_CSV.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for row in reader:
            disp=int(row["discharge_disposition_id"])
            if disp in pub.DISPOSITION_EXCLUDED:continue
            patient=str(row["patient_nbr"]);encounter=str(row["encounter_id"])
            h=pub.stable_group_hash(patient,encounter)
            old=choices.get(patient)
            if old is None or (h,encounter)<old:choices[patient]=(h,encounter)
    selected={}
    for patient,(_,encounter) in choices.items():
        mod=pub.stable_hash_int(f"split_v2_{patient}")%10000
        role="TRAIN" if mod<6000 else ("CAL" if mod<8000 else "EVAL")
        if role==split:selected[patient]=encounter
    labels={}
    with DIABETES_CSV.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for row in reader:
            patient=str(row["patient_nbr"])
            if patient not in selected or str(row["encounter_id"])!=selected[patient]:continue
            # Only now, after the requested split check, read the outcome value.
            labels[patient]=int(row["readmitted"]=="<30")
    if labels.keys()!=selected.keys():raise WorkError(f"missing selected outcome rows for split={split}")
    ordered=sorted(labels)
    return ordered,np.asarray([labels[p] for p in ordered],dtype=np.int8)


def _w3_bank_arrays(bank: int):
    ncal,p0,zi,z0,gamma,c,e,tcal,audit,saved=_p3_saved_inputs(bank)
    z0c,z0e=z0[:ncal][c],z0[ncal:][e]
    zic,zie=zi[:ncal][c],zi[ncal:][e]
    maps=json.loads((P3_FOLLOW/f"bank{bank}/calibrators.json").read_text())
    fusion_map=json.loads((P3_FOLLOW/f"bank{bank}/shrunk_mean_ablation.json").read_text())["calibrator"]
    sig2=float(np.var(zic,axis=1,ddof=1).mean())
    stc=.5*(z0c+zic.mean(axis=1))/np.sqrt(.5*np.var(zic,axis=1,ddof=1)+.5*sig2)
    ste=.5*(z0e+zie.mean(axis=1))/np.sqrt(.5*np.var(zie,axis=1,ddof=1)+.5*sig2)
    qmv_c=_apply_cal_map(maps["qMV_calibrated"],stc)
    qmv_e=_apply_cal_map(maps["qMV_calibrated"],ste)
    mean_c=_apply_cal_map(fusion_map,.5*(z0c+zic.mean(axis=1)))
    mean_e=_apply_cal_map(fusion_map,.5*(z0e+zie.mean(axis=1)))
    # Corresponding CAL/EVAL D0 outcome probabilities; no EVAL labels are touched here.
    p0=expit(z0+gamma)
    pmean=expit(zi.mean(axis=1)+gamma)
    ac=p0[:ncal][c];ae=p0[ncal:][e]
    mc=pmean[:ncal][c];me=pmean[ncal:][e]
    # Current outcome labels are accessed only in the distinct evaluation stage.
    return {"z0c":z0c,"z0e":z0e,"qmv_c":qmv_c,"qmv_e":qmv_e,
        "mean_c":mean_c,"mean_e":mean_e,"p0c":ac,"p0e":ae,
        "pmeanc":mc,"pmeane":me,"tcal":tcal,"audit":audit,
        "cal_alerts":int(c.sum()),"eval_alerts":int(e.sum()),
        "cal_alert_index":np.flatnonzero(c),"eval_index":np.flatnonzero(e),
        "ncal":ncal,"maps":maps,"fusion_map":fusion_map,
        "gamma":gamma}


def _candidate_configs(arr: dict,bank: int,cal_y: np.ndarray) -> tuple[list[dict],dict]:
    n=arr["cal_alerts"];z=arr["z0c"];y=np.asarray(cal_y,int);t=arr["tcal"]
    if y.shape!=(n,):raise WorkError("CAL outcome vector does not align to CAL D0 alert rows")
    margin_w=_fractional_pick(-z,.70*n)
    margin=_outcome_metric(margin_w,y,t)
    q_methods={"qMV":arr["qmv_c"],"fusion_mean_only":arr["mean_c"]}
    rows=[]
    for meth,q in q_methods.items():
        rows.append({"bank":bank,"config_id":f"pure_{meth}","strategy":"pure_stability","stability_method":meth,
            "rho":None,"lambda":None,"outcome_score":None,**_outcome_metric(_fractional_pick(q,.70*n),y,t)})
        for rho in (.30,.40,.50):
            rows.append({"bank":bank,"config_id":f"banded_{meth}_rho{rho:.2f}","strategy":"banded",
                "stability_method":meth,"rho":rho,"lambda":None,"outcome_score":None,
                **_outcome_metric(_banded_weights(z,q,rho,.70),y,t)})
        for outcome_name,pout in (("p_D0",arr["p0c"]),("p_INT_mean",arr["pmeanc"])):
            for lam in (0.,.1,.2,.3,.5,.7,1.):
                weights=_utility_weights(q,pout,lam,.70)
                rows.append({"bank":bank,"config_id":f"utility_{meth}_{outcome_name}_lambda{lam:.1f}",
                    "strategy":"utility_weighted","stability_method":meth,"rho":None,
                    "lambda":lam,"outcome_score":outcome_name,**_outcome_metric(weights,y,t)})
    eligible=[r for r in rows if r["strategy"] in ("banded","utility_weighted") and
              r["PPV"]>=margin["PPV"]-1e-4]
    if not eligible:raise WorkError("no W3 CAL candidate meets the locked PPV non-inferiority constraint")
    chosen=sorted(eligible,key=lambda r:(r["R"],-r["PPV"],r["config_id"]))[0]
    for r in rows:
        r["meets_PPv_constraint"]=bool(r["PPV"]>=margin["PPV"]-1e-4)
        r["pareto_nondominated"]=not any((x["R"]<=r["R"]+1e-14 and x["PPV"]>=r["PPV"]-1e-14 and
            (x["R"]<r["R"]-1e-14 or x["PPV"]>r["PPV"]+1e-14)) for x in rows)
        r["is_cal_selected"]=r["config_id"]==chosen["config_id"]
    return rows,{"bank":bank,"margin_cal_PPv":margin["PPV"],"constraint_floor":margin["PPV"]-1e-4,
        "selection_rule":"minimize CAL R70 among PPV-feasible candidates; tie-break higher PPV, then lexical config_id",
        "selected":chosen}


def w3_select() -> dict:
    out=OUT/"w3_utility_tradeoff"; frozen=out/"cal_frozen_selection.json"
    if frozen.exists() or frozen.is_symlink():raise WorkError("W3 CAL selection already frozen; refusing overwrite")
    all_cal=[];selections=[];inputs=[]
    cal_patient_ids,cal_y_full=_load_split_y_only("CAL")
    manifest_path=pub.OUT_ROOT/"diabetes_data_audit/selected_patient_split_manifest.csv"
    manifest=pd.read_csv(manifest_path,usecols=["patient_key_sha256","split"],dtype=str)
    expected_cal=manifest.loc[manifest.split=="CAL","patient_key_sha256"].tolist()
    actual_cal=[pub.sha_text("patient:"+p) for p in cal_patient_ids]
    if actual_cal!=expected_cal:raise WorkError("CAL-only outcome stream order differs from frozen split manifest")
    for bank in range(3):
        arr=_w3_bank_arrays(bank)
        if len(cal_patient_ids)!=arr["ncal"]:raise WorkError("CAL patient order/count differs from saved P3 cache")
        # The P3 cache follows the frozen numeric-patient sort.  Alert labels
        # are selected from the CAL-only vector using the same D0 alert mask.
        cal_labels=cal_y_full[arr["cal_alert_index"]]
        rows,sel=_candidate_configs(arr,bank,cal_labels)
        all_cal.extend(rows);selections.append(sel)
        inputs.append({"bank":bank,"CAL_alerts":arr["cal_alerts"],"EVAL_alerts":arr["eval_alerts"],
            "tcal_mean":float(arr["tcal"].mean()),"cal_y_count":int(cal_labels.sum()),
            "p3_prediction_cache_sha256":_sha_file(pub._p3_bank_dir(bank)/"prediction_cache/arrays.npz"),
            "CALREF_score_cache_sha256":_sha_file(P3_FOLLOW/f"bank{bank}/scores.npz")})
    chosen_doc={"status":"CAL_SELECTION_FROZEN_BEFORE_EVAL_LABEL_USE",
        "authorization":"current user W1-W4 message","selection_population":"CAL alert rows only",
        "CAL_outcomes_used":True,"EVAL_outcomes_used":False,
        "candidate_count_per_bank":len(all_cal)//3,"rho_grid":[.30,.40,.50],
        "lambda_grid":[0.,.1,.2,.3,.5,.7,1.0],"stability_candidates":["qMV","fusion_mean_only"],
        "outcome_scores":["p_D0","p_INT_mean"],"margin_baseline_R70":selections,
        "selected_by_bank":[s["selected"] for s in selections],"provenance":inputs,
        "selection_metrics":all_cal}
    _write_json(frozen,chosen_doc)
    _write_new(out/"cal_candidates_and_frontier.csv",_csv_bytes(all_cal))
    report={k:v for k,v in chosen_doc.items() if k!="selection_metrics"}
    report["candidate_count_total"]=len(all_cal)
    _write_json(out/"cal_selection_report.json",report)
    return report


def _eval_strategy_weights(config: dict,arr: dict,c: float) -> np.ndarray:
    z=arr["z0e"];qm=arr["qmv_e"];fm=arr["mean_e"]
    if config["strategy"]=="pure_stability":
        q=qm if config["stability_method"]=="qMV" else fm
        return _fractional_pick(q,c*len(z))
    if config["strategy"]=="banded":
        q=qm if config["stability_method"]=="qMV" else fm
        return _banded_weights(z,q,float(config["rho"]),c)
    if config["strategy"]=="utility_weighted":
        if config["outcome_score"]=="p_D0":pout=arr["p0e"]
        else:pout=arr["pmeane"]
        q=qm if config["stability_method"]=="qMV" else fm
        return _utility_weights(q,pout,float(config["lambda"]),c)
    raise WorkError("unknown frozen W3 strategy")


def w3_eval() -> dict:
    out=OUT/"w3_utility_tradeoff";frozen=out/"cal_frozen_selection.json"
    if not frozen.is_file() or frozen.is_symlink():raise WorkError("W3 evaluation requires frozen CAL selection")
    selection=json.loads(frozen.read_text())
    if selection.get("status")!="CAL_SELECTION_FROZEN_BEFORE_EVAL_LABEL_USE" or selection.get("EVAL_outcomes_used") is not False:
        raise WorkError("W3 frozen selection manifest invalid")
    eval_patient_ids,eval_y_full=_load_split_y_only("EVAL")
    manifest_path=pub.OUT_ROOT/"diabetes_data_audit/selected_patient_split_manifest.csv"
    manifest=pd.read_csv(manifest_path,usecols=["patient_key_sha256","split"],dtype=str)
    expected_eval=manifest.loc[manifest.split=="EVAL","patient_key_sha256"].tolist()
    actual_eval=[pub.sha_text("patient:"+p) for p in eval_patient_ids]
    if actual_eval!=expected_eval:raise WorkError("EVAL outcome stream order differs from frozen split manifest")
    rows=[];summary=[]
    for bank in range(3):
        arr=_w3_bank_arrays(bank)
        if len(eval_patient_ids)!=13949:raise WorkError("EVAL patient count differs from frozen partition")
        y=eval_y_full[arr["eval_index"]]
        t=arr["audit"].mean(axis=1)
        pure=[("margin",_fractional_pick(-arr["z0e"],.70*len(y))),
              ("qMV",_fractional_pick(arr["qmv_e"],.70*len(y))),
              ("fusion_mean_only",_fractional_pick(arr["mean_e"],.70*len(y)))]
        chosen=selection["selected_by_bank"][bank]
        configs=[c for c in selection["selection_metrics"] if c["bank"]==bank]
        # Full EVAL Pareto table is descriptive only; selection remains the frozen CAL choice.
        for config in configs:
            for cov in (.70,.85):
                w=_eval_strategy_weights(config,arr,cov)
                m=_outcome_metric(w,y,t,float(eval_y_full.sum()))
                rows.append({"bank":bank,"config_id":config["config_id"],"strategy":config["strategy"],
                    "stability_method":config["stability_method"],"rho":config["rho"],
                    "lambda":config["lambda"],"outcome_score":config["outcome_score"],
                    "coverage":cov,"selected_on_CAL":config["config_id"]==chosen["config_id"],**m})
        for name,w in pure:
            for cov in (.70,.85):
                if cov!=.70:
                    if name=="margin":w=_fractional_pick(-arr["z0e"],cov*len(y))
                    elif name=="qMV":w=_fractional_pick(arr["qmv_e"],cov*len(y))
                    else:w=_fractional_pick(arr["mean_e"],cov*len(y))
                m=_outcome_metric(w,y,t,float(eval_y_full.sum()))
                rows.append({"bank":bank,"config_id":f"pure_{name}","strategy":"pure_reference",
                    "stability_method":name,"rho":None,"lambda":None,"outcome_score":None,
                    "coverage":cov,"selected_on_CAL":False,**m})
        cm=next(r for r in rows if r["bank"]==bank and r["config_id"]==chosen["config_id"] and r["coverage"]==.70)
        summary.append({"bank":bank,"cal_selected_config":chosen["config_id"],
            "eval_R70":cm["R"],"eval_PPV70":cm["PPV"],
            "eval_D0_true_positive_retention70":cm["D0_true_positive_retention"],
            "eval_population_sensitivity70":cm["population_sensitivity"],
            "eval_alerts":len(y),"eval_readmissions_in_alerts":int(y.sum()),
            "evaluation_labels_accessed_after_cal_freeze":True})
    # Mark nondominated EVAL points within bank and coverage, without using them to select.
    for cov in (.70,.85):
        group=[r for r in rows if r["coverage"]==cov]
        for r in group:
            peers=[x for x in group if x["bank"]==r["bank"]]
            r["eval_pareto_nondominated"]=not any((x["R"]<=r["R"]+1e-14 and x["PPV"]>=r["PPV"]-1e-14 and
                (x["R"]<r["R"]-1e-14 or x["PPV"]>r["PPV"]+1e-14)) for x in peers)
    _write_new(out/"eval_full_pareto.csv",_csv_bytes(rows))
    _write_new(out/"selected_config_eval_summary.csv",_csv_bytes(summary))
    report={"status":"W3_CAL_SELECTED_EVAL_EVALUATED","CAL_selection_sha256":_sha_file(frozen),
        "selected_config_eval":summary,"eval_rows":len(rows),
        "selection_used_EVAL_labels":False,"EVAL_labels_read_only_after_selection_freeze":True,
        "full_pareto_table":"eval_full_pareto.csv; descriptive, never used to select"}
    _write_json(out/"eval_report.json",report)
    return report


def w2b_inference_cost_replay(fulltrain: bool=False) -> dict:
    """Time only the six saved INT coefficient-vector evaluations; no refits."""
    suffix="_fulltrain" if fulltrain else ""
    out=OUT/f"w2b_diabetes_logistic{suffix}"/"inference_cost_replay.json"
    if out.exists() or out.is_symlink():raise WorkError("W2-B inference replay output exists")
    selected,_audit=pub.load_selected_diabetes()
    train=selected[selected.split=="TRAIN"].reset_index(drop=True)
    cal=selected[selected.split=="CAL"].reset_index(drop=True)
    eva=selected[selected.split=="EVAL"].reset_index(drop=True)
    rows=[];total_cpu=0.;total_batches=0
    for bank in range(3):
        if fulltrain:
            rng=np.random.default_rng(SEED_BASE+bank*10000)
            take=np.sort(rng.choice(len(train),size=LGB_FULL_N,replace=False))
            tr=train.iloc[take].reset_index(drop=True)
        else:
            tr=train[train.bank_id==bank].sort_values("patient_nbr",kind="mergesort").reset_index(drop=True)
        prep=pub.fit_diabetes_preprocessor(tr)
        fitted=pub.fit_sparse_proxy_preprocessor(tr,prep)
        xc=_diabetes_sparse_features(cal,prep,fitted);xe=_diabetes_sparse_features(eva,prep,fitted)
        saved_path=RUNS/f"w2b{suffix}/bank{bank}/predictions.npz"
        with np.load(saved_path,allow_pickle=False) as saved:
            zic=saved["zic"].copy();zie=saved["zie"].copy()
        start_wall,start_cpu=time.monotonic(),time.process_time();max_error=0.
        for j in range(1,7):
            model_path=RUNS/f"w2b{suffix}/bank{bank}/models/INT-{j:02d}/model.json"
            model=json.loads(model_path.read_text())
            beta=np.asarray(model["beta"],float)
            rawc=_logit_l2(xc,beta);rawe=_logit_l2(xe,beta)
            offset=float(np.median(rawc-zic[:,j-1]))
            max_error=max(max_error,float(np.max(np.abs(rawc-offset-zic[:,j-1]))),
                          float(np.max(np.abs(rawe-offset-zie[:,j-1]))))
            total_batches+=2
        cpu=time.process_time()-start_cpu;wall=time.monotonic()-start_wall;total_cpu+=cpu
        if max_error>1e-7:raise WorkError(f"W2-B saved INT replay mismatch bank={bank}, max={max_error}")
        rows.append({"bank":bank,"INT_models":6,"prediction_batches":12,
            "inference_CPU_seconds":cpu,"inference_wall_seconds":wall,
            "mean_CPU_seconds_per_INT_model":cpu/6,"saved_centered_logit_max_abs_difference":max_error,
            "feature_preprocessor_fit_rows":len(tr),"model_fits":0})
    report={"status":"W2B_INT_INFERENCE_COST_REPLAY_COMPLETE" if not fulltrain else "W2B_FULLTRAIN_INT_INFERENCE_COST_REPLAY_COMPLETE","new_model_fits":0,
        "preprocessing_reconstruction_fits":3,"prediction_batches":total_batches,
        "total_INT_inference_CPU_seconds":total_cpu,"banks":rows,
        "original_W2B_model_prediction_batches":93,
        "original_W2B_INT_prediction_batches":36,
        "measurement":"CPU time for 6 saved coefficient-vector predictions on CAL and EVAL matrices; preprocessing rebuild excluded; outputs reconciled to saved centered INT logits",
        "outcome_labels_used":False}
    _write_json(out,report)
    return report


def w1_p2_if_audit_replay() -> dict:
    """Add empirical AUDIT-variance correlations and CPU timing to W1 P2 IF rows."""
    out=OUT/"w1_p2"/"if_audit_diagnostics_replay.csv"
    if out.exists() or out.is_symlink():raise WorkError("P2 IF replay output exists")
    rows=[];total_cpu=0.
    for bank in range(3):
        a,b,fit,xa,xb,xc,xe,parts=_adult_p2_design(bank)
        beta0,_=pub._p2_load_beta(bank,"shared","D0")
        xall=np.vstack((xc,xe));ncal=len(xc);z0=xall@beta0
        eval_alert=z0[ncal:]>=0;ya=pub._group_soft_labels(a);yb=pub._group_soft_labels(b)
        for op,alpha in (("alpha_0",0.),("alpha_0_5",.5),("bootstrap",None)):
            bundle=P2_FOLLOW/f"bank{bank}/{op}.npz"
            with np.load(bundle,allow_pickle=False) as saved:logits=saved["logits"].copy()
            audvar=np.var(logits[ncal:,12:][eval_alert],axis=1,ddof=1)
            if op=="bootstrap":beta_r=beta0.copy()
            else:beta_r=np.asarray(pub.read_json(P2_FOLLOW/f"bank{bank}/{op}_anchor.json")["beta"],float)
            t0=time.process_time()
            v0=_pool_a_if_variance(xa,ya,beta0,xall)[ncal:][eval_alert]
            _,vr=follow.anchor_if(xa,ya,xb,yb,beta_r,xall,
                0. if op=="bootstrap" else alpha,op=="bootstrap")
            vr=vr[ncal:][eval_alert]
            cpu=time.process_time()-t0;total_cpu+=cpu
            corr0=float(np.corrcoef(v0,audvar)[0,1]) if np.std(v0)>0 and np.std(audvar)>0 else None
            corrr=float(np.corrcoef(vr,audvar)[0,1]) if np.std(vr)>0 and np.std(audvar)>0 else None
            rows.append({"task":"P2_Adult_Logistic","bank":bank,"operation":op,
                "D0_center_IF_variance_mean":float(v0.mean()),"operator_center_IF_variance_mean":float(vr.mean()),
                "AUDIT_empirical_variance_mean":float(audvar.mean()),
                "corr_D0_IF_variance_vs_AUDIT":corr0,"corr_operator_IF_variance_vs_AUDIT":corrr,
                "D0_vs_operator_variance_corr":float(np.corrcoef(v0,vr)[0,1]) if np.std(v0)>0 and np.std(vr)>0 else None,
                "C_over_V0_design_note":"see finite-pool IF design; this row reports variance prediction accuracy, not an assumed universal ratio",
                "IF_CPU_seconds":cpu,"model_fits":0})
    _write_new(out,_csv_bytes(rows))
    report={"status":"P2_IF_AUDIT_VARIANCE_REPLAY_COMPLETE","new_model_fits":0,
        "IF_evaluations":18,"total_CPU_seconds":total_cpu,"rows":rows,
        "source":"saved corrected P2 reference coefficients/logits and fixed A/B pools"}
    _write_json(out.with_name("if_audit_diagnostics_replay.json"),report)
    return report


def w4_assemble() -> dict:
    # Versioned assembly: earlier W4 snapshots are retained unchanged.
    out=OUT/"w4_tables_v3"
    tables=[]
    # P1 is ranking-only because the saved P1 role bank has no separate CALREF models.
    p1rows=pd.read_csv(OUT/"w1_p1/ranking_ablation.csv")
    for _,r in p1rows.iterrows():
        tables.append({"task":r.cohort,"operation":"P1_LightGBM","method":r.method,
            "R_070":r.R_070,"AURC100":r.AURC100,"AUC":r.AUC_flip,
            "Brier":None,"ECE10":None,"CALREF_calibrated":False})
    # Corrected P2 finite-pool banks: the same CALREF map family on every method.
    p2=pd.read_csv(OUT/"w1_p2/per_bank_metrics.csv")
    keep={"margin_logistic":"Margin","int_mean_logistic":"INT均值","fusion_mean_only":"仅融合均值",
        "operator_center_mean":"算子中心均值","qV":"仅方差 qV","D0_center_IF":"D0中心IF",
        "qMV":"局部转移 qMV","operator_center_IF":"算子中心IF",
        "int_freq":"INT翻转频率","int_freq_margin_tiebreak":"INT频率+Margin破并列"}
    for _,r in p2.iterrows():
        if r.method in keep:
            tables.append({"task":f"P2_Adult_Logistic_bank{int(r.bank)}","operation":r.operation,
                "method":keep[r.method],"R_070":r.R_070,"AURC100":r.AURC100,"AUC":r.AUC_flip,
                "Brier":r.Brier_finite_AUDIT,"ECE10":r.ECE10,"CALREF_calibrated":True})
    p3=pd.read_csv(OUT/"p3_calref_upgrade/per_bank_metrics.csv")
    p3keep={"margin_logistic":"Margin","int_mean_logistic":"INT均值","fusion_mean_only":"仅融合均值",
        "qV":"仅方差 qV","qMV":"局部转移 qMV","q_MV_surrogate":"零INT代理",
        "int_freq":"INT翻转频率","int_freq_margin_tiebreak":"INT频率+Margin破并列"}
    for _,r in p3.iterrows():
        if r.method in p3keep:
            tables.append({"task":f"Diabetes_LGB_14k_bank{int(r.bank)}","operation":"P3_bootstrap",
                "method":p3keep[r.method],"R_070":r.R_070,"AURC100":r.AURC100,"AUC":r.AUC_flip,
                "Brier":r.Brier_finite_AUDIT,"ECE10":r.ECE10,"CALREF_calibrated":True})
    for stage,task in (("w2a_diabetes_lgb_fulltrain","Diabetes_LGB_42k"),
                       ("w2b_diabetes_logistic","Diabetes_Logistic_14k"),
                       ("w2b_diabetes_logistic_fulltrain","Diabetes_Logistic_42k")):
        df=pd.read_csv(OUT/stage/"per_bank_metrics.csv")
        allowed={"margin_logistic":"Margin","int_mean_logistic":"INT均值","fusion_mean_only":"仅融合均值",
            "operator_center_mean":"算子中心均值","qV":"仅方差 qV","D0_center_IF":"D0中心IF",
            "qMV":"局部转移 qMV","operator_center_IF":"算子中心IF",
            "q_MV_surrogate":"零INT代理","ridge_surrogate":"零INT代理",
            "int_freq":"INT翻转频率","int_freq_margin_tiebreak":"INT频率+Margin破并列"}
        for _,r in df.iterrows():
            if r.method in allowed:
                tables.append({"task":f"{task}_bank{int(r.bank)}","operation":"bootstrap",
                    "method":allowed[r.method],"R_070":r.R_070,"AURC100":r.AURC100,"AUC":r.AUC_flip,
                    "Brier":r.Brier_finite_AUDIT,"ECE10":r.ECE10,"CALREF_calibrated":True})
        if task == "Diabetes_LGB_42k":
            # Under the bootstrap operator, E[training objective] is the
            # original D0 objective. Preserve that identity as an explicit
            # center-only row rather than implying a distinct fit.
            for bank in sorted(df.bank.unique()):
                r=df[(df.bank==bank)&(df.method=="margin_logistic")].iloc[0]
                tables.append({"task":f"{task}_bank{int(bank)}","operation":"bootstrap",
                    "method":"算子中心均值","R_070":r.R_070,"AURC100":r.AURC100,
                    "AUC":r.AUC_flip,"Brier":r.Brier_finite_AUDIT,"ECE10":r.ECE10,
                    "CALREF_calibrated":True,"identity":"operator center beta_R equals beta_D0 under bootstrap"})
    fields=["task","operation","method","R_070","AURC100","AUC","Brier","ECE10","CALREF_calibrated"]
    _write_new(out/"table1_center_variance_ablation.csv",_csv_bytes(tables,fields))
    aggregate_input=pd.DataFrame(tables)
    aggregate_input["task"]=aggregate_input["task"].str.replace(r"_bank\d+$","",regex=True)
    aggregate_frame=aggregate_input.groupby(
        ["task","operation","method","CALREF_calibrated"],dropna=False
    )[["R_070","AURC100","AUC","Brier","ECE10"]].agg(["mean","min","max"]).reset_index()
    aggregate_frame.columns=["_".join(str(x) for x in col if x!="").rstrip("_") if isinstance(col,tuple) else col
                             for col in aggregate_frame.columns]
    _write_new(out/"table1_mean_range.csv",aggregate_frame.to_csv(index=False).encode())
    # P2/W2-B variance diagnostics. The theoretical covariance ratios remain design facts.
    p2v=pd.read_csv(OUT/"w1_p2/if_audit_diagnostics_replay.csv")
    w2v_frames=[pd.read_csv(OUT/"w2b_diabetes_logistic/if_variance_diagnostics.csv"),
                pd.read_csv(OUT/"w2b_diabetes_logistic_fulltrain/if_variance_diagnostics.csv")]
    table2=[]
    for _,r in p2v.iterrows():
        for method,vname,corrname in (("D0_center_IF","D0_center_IF_variance_mean","corr_D0_IF_variance_vs_AUDIT"),
                                      ("operator_center_IF","operator_center_IF_variance_mean","corr_operator_IF_variance_vs_AUDIT")):
            table2.append({"task":"P2_Adult","bank":r.bank,"operation":r.operation,"method":method,
                "predicted_variance_mean":r[vname],"AUDIT_variance_mean":r.AUDIT_empirical_variance_mean,
                "correlation_with_AUDIT_variance":r[corrname],"D0_vs_operator_variance_corr":r.D0_vs_operator_variance_corr,
                "IF_CPU_seconds":r.IF_CPU_seconds,"C_over_V0":None,"VR_over_V0":None})
    for w2v in w2v_frames:
        for _,r in w2v.iterrows():
            table2.append({"task":r.task,"bank":r.bank,"operation":"bootstrap","method":r.method,
                "predicted_variance_mean":r.predicted_variance_mean,
                "AUDIT_variance_mean":r.AUDIT_empirical_variance_mean,
                "correlation_with_AUDIT_variance":r.correlation_with_AUDIT_variance,
                "D0_vs_operator_variance_corr":None,"IF_CPU_seconds":r.variance_CPU_seconds,
                "C_over_V0":r.C_over_V0_design,"VR_over_V0":r.VR_over_V0_design})
    _write_new(out/"table2_if_diagnostics.csv",_csv_bytes(table2))
    w3=pd.read_csv(OUT/"w3_utility_tradeoff/eval_full_pareto.csv")
    w3selected=pd.read_csv(OUT/"w3_utility_tradeoff/selected_config_eval_summary.csv")
    population_positives=json.loads((pub.OUT_ROOT/"diabetes_data_audit/audit.json").read_text())[
        "split_patient_counts_and_positive_rates"]["EVAL"]["positive_count"]
    selected_ids={int(r.bank):r.cal_selected_config for _,r in w3selected.iterrows()}
    table3=[]
    for _,r in w3.drop_duplicates(["bank","config_id","coverage"]).iterrows():
        chosen=r.config_id==selected_ids[int(r.bank)]
        pure=r.config_id in ("pure_margin","pure_qMV","pure_fusion_mean_only")
        if chosen or pure:
            table3.append({"bank":int(r.bank),"config_id":r.config_id,"selected_on_CAL":bool(chosen),
                "coverage":float(r.coverage),"R_flip":r.R,"readmission_PPV":r.PPV,
                "D0_true_positive_retention":r.get("D0_true_positive_retention",r.get("Sensitivity")),
                "population_sensitivity":r.get("population_sensitivity",r.retained_positive_mass/population_positives),
                "pareto_nondominated_on_EVAL":r.eval_pareto_nondominated})
    _write_new(out/"table3_diabetes_outcome_pareto.csv",_csv_bytes(table3))
    w2b_cost={"14k":json.loads((OUT/"w2b_diabetes_logistic/inference_cost_replay.json").read_text()),
              "42k":json.loads((OUT/"w2b_diabetes_logistic_fulltrain/inference_cost_replay.json").read_text())}
    report={"status":"W4_TABLES_ASSEMBLED","table1_rows":len(tables),
        "table1":"table1_center_variance_ablation.csv","table2":"table2_if_diagnostics.csv",
        "table3":"table3_diabetes_outcome_pareto.csv",
        "table1_mean_range":"table1_mean_range.csv","supplemental":"../w4_tables/supplemental_proxy_center_diagnostics.json",
        "W2B_INT_inference_cost_replay":w2b_cost,
        "boundaries":["All results concern finite public Adult/Diabetes cohorts and finite model banks.",
            "P1 provides ranking endpoints only; its saved bank has no independent CALREF role for probability recalibration.",
            "P2/W2 calibration uses CALREF model flips on CAL; AUDIT is evaluation-only.",
            "No ICU or confirmation data are represented."]}
    _write_json(out/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["w1-p1","w1-p2","p3-upgrade","w2a","w2b","w2b-fulltrain",
        "w3-select","w3-eval","p2-if-replay","w2b-cost-replay","w2b-fulltrain-cost-replay","w4","all"])
    args=parser.parse_args()
    if hasattr(os,"sched_getaffinity"):
        aff=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(aff)})
    if args.stage in ("w1-p1","all"):w1_p1()
    if args.stage in ("w1-p2","all"):w1_p2()
    if args.stage in ("p3-upgrade","all"):p3_calref_upgrade()
    if args.stage in ("w2a","all"):w2a_fulltrain()
    if args.stage in ("w2b","all"):w2b_diabetes_logistic()
    if args.stage=="w2b-fulltrain":w2b_diabetes_logistic(fulltrain=True)
    if args.stage in ("w3-select","all"):w3_select()
    if args.stage in ("w3-eval","all"):w3_eval()
    if args.stage in ("p2-if-replay","all"):w1_p2_if_audit_replay()
    if args.stage in ("w2b-cost-replay","all"):w2b_inference_cost_replay()
    if args.stage=="w2b-fulltrain-cost-replay":w2b_inference_cost_replay(fulltrain=True)
    if args.stage in ("w4","all"):w4_assemble()


if __name__=="__main__":main()
