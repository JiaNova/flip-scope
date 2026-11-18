#!/usr/bin/env python3
"""Cache-only fair baselines: Center_EmpVar and Tuned_INT4_Ratio.

This script performs no predictive-model fitting. It reads the frozen Diabetes
D1/D2, U1 and Credit R4 arrays, fits only CAL score maps / low-dimensional CAL
logistic Brier maps, then evaluates against the saved AUDIT decisions.
Outputs are create-only and live in a new versioned directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "reports/core"
OUT = BASE / "STEP1_FAIR_BASELINES_20261113"
from experiments.core import run_c1_c4_v28 as c1
from experiments.core import run_d1_d3 as d1
from experiments.core import run_u1_u3 as u1

BOOT_REPS = 1000
BOOT_SEED = 2026111301
TIMING_REPS = 50
PAIRS = {
    "Center_EmpVar_minus_Conditional_IF": ("Center_EmpVar", "OpCenter_IF"),
    "Tuned_INT4_Ratio_minus_Tuned_INT6_3D": ("Tuned_INT4_Ratio", "Tuned_INT6_3D"),
    "Center_EmpVar_minus_Tuned_INT6_3D": ("Center_EmpVar", "Tuned_INT6_3D"),
    "Tuned_INT4_Ratio_minus_Conditional_IF": ("Tuned_INT4_Ratio", "OpCenter_IF"),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing/unsafe cache: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing/unsafe JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        raise FileExistsError(f"temporary output exists: {tmp}")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status"]
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        raise FileExistsError(f"temporary output exists: {tmp}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def _q_fit(cal_x: np.ndarray, target: np.ndarray, eval_x: np.ndarray, bank: int,
           dataset: str, operation: str) -> tuple[dict, np.ndarray, list[dict]]:
    seed_code = {"Bootstrap": 0, "Op_Half_Replace": 1, "Op_Full_Replace": 2}[operation]
    seed = 2026111301 + (0 if dataset == "Diabetes_Scheme_D" else 100000) + bank * 100 + seed_code
    fit, pred, folds = u1._cv_brier("Tuned_INT4_Ratio", cal_x, target, eval_x, bank, seed)
    fit.update({"dataset": dataset, "operation": operation, "bank": bank,
                "seed": seed, "features": ["z0", "mean_zINT", "log(vtilde_INT)", "mtilde/sqrt(vtilde_INT)"],
                "calibration_target": "mean of six saved CALREF alert-flip decisions",
                "prediction_model_fit_count": 0, "CAL_low_dimensional_fit_count": 51})
    return fit, pred, folds


def _d1_bootstrap_raw(bank: int) -> dict:
    root = d1.OUT / "Scheme_D_Merged_Rare50_Unscaled" / f"bank{bank}"
    pred = read_npz(root / "predictions.npz")
    iv = read_npz(root / "if_arrays.npz")
    maskc = pred["z0c"] >= 0
    maske = pred["z0e"] >= 0
    idx = np.flatnonzero(maske)
    if not np.array_equal(iv["cal_alert_mask"].astype(bool), maskc) or not np.array_equal(iv["eval_alert_mask"].astype(bool), maske):
        raise RuntimeError(f"D1 alert mask mismatch bank={bank}")
    z0c, z0e = pred["z0c"][maskc], pred["z0e"][maske]
    zic, zie = pred["zic"][maskc], pred["zie"][maske]
    svarc = np.var(zic, axis=1, ddof=1)
    sigma2 = float(np.mean(svarc))
    vc = 0.5 * svarc + 0.5 * sigma2
    ve = 0.5 * np.var(zie, axis=1, ddof=1) + 0.5 * sigma2
    zbarc, zbare = zic.mean(axis=1), zie.mean(axis=1)
    mcal, meval = 0.5 * (z0c + zbarc), 0.5 * (z0e + zbare)
    tcal = (pred["zcalref"][maskc] < 0).mean(axis=1)
    maps = read_json(root / "d1_calref_maps.json")
    d2root = d1.OUT / "D2_lambda_boundary_extension" / f"Scheme_D_Merged_Rare50_Unscaled_bank{bank}_fits.json"
    d2fits = read_json(d2root)
    item = next(x for x in d2fits if x["learner"] == "Learned3D_MV_Unconstrained_CV")
    tuned = item["fit"]
    x3e = np.column_stack((z0e, zbare, np.log(np.maximum(ve, 1e-300))))
    q3 = expit(np.clip(tuned["intercept"] + ((x3e - np.asarray(tuned["scaler_mean"])) /
                   np.asarray(tuned["scaler_scale"])) @ np.asarray(tuned["coefficients"]), -40, 40))
    audit = (pred["zaudit"][idx, :11] < 0).astype(np.int8)
    cal_emp = z0c / np.sqrt(np.maximum(vc, 1e-300))
    eval_emp = z0e / np.sqrt(np.maximum(ve, 1e-300))
    cal_ratio = mcal / np.sqrt(np.maximum(vc, 1e-300))
    eval_ratio = meval / np.sqrt(np.maximum(ve, 1e-300))
    return {"dataset": "Diabetes_Scheme_D", "operation": "Bootstrap", "bank": bank,
            "cal_target": tcal, "cal_emp_score": cal_emp, "eval_emp_score": eval_emp,
            "cal_ratio_features": np.column_stack((z0c, zbarc, np.log(np.maximum(vc, 1e-300)), cal_ratio)),
            "eval_ratio_features": np.column_stack((z0e, zbare, np.log(np.maximum(ve, 1e-300)), eval_ratio)),
            "audit": audit, "alert_idx": idx, "n_eval": len(pred["z0e"]), "sigma_g2": sigma2,
            "q_existing": {"Tuned_INT6_3D": q3,
                "OpCenter_IF": c1._apply_saved_map(maps["ConditionalIF"], pred["z0e"][maske] /
                    np.sqrt(np.maximum(iv["v_if_eval_alert"], 1e-300))),
                "Margin": c1._apply_saved_map(maps["Margin"], z0e),
                "qMV": c1._apply_saved_map(maps["qMV"], eval_ratio)},
            "source": str((root / "predictions.npz").relative_to(ROOT)),
            "source_sha256": sha((root / "predictions.npz").read_bytes()),
            "tuned3d_fit_sha256": sha(d2root.read_bytes())}


def _raw_update_or_credit(dataset: str, operation: str, bank: int, credit_identity=None) -> dict:
    if dataset == "Diabetes_Scheme_D":
        path = c1.U1 / f"bank{bank}" / operation / "prediction_arrays.npz"
        arr = read_npz(path)
        tcal = arr["tcal_update"]
        if not np.allclose(tcal, (arr["zCALREF_cal_alert"] < 0).mean(axis=1), rtol=0, atol=0):
            raise RuntimeError(f"saved update CALREF target differs from six CALREF logits: bank={bank}, {operation}")
        audit = arr["audit_eval_alert"].astype(np.int8)
        idx = c1._read_npz(c1.U1 / f"bank{bank}" / "prospective/prospective_scores.npz")["eval_alert_mask"].astype(bool).nonzero()[0]
        n_eval = len(c1._read_npz(c1.U1 / f"bank{bank}" / "prospective/prospective_scores.npz")["eval_alert_mask"])
        z0c, z0e = arr["z0_cal_alert"], arr["z0_eval_alert"]
        zrc, zre = arr["zR_cal_alert"], arr["zR_eval_alert"]
        ziC, ziE = arr["zINT_cal_alert"], arr["zINT_eval_alert"]
        vc, ve = arr["vtilde_cal_alert"], arr["vtilde_eval_alert"]
        maps = None
        qexisting = {"Margin": arr["q_Margin_D0_1D"], "OpCenter_Mean": arr["q_OpCenter_Mean_1D"],
                     "OpCenter_IF": arr["q_OpCenter_IF_1D"], "Tuned_INT6_3D": arr["q_Update_INT6_Tuned3D_MV"],
                     "qMV": arr["q_Update_INT6_qMV"]}
    elif dataset == "UCI_Credit":
        path = c1.CREDIT_ROOT / f"bank{bank}" / operation / "prediction_arrays.npz"
        arr = read_npz(path)
        tcal = arr["tcal_calref"]
        if not np.allclose(tcal, (arr["zCALREF_cal_alert"] < 0).mean(axis=1), rtol=0, atol=0):
            raise RuntimeError(f"saved Credit CALREF target differs from six CALREF logits: bank={bank}, {operation}")
        audit = arr["audit_eval_alert"].astype(np.int8)
        identity = credit_identity if credit_identity is not None else c1._credit_eval_ids_and_alert_indices()
        idx = identity[1][bank]
        n_eval = len(identity[0])
        z0c, z0e = arr["z0_cal_alert"], arr["z0_eval_alert"]
        zrc, zre = arr["zR_cal_alert"], arr["zR_eval_alert"]
        ziC, ziE = arr["zINT_cal_alert"], arr["zINT_eval_alert"]
        vc, ve = arr["vtilde_cal_alert"], arr["vtilde_eval_alert"]
        all_names = [str(x) for x in arr["scored_method_names"].tolist()]
        scored = {n: arr["scored"][:, j] for j, n in enumerate(all_names)}
        qexisting = {"Margin": scored["Margin_D0_1D"], "OpCenter_Mean": scored["OpCenter_Mean_1D"],
                     "OpCenter_IF": scored["OpCenter_IF_1D"], "Tuned_INT6_3D": scored["Update_INT6_Tuned3D_MV"],
                     "qMV": scored["Update_INT6_qMV"]}
        maps = None
    else:
        raise ValueError(dataset)
    if np.any(vc <= 0) or np.any(ve <= 0):
        raise RuntimeError(f"nonpositive saved vtilde: {dataset}/{operation}/bank{bank}")
    if np.any(np.abs(zrc - z0c) > 1e-9) and operation == "Bootstrap":
        raise RuntimeError(f"Bootstrap cache expected zR == z0: {dataset}/bank{bank}")
    zbarc, zbare = ziC.mean(axis=1), ziE.mean(axis=1)
    mcal, meval = 0.5 * (z0c + zbarc), 0.5 * (z0e + zbare)
    cal_emp = zrc / np.sqrt(vc)
    eval_emp = zre / np.sqrt(ve)
    cal_ratio = mcal / np.sqrt(vc)
    eval_ratio = meval / np.sqrt(ve)
    return {"dataset": dataset, "operation": operation, "bank": bank,
            "cal_target": tcal, "cal_emp_score": cal_emp, "eval_emp_score": eval_emp,
            "cal_ratio_features": np.column_stack((z0c, zbarc, np.log(vc), cal_ratio)),
            "eval_ratio_features": np.column_stack((z0e, zbare, np.log(ve), eval_ratio)),
            "audit": audit, "alert_idx": idx, "n_eval": n_eval,
            "sigma_g2": None, "q_existing": qexisting,
            "source": str(path.relative_to(ROOT)), "source_sha256": sha(path.read_bytes()),
            "zR_minus_z0_maxabs": float(np.max(np.abs(zre - z0e)))}


def _metrics(q: np.ndarray, audit: np.ndarray, weights: np.ndarray | None = None) -> dict:
    t = np.asarray(audit, float).mean(axis=1)
    w = np.ones(len(t), float) if weights is None else np.asarray(weights, float)
    out = c1._WeightedRisk(q).metrics(t, w)
    total = float(w.sum())
    out["Brier"] = float(np.sum(w * np.mean((q[:, None] - audit) ** 2, axis=1)) / total)
    out["AUC"] = float(c1.roc_auc_score(audit.ravel(), np.repeat(q, audit.shape[1])))
    return out


def _paired(q_left: np.ndarray, q_right: np.ndarray, audit: np.ndarray, weights: np.ndarray) -> dict:
    t = audit.mean(axis=1)
    ml, mr = c1._WeightedRisk(q_left).metrics(t, weights), c1._WeightedRisk(q_right).metrics(t, weights)
    d = c1._paired_delta(ml, mr, q_left, q_right, t, weights)
    d["Brier"] = float(np.average(np.mean((q_left[:, None]-audit)**2-(q_right[:, None]-audit)**2, axis=1), weights=weights))
    return d


def _run() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError(f"refusing to overwrite versioned output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    raw: dict[tuple[str, str, int], dict] = {}
    fits, fold_rows, point_rows, timing_rows, cache_rows = [], [], [], [], []
    wall0, cpu0 = time.monotonic(), time.process_time()
    # Read only the frozen prediction/variance/CALREF/AUDIT arrays named in the protocol.
    for bank in range(6):
        raw[("Diabetes_Scheme_D", "Bootstrap", bank)] = _d1_bootstrap_raw(bank)
    # Recover Credit row identity once. This loads ID/predictor columns only, no label;
    # saved D0 logits verify alert order. No predictive model is fitted.
    credit_identity = c1._credit_eval_ids_and_alert_indices()
    # U1 and Credit caches are loaded in fixed operation/bank order.
    for dataset in ("Diabetes_Scheme_D", "UCI_Credit"):
        if dataset == "Diabetes_Scheme_D":
            operations = ("Op_Half_Replace", "Op_Full_Replace")
        else:
            operations = ("Bootstrap", "Op_Half_Replace", "Op_Full_Replace")
        for op in operations:
            for bank in range(6):
                raw[(dataset, op, bank)] = _raw_update_or_credit(dataset, op, bank, credit_identity)

    # Fit CAL-only maps/models and persist parameters before metric computation.
    eval_q: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    cal_fit_started = time.process_time()
    for key, case in raw.items():
        dataset, op, bank = key
        map_cpu0 = time.process_time()
        calmap = c1._fit_score_map(case["cal_emp_score"], case["cal_target"])
        map_fit_cpu = time.process_time() - map_cpu0
        q_emp = c1._apply_score_map(calmap, case["eval_emp_score"])
        cv_cpu0 = time.process_time()
        fit4, q4, folds = _q_fit(case["cal_ratio_features"], case["cal_target"],
                                 case["eval_ratio_features"], bank, dataset, op)
        cv_fit_cpu = time.process_time() - cv_cpu0
        qmap = {**case["q_existing"], "Center_EmpVar": q_emp, "Tuned_INT4_Ratio": q4}
        if any(v.shape != case["audit"].shape[:1] or not np.isfinite(v).all() or np.any((v<0)|(v>1)) for v in qmap.values()):
            raise RuntimeError(f"method probability mismatch: {key}")
        eval_q[key] = qmap
        fit_payload = {"dataset": dataset, "operation": op, "bank": bank,
                       "Center_EmpVar_map": calmap, "Tuned_INT4_Ratio_fit": fit4,
                       "Center_EmpVar_map_fit_CPU_seconds": map_fit_cpu,
                       "Tuned_INT4_CV_plus_final_CPU_seconds": cv_fit_cpu,
                       "Center_EmpVar_mapping_fit_count": 1, "Tuned_INT4_CV_plus_final_fit_count": 51,
                       "prediction_model_fit_count": 0,
                       "CAL_target_sha256": sha(np.asarray(case["cal_target"], dtype="<f8").tobytes()),
                       "CAL_emp_score_sha256": sha(np.asarray(case["cal_emp_score"], dtype="<f8").tobytes()),
                       "CAL_ratio_features_sha256": sha(np.asarray(case["cal_ratio_features"], dtype="<f8").tobytes()),
                       "EVAL_predictor_sha256": sha(np.asarray(case["eval_ratio_features"], dtype="<f8").tobytes()),
                       "source": case["source"], "source_sha256": case["source_sha256"]}
        fit_path = OUT / "fits" / dataset / op / f"bank{bank}.json"
        write_json(fit_path, fit_payload)
        for row in folds:
            fold_rows.append({"dataset": dataset, "operation": op, "bank": bank, **row})
        fits.append({"dataset": dataset, "operation": op, "bank": bank,
                     "selected_lambda": float(fit4["lambda"]),
                     "Center_EmpVar_map_slope": float(calmap["slope"]),
                     "Center_EmpVar_map_fit_cpu_seconds": map_fit_cpu,
                     "Tuned_INT4_cv_fit_cpu_seconds": cv_fit_cpu,
                     "prediction_model_fit_count": 0,
                     "CAL_mapping_and_low_dim_fit_count": 52,
                     "source": case["source"], "source_sha256": case["source_sha256"]})
        cache_rows.append({"dataset": dataset, "operation": op, "bank": bank,
                           "alert_n": int(len(case["audit"])), "eval_n": case["n_eval"],
                           "audit_B": int(case["audit"].shape[1]), "sigma_g2": case["sigma_g2"],
                           "zR_minus_z0_maxabs": case.get("zR_minus_z0_maxabs"),
                           "source": case["source"], "source_sha256": case["source_sha256"]})
    lowdim_cpu = time.process_time() - cal_fit_started

    # Evaluation uses only the finite saved AUDIT reference columns, after all CAL fits are persisted.
    for key, case in raw.items():
        dataset, op, bank = key
        for method, q in eval_q[key].items():
            metrics = _metrics(q, case["audit"])
            point_rows.append({"dataset": dataset, "operation": op, "bank": bank,
                               "method": method, **metrics,
                               "n_alerts": len(q), "audit_B": case["audit"].shape[1],
                               "prediction_model_fit_count": 0})
        # Postprocessing-only timing from cached score arrays; excludes feature/model inference.
        for method in ("Center_EmpVar", "Tuned_INT4_Ratio"):
            fn = (lambda: c1._apply_score_map(read_json(OUT/"fits"/dataset/op/f"bank{bank}.json")["Center_EmpVar_map"], case["eval_emp_score"])) if method == "Center_EmpVar" else (
                lambda: u1._predict_brier(read_json(OUT/"fits"/dataset/op/f"bank{bank}.json")["Tuned_INT4_Ratio_fit"], case["eval_ratio_features"]))
            # Load model once outside timed region.
            fitrec = read_json(OUT / "fits" / dataset / op / f"bank{bank}.json")
            if method == "Center_EmpVar":
                mapping = fitrec["Center_EmpVar_map"]
                fn = lambda: c1._apply_score_map(mapping, case["eval_emp_score"])
            else:
                model = fitrec["Tuned_INT4_Ratio_fit"]
                fn = lambda: u1._predict_brier(model, case["eval_ratio_features"])
            fn(); samples=[]
            for _ in range(TIMING_REPS):
                t0=time.process_time(); fn(); samples.append((time.process_time()-t0)*1000.0)
            timing_rows.append({"dataset":dataset,"operation":op,"bank":bank,"method":method,
                "repetitions":TIMING_REPS,"cpu_ms_median":statistics.median(samples),
                "cpu_ms_mean":statistics.mean(samples),"cpu_ms_iqr":float(np.quantile(samples,.75)-np.quantile(samples,.25)),
                "timing_scope":"score postprocessing on cached alert vectors; excludes raw feature preprocessing and base-model inference"})

    # Paired point estimates.
    pair_rows=[]
    cases_by_dataset_op: dict[tuple[str,str],dict[int,dict]]={}
    for (dataset,op,bank),case in raw.items():
        cases_by_dataset_op.setdefault((dataset,op),{})[bank]=case
        qmap=eval_q[(dataset,op,bank)]
        for pair,(left,right) in PAIRS.items():
            delta=_paired(qmap[left],qmap[right],case["audit"],np.ones(len(case["audit"])))
            pair_rows.append({"dataset":dataset,"operation":op,"bank":bank,"pair":pair,
                              "left":left,"right":right,**delta})
    point_summary=[]
    for key in sorted({(r["dataset"],r["operation"],r["pair"]) for r in pair_rows}):
        rows=[r for r in pair_rows if (r["dataset"],r["operation"],r["pair"])==key]
        for metric in ("R_070","AURC100","Brier","ECE10"):
            vals=np.asarray([r[metric] for r in rows],float)
            point_summary.append({"dataset":key[0],"operation":key[1],"pair":key[2],"metric":metric,
                "paired_mean_across_banks":float(vals.mean()),"paired_min":float(vals.min()),"paired_max":float(vals.max()),
                "paired_sd_across_banks":float(vals.std(ddof=1)),"wins_negative":int(np.sum(vals<0)),"banks_n":len(vals),
                "interpretation":"descriptive bank variation; shared EVAL rows, not population CI"})

    # 1,000 shared patient/record row-bootstrap draws; ties are recut at every draw.
    boot_rows=[]; boot_weights=[]; boot_summary=[]
    for d_ix,dataset in enumerate(sorted({k[0] for k in raw})):
        ops=[op for ds,op in cases_by_dataset_op if ds==dataset]
        n_eval=next(c["n_eval"] for (ds,op,b),c in raw.items() if ds==dataset)
        rng=np.random.default_rng(np.random.SeedSequence([BOOT_SEED,d_ix]))
        samples={(op,pair,metric):[] for op in ops for pair in PAIRS for metric in ("R_070","AURC100","Brier","ECE10")}
        for rep in range(BOOT_REPS):
            counts=rng.multinomial(n_eval,np.full(n_eval,1.0/n_eval)).astype(np.uint32)
            digest=sha(counts.astype("<u4").tobytes())
            boot_weights.append({"dataset":dataset,"replicate":rep+1,"seed":BOOT_SEED,"eval_rows":n_eval,
                                 "weight_sha256":digest,"resampled_total":int(counts.sum())})
            for op in ops:
                bank_deltas={pair:{m:[] for m in ("R_070","AURC100","Brier","ECE10")} for pair in PAIRS}
                for bank,case in sorted(cases_by_dataset_op[(dataset,op)].items()):
                    w=counts[case["alert_idx"]].astype(float)
                    if w.sum()<=0: raise RuntimeError(f"zero alert mass in bootstrap {dataset}/{op}/{bank}/{rep}")
                    qmap=eval_q[(dataset,op,bank)]
                    for pair,(left,right) in PAIRS.items():
                        delta=_paired(qmap[left],qmap[right],case["audit"],w)
                        for metric in bank_deltas[pair]: bank_deltas[pair][metric].append(delta[metric])
                for pair in PAIRS:
                    for metric,vals in bank_deltas[pair].items():
                        if len(vals)!=6: raise RuntimeError("bootstrap pair missing banks")
                        avg=float(np.mean(vals));samples[(op,pair,metric)].append(avg)
                        boot_rows.append({"dataset":dataset,"operation":op,"replicate":rep+1,"pair":pair,
                                          "metric":metric,"mean_delta_across_6_banks":avg,"weight_sha256":digest})
        for (op,pair,metric),vals in samples.items():
            a=np.asarray(vals,float)
            boot_summary.append({"dataset":dataset,"operation":op,"pair":pair,"metric":metric,
                "replicates":BOOT_REPS,"seed":BOOT_SEED,"bootstrap_mean":float(a.mean()),
                "row_bootstrap_SE":float(a.std(ddof=1)),"percentile_2_5":float(np.quantile(a,.025)),
                "percentile_97_5":float(np.quantile(a,.975)),
                "resampling_unit":"shared EVAL patient rows across six banks; Credit uses unique record IDs; weighted score ties re-cut each replicate"})

    # AUDIT MC sensitivity: independent column deletion within each bank, stratified jackknife SE.
    jack_rows=[]; jack_summary=[]
    for (dataset,op), banks in sorted(cases_by_dataset_op.items()):
        per_pair={pair:{} for pair in PAIRS}
        for bank,case in sorted(banks.items()):
            B=case["audit"].shape[1]
            if B!=11: raise RuntimeError("expected exactly 11 frozen AUDIT columns")
            for drop in range(B):
                audit=np.delete(case["audit"],drop,axis=1)
                qmap=eval_q[(dataset,op,bank)]
                for pair,(left,right) in PAIRS.items():
                    delta=_paired(qmap[left],qmap[right],audit,np.ones(len(audit)))
                    per_pair[pair].setdefault(bank,[]).append(delta)
                    jack_rows.append({"dataset":dataset,"operation":op,"bank":bank,"dropped_AUDIT_column":drop+1,
                                      "pair":pair,**delta})
        for pair,bank_values in per_pair.items():
            for metric in ("R_070","AURC100","Brier","ECE10"):
                deleted=np.asarray([[row[metric] for row in bank_values[b]] for b in sorted(bank_values)],float)
                # The helper expects deletions of the equal-bank GRAND mean.
                # Centering removes the unchanged-bank offset, but not 1/K.
                se=c1._stratified_jackknife_se(deleted / len(bank_values))
                # Rebuild equal-bank grand deletion values for min/max.
                all_delete=[]
                banks_sorted=sorted(bank_values)
                base_points=[]
                for b in banks_sorted:
                    qmap=eval_q[(dataset,op,b)];case=banks[b]
                    base_points.append(_paired(qmap[PAIRS[pair][0]],qmap[PAIRS[pair][1]],case["audit"],np.ones(len(case["audit"])))[metric])
                grand=float(np.mean(base_points))
                for bi,b in enumerate(banks_sorted):
                    for dd in range(11):
                        all_delete.append(grand+(deleted[bi,dd]-base_points[bi])/len(banks_sorted))
                jack_summary.append({"dataset":dataset,"operation":op,"pair":pair,"metric":metric,
                    "LOAO_min_equal_bank":float(np.min(all_delete)),"LOAO_max_equal_bank":float(np.max(all_delete)),
                    "stratified_AUDIT_MC_SE":se,"banks":len(banks_sorted),"AUDIT_B_per_bank":11,
                    "scope":"within-bank independent AUDIT column deletion, then equal-bank mean; fixed EVAL rows and fitted CAL maps"})

    # Expose existing setup-time accounting for the empirical-variance route: center + six INTs.
    cost_rows=[]
    cost_file=BASE/"paper_evidence/review/Table4_partial_scoring_cost_labeled.csv"
    if cost_file.is_file() and not cost_file.is_symlink():
        with cost_file.open(encoding="utf-8",newline="") as f:
            saved=list(csv.DictReader(f))
        for op in ("Op_Half_Replace","Op_Full_Replace"):
            for cohort in ("all_eval","d0_alert_only"):
                lookup={(r["operation"],r["cohort"],r["route"]):r for r in saved}
                center=lookup[(op,cohort,"OpCenter_Mean")]
                int6=lookup[(op,cohort,"Update_INT6_qMV")]
                cost_rows.append({"operation":op,"cohort":cohort,"route":"Center_EmpVar",
                    "setup_CPU_seconds_center_plus_INT6":float(center["setup_CPU_seconds"])+float(int6["setup_CPU_seconds"]),
                    "center_setup_CPU_seconds":float(center["setup_CPU_seconds"]),
                    "INT6_setup_CPU_seconds_in_qMV_reference":float(int6["setup_CPU_seconds"]),
                    "online_CPU_ms_per_input_row_center_plus_qMV_reference":float(center["online_CPU_ms_per_input_row"])+float(int6["online_CPU_ms_per_input_row"]),
                    "scope":"derived from measured partial-path components; includes operator-center fit and six-INT setup; D0/CALREF/CV/preprocessing excluded"})
                cost_rows.append({"operation":op,"cohort":cohort,"route":"Tuned_INT4_Ratio",
                    "setup_CPU_seconds_INT6_plus_new_CAL_CV":None,
                    "INT6_setup_CPU_seconds_reference":float(int6["setup_CPU_seconds"]),
                    "CV_map_fit_CPU_seconds_measured_all_banks":None,
                    "online_CPU_ms_per_input_row":"not fully measured; model inference path equals D0 + six INT before 4-feature logistic head",
                    "scope":"not conflated with cached-vector postprocessing timings"})

    # Persist tables after the complete calculation succeeds.
    write_csv(OUT/"point_metrics_by_bank.csv",point_rows)
    write_csv(OUT/"paired_point_by_bank.csv",pair_rows)
    write_csv(OUT/"paired_point_summary.csv",point_summary)
    write_csv(OUT/"row_bootstrap_replicates.csv",boot_rows)
    write_csv(OUT/"row_bootstrap_weight_digests.csv",boot_weights)
    write_csv(OUT/"row_bootstrap_summary.csv",boot_summary)
    write_csv(OUT/"audit_loao_replicates.csv",jack_rows)
    write_csv(OUT/"audit_loao_summary.csv",jack_summary)
    write_csv(OUT/"cv_fold_losses.csv",fold_rows)
    write_csv(OUT/"fit_summary.csv",fits)
    write_csv(OUT/"cache_inventory.csv",cache_rows)
    write_csv(OUT/"timing_postprocessing.csv",timing_rows)
    write_csv(OUT/"cost_accounting.csv",cost_rows)
    elapsed_wall=time.monotonic()-wall0;elapsed_cpu=time.process_time()-cpu0
    audit={"status":"COMPLETED_CACHE_ONLY_BASELINES","task":"Step 1 / operation A",
        "output_root":str(OUT.relative_to(ROOT)),"predictive_model_fit_count":0,
        "CAL_score_map_fits":len(raw),"Tuned_INT4_CV_plus_final_CAL_fits":len(raw)*51,
        "CV_fold_count":len(fold_rows),"row_bootstrap_replicates_per_dataset_operation":BOOT_REPS,
        "row_bootstrap_seed":BOOT_SEED,"AUDIT_LOAO_columns_per_bank":11,
        "timing_repetitions":TIMING_REPS,"preprocessing_or_raw_feature_reads":0,
        "Diabetes_cache_units":18,"Credit_cache_units":18,"total_cache_units":len(raw),
        "low_dimensional_fit_CPU_seconds":lowdim_cpu,"total_CPU_seconds":elapsed_cpu,
        "total_wall_seconds":elapsed_wall,"peak_RSS_platform_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_RSS_units":"KiB on Linux",
        "scope_limits":["postprocessing timing uses saved score vectors; excludes matrix model inference",
            "Center_EmpVar update setup adds operator-center and six-INT setup costs from the locked partial cost table",
            "Tuned_INT4 model-inference timing is not measured here; full path requires D0 and six INT inference",
            "bootstrap intervals condition on saved models, CAL fits, EVAL covariates, and finite AUDIT references"]}
    write_json(OUT/"execution_audit.json",audit)
    report=["# Step 1 fair-baseline cache replay", "",
        "This work package adds `Center_EmpVar` and `Tuned_INT4_Ratio` using the saved D1/D2, U1 and Credit R4 arrays. It fits no predictive models.",
        "",f"- Predictive-model fits: **0**.",f"- New CAL one-dimensional mappings: **{len(raw)}**.",
        f"- Tuned_INT4 CV and final CAL fits: **{len(raw)*51}** (5 folds × 10 penalties + final refit per cache unit).",
        f"- Shared row-bootstrap: **{BOOT_REPS}** repetitions per dataset-operation; seed `{BOOT_SEED}`.",
        f"- Elapsed CPU/wall: **{elapsed_cpu:.2f}s / {elapsed_wall:.2f}s**.",
        "- The empirical-variance score uses the same operator-center numerator as Conditional IF and a separately fitted non-increasing 1D Brier map.",
        "- Tuned_INT4 uses CAL-fold-local standardization and the extended lambda grid; EVAL AUDIT columns enter only after all fit records have been written.",
        "- Cached-vector timing excludes input feature transforms and base-model inference; it is not an end-to-end serving latency.",
        "", "See `paired_point_summary.csv`, `row_bootstrap_summary.csv`, `audit_loao_summary.csv`, and `cost_accounting.csv` for results.",""]
    (OUT/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    _run()
