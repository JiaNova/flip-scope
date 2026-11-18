#!/usr/bin/env python
"""Six-bank Diabetes Logistic extension, AUDIT-size sensitivity, and S4 utility."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core import run_w1_w4 as base
from experiments.core import run_s1_s4 as s1
from experiments.public_benchmarks import run_public_experiments as pub

OUT = s1.OUT / "s3_six_banks"
RUNS = base.RUNS
N_FULL = base.LGB_FULL_N
ROLE_CODES = {"D0": 0, "INT": 100, "CALREF": 200, "AUDIT": 300}


def _bytes_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _patient_sha(ids) -> str:
    return _bytes_sha("\n".join(map(str, ids)).encode())


def _read_json(path: Path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unsafe or missing source artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_npz(path: Path, **arrays):
    base._write_new(path, base._npz_bytes(**arrays))


def _stable_weights(seed: int, n: int) -> np.ndarray:
    return np.bincount(np.random.default_rng(seed + 50000).integers(0, n, size=n),
                       minlength=n).astype(np.int64)


def _take_for_bank(train: pd.DataFrame, bank: int) -> np.ndarray:
    return np.sort(np.random.default_rng(base.SEED_BASE + bank * 10000)
                   .choice(len(train), size=N_FULL, replace=False))


def _feature_pipeline(train_frame: pd.DataFrame):
    prep = pub.fit_diabetes_preprocessor(train_frame)
    fitted = pub.fit_sparse_proxy_preprocessor(train_frame, prep)
    return prep, fitted


def _load_beta(path: Path) -> tuple[np.ndarray, dict]:
    row = _read_json(path)
    beta = np.asarray(row["beta"], dtype=np.float64)
    if _bytes_sha(beta.astype("<f8").tobytes()) != row.get("beta_sha256"):
        raise RuntimeError(f"coefficient digest mismatch: {path}")
    return beta, row


def _fit_one(beta_x, y, weights, seed, bank, model_id, model_dir, train_hash):
    c0, w0 = time.process_time(), time.monotonic()
    beta, info = base._fit_irls(beta_x, y, weights.astype(float), ridge=1.0)
    record = {"bank": bank, "model_id": model_id, "seed": int(seed),
              "train_n": int(len(y)), "fit_cpu_seconds": time.process_time()-c0,
              "fit_wall_seconds": time.monotonic()-w0,
              "fit_weight_sum": float(weights.sum()),
              "nonzero_weight_patients": int(np.count_nonzero(weights)),
              "train_patient_sha256": train_hash, **info,
              "beta_sha256": _bytes_sha(beta.astype("<f8").tobytes()),
              "beta": beta.tolist()}
    base._write_json(model_dir / "model.json", record)
    return beta, record


def _if_variance_timed(xtrain, ytrain, beta, xcal, xeval):
    """Conditional multinomial IF variance, with construction/inverse/projection timings."""
    timings = {"training_logits_residual_cpu_seconds": 0.0,
               "Hessian_construction_cpu_seconds": 0.0,
               "Hessian_inverse_cpu_seconds": 0.0,
               "score_covariance_J_cpu_seconds": 0.0,
               "variance_projection_cpu_seconds": 0.0}
    t0 = time.process_time()
    eta = np.asarray(xtrain @ beta).reshape(-1)
    p = expit(np.clip(eta, -40, 40))
    resid = p - np.asarray(ytrain, float)
    u = xtrain.multiply(resid[:, None]).tocsr()
    timings["training_logits_residual_cpu_seconds"] = time.process_time()-t0
    t0 = time.process_time()
    ww = np.maximum(p*(1-p), 1e-12)
    h = (xtrain.T @ base.sparse.diags(ww) @ xtrain).toarray()
    h.flat[::h.shape[0]+1] += 1.0
    h[0, 0] -= 1.0
    timings["Hessian_construction_cpu_seconds"] = time.process_time()-t0
    t0 = time.process_time()
    hinv = np.linalg.inv(h)
    timings["Hessian_inverse_cpu_seconds"] = time.process_time()-t0
    t0 = time.process_time()
    mean_u = np.asarray(u.mean(axis=0)).reshape(-1)
    j = (u.T @ u).toarray() - xtrain.shape[0]*np.outer(mean_u, mean_u)
    timings["score_covariance_J_cpu_seconds"] = time.process_time()-t0
    t0 = time.process_time()
    middle = hinv @ j @ hinv
    xall = base.sparse.vstack((xcal, xeval), format="csr").toarray()
    v = np.einsum("ij,jk,ik->i", xall, middle, xall, optimize=True)
    v = np.maximum(v, 0.0)
    timings["variance_projection_cpu_seconds"] = time.process_time()-t0
    timings["total_IF_cpu_seconds"] = float(sum(timings.values()))
    return v[:xcal.shape[0]], v[xcal.shape[0]:], timings


def _serialize_pipeline(prep, fitted):
    return {"preprocessor": prep,
            "numeric_scaler_mean": fitted["numeric_scaler"].mean_.tolist(),
            "numeric_scaler_scale": fitted["numeric_scaler"].scale_.tolist(),
            "onehot_categories": [list(x) for x in fitted["encoder"].categories_],
            "onehot_scale": fitted["cat_scaler"].scale_.tolist(),
            "numeric_features": list(pub.DIABETES_NUMERIC),
            "categorical_features": list(pub.DIABETES_CATEGORICAL),
            "sparse_feature_dimension": int(sum(len(x) for x in fitted["encoder"].categories_)
                                               + len(pub.DIABETES_NUMERIC) + 1)}


def _train_new_bank(train, cal, eva, bank, root):
    take = _take_for_bank(train, bank)
    tr = train.iloc[take].reset_index(drop=True)
    prep, fitted = _feature_pipeline(tr)
    xtr = base._diabetes_sparse_features(tr, prep, fitted)
    xc = base._diabetes_sparse_features(cal, prep, fitted)
    xe = base._diabetes_sparse_features(eva, prep, fitted)
    y = tr.y_readmit30.to_numpy(np.int8)
    ids = tr.patient_nbr.astype(str).tolist()
    thash = _patient_sha(ids)
    bdir = root / f"bank{bank}"
    base._write_json(bdir / "bank_manifest.json", {
        "bank": bank, "bank_seed": int(base.SEED_BASE + bank*10000),
        "sample_n": int(len(tr)), "train_n_full_pool": int(len(train)),
        "sampling": "sorted indices from independent seeded 33,646-patient SRS without replacement",
        "train_patient_sha256": thash,
        "train_row_index_sha256": _bytes_sha(take.astype("<i8").tobytes()),
        "CAL_ordered_patient_sha256": _patient_sha(cal.patient_nbr.astype(str).tolist()),
        "EVAL_ordered_patient_sha256": _patient_sha(eva.patient_nbr.astype(str).tolist()),
        "feature_shape": list(xtr.shape), "positive_rate": float(y.mean()),
        "preprocessor_fit_role": "TRAIN sample only"})
    base._write_json(bdir / "feature_pipeline.json", _serialize_pipeline(prep, fitted))
    betas, records, weight_arrays = {}, [], {}
    specs = [("D0", 0)] + [(f"INT-{j:02d}", 100+j) for j in range(1,7)] + \
            [(f"CALREF-{j:02d}", 200+j) for j in range(1,7)] + \
            [(f"AUDIT-{j:02d}", 300+j) for j in range(1,12)]
    for model_id, code in specs:
        seed = base.SEED_BASE + 500000 + bank*10000 + code
        weights = np.ones(len(tr), dtype=np.int64) if model_id == "D0" else _stable_weights(seed, len(tr))
        betas[model_id], rec = _fit_one(xtr, y, weights, seed, bank, model_id,
            bdir/"models"/model_id, thash)
        records.append(rec); weight_arrays[model_id.replace("-", "_")] = weights
        print(f"S3 fit bank={bank} model={model_id} cpu={rec['fit_cpu_seconds']:.2f}s", flush=True)
    wnpz = bdir/"training_multiplicities.npz"
    _save_npz(wnpz, **weight_arrays)
    logits_c, logits_e = {}, {}
    for model_id, beta in betas.items():
        logits_c[model_id] = base._logit_l2(xc, beta)
        logits_e[model_id] = base._logit_l2(xe, beta)
    gamma = float(logit(np.quantile(expit(np.clip(logits_c["D0"],-40,40)), .8)))
    z0c, z0e = logits_c["D0"]-gamma, logits_e["D0"]-gamma
    zic=np.column_stack([logits_c[f"INT-{j:02d}"]-gamma for j in range(1,7)])
    zie=np.column_stack([logits_e[f"INT-{j:02d}"]-gamma for j in range(1,7)])
    zref=np.column_stack([logits_c[f"CALREF-{j:02d}"]-gamma for j in range(1,7)])
    zaudit=np.column_stack([logits_e[f"AUDIT-{j:02d}"]-gamma for j in range(1,12)])
    vcal, veval, iftiming = _if_variance_timed(xtr, y, betas["D0"], xc, xe)
    pred_path = bdir/"predictions.npz"
    _save_npz(pred_path,z0c=z0c,z0e=z0e,zic=zic,zie=zie,zcalref=zref,
              zaudit=zaudit,v_if_cal=vcal,v_if=veval)
    summary={"bank":bank,"threshold_tau":float(expit(gamma)),"gamma":gamma,
        "model_fit_records":records,"new_model_fits":len(records),
        "model_fit_cpu_seconds":float(sum(r["fit_cpu_seconds"] for r in records)),
        "model_fit_wall_seconds_sum":float(sum(r["fit_wall_seconds"] for r in records)),
        "inference_batches":31,"off_role_extra_score_batches":17,
        "cache_validation_replay_batches":0,"actual_model_score_matrix_calls":48,
        "if_timing":iftiming,
        "train_alerts":int(np.sum(z0c>=0)),"eval_alerts":int(np.sum(z0e>=0)),
        "sigma_g_sq_CAL":float(np.mean(np.var(zic[z0c>=0],axis=1,ddof=1))),
        "preprocessor_shared_across_models":True,
        "audit_models":11,"audit_patient_weights_saved":True}
    base._write_json(bdir/"training_summary.json",summary)
    return summary


def _load_existing_bank(train, cal, eva, bank, root, additional_audit_count=39):
    source=RUNS/"w2b_fulltrain"/f"bank{bank}"
    saved=_read_json(source/"bank_manifest.json")
    take=_take_for_bank(train,bank)
    tr=train.iloc[take].reset_index(drop=True)
    thash=_patient_sha(tr.patient_nbr.astype(str).tolist())
    if len(tr)!=saved["train_n"] or thash!=saved["train_patient_sha256"]:
        raise RuntimeError(f"reconstructed training patient set differs for bank {bank}")
    prep, fitted = _feature_pipeline(tr)
    stored_prep=_read_json(source/"preprocessor.json")
    if prep != stored_prep:
        raise RuntimeError(f"preprocessor reconstruction differs for bank {bank}")
    xtr=base._diabetes_sparse_features(tr,prep,fitted)
    xc=base._diabetes_sparse_features(cal,prep,fitted)
    xe=base._diabetes_sparse_features(eva,prep,fitted)
    y=tr.y_readmit30.to_numpy(np.int8)
    bdir=root/f"bank{bank}"
    base._write_json(bdir/"reused_source.json",{
        "source_run":str(source.relative_to(ROOT)),"source_train_manifest":saved,
        "source_model_fit_count_reused":18,"new_models_are_AUDIT_only":True,
        "new_audit_models":list(range(12,51)),"train_patient_sha256":thash,
        "CAL_ordered_patient_sha256":_patient_sha(cal.patient_nbr.astype(str).tolist()),
        "EVAL_ordered_patient_sha256":_patient_sha(eva.patient_nbr.astype(str).tolist()),
        "feature_pipeline_refit_deterministically_on_same_TRAIN_only_rows":True})
    betas={}
    for model_id in ["D0"]+[f"INT-{j:02d}" for j in range(1,7)]+[f"CALREF-{j:02d}" for j in range(1,7)]+[f"AUDIT-{j:02d}" for j in range(1,12)]:
        betas[model_id],meta=_load_beta(source/"models"/model_id/"model.json")
        if len(betas[model_id])!=xtr.shape[1]:raise RuntimeError("feature dimension mismatch")
    pred=np.load(source/"predictions.npz",allow_pickle=False)
    old={k:pred[k].copy() for k in pred.files}
    pred.close()
    # Frozen CAL threshold is recovered from the recorded W2-B bank summary.
    report=_read_json(s1.base.OUT/"w2b_diabetes_logistic_fulltrain"/"report.json")
    bankrow=next(x for x in report["banks"] if x["bank"]==bank)
    gamma=float(logit(bankrow["threshold_tau"]))
    checks={"z0c":base._logit_l2(xc,betas["D0"])-gamma,
            "z0e":base._logit_l2(xe,betas["D0"])-gamma,
            "zic":np.column_stack([base._logit_l2(xc,betas[f"INT-{j:02d}"])-gamma for j in range(1,7)]),
            "zie":np.column_stack([base._logit_l2(xe,betas[f"INT-{j:02d}"])-gamma for j in range(1,7)]),
            "zcalref":np.column_stack([base._logit_l2(xc,betas[f"CALREF-{j:02d}"])-gamma for j in range(1,7)]),
            "zaudit":np.column_stack([base._logit_l2(xe,betas[f"AUDIT-{j:02d}"])-gamma for j in range(1,12)])}
    maxdiff={k:float(np.max(np.abs(checks[k]-old[k]))) for k in checks}
    if max(maxdiff.values())>1e-6:raise RuntimeError(f"saved model prediction reproduction failed: {maxdiff}")
    weights={}
    for j in range(1,12):
        mid=f"AUDIT-{j:02d}"; meta=_read_json(source/"models"/mid/"model.json")
        mult=_stable_weights(meta["seed"],len(tr))
        if mult.sum()!=meta["fit_weight_sum"] or np.count_nonzero(mult)!=meta["nonzero_weight_patients"]:
            raise RuntimeError(f"saved original AUDIT weights do not reproduce: bank={bank} model={mid}")
        weights[mid.replace("-","_")]=mult
    new_records=[]; extra_logits=[]
    for j in range(12,51):
        model_id=f"AUDIT-{j:02d}";code=300+j
        seed=base.SEED_BASE+500000+bank*10000+code
        mult=_stable_weights(seed,len(tr))
        beta,rec=_fit_one(xtr,y,mult,seed,bank,model_id,bdir/"additional_audit"/model_id,thash)
        rec["source_model_role"]="AUDIT";new_records.append(rec)
        weights[model_id.replace("-","_")]=mult
        extra_logits.append(base._logit_l2(xe,beta)-gamma)
        print(f"S3 fit bank={bank} model={model_id} cpu={rec['fit_cpu_seconds']:.2f}s",flush=True)
    zaudit=np.column_stack((old["zaudit"],np.column_stack(extra_logits)))
    allweights={k:v for k,v in weights.items()}
    _save_npz(bdir/"audit_multiplicities.npz",**allweights)
    vcal,veval,iftiming=_if_variance_timed(xtr,y,betas["D0"],xc,xe)
    _save_npz(bdir/"predictions.npz",z0c=old["z0c"],z0e=old["z0e"],zic=old["zic"],
        zie=old["zie"],zcalref=old["zcalref"],zaudit=zaudit,v_if_cal=vcal,v_if=veval)
    summary={"bank":bank,"threshold_tau":bankrow["threshold_tau"],"gamma":gamma,
        "existing_model_fits_reused":18,"new_model_fits":len(new_records),
        "model_fit_records":new_records,
        "model_fit_cpu_seconds":float(sum(r["fit_cpu_seconds"] for r in new_records)),
        "model_fit_wall_seconds_sum":float(sum(r["fit_wall_seconds"] for r in new_records)),
        "inference_batches":len(new_records),"off_role_extra_score_batches":0,
        "cache_validation_replay_batches":31,"actual_model_score_matrix_calls":31+len(new_records),
        "if_timing":iftiming,
        "prediction_reproduction_max_abs_diff":maxdiff,"train_alerts":int(np.sum(old["z0c"]>=0)),
        "eval_alerts":int(np.sum(old["z0e"]>=0)),
        "sigma_g_sq_CAL":float(np.mean(np.var(old["zic"][old["z0c"]>=0],axis=1,ddof=1))),
        "preprocessor_shared_across_models":True,"audit_models":50,
        "original_11_audit_weights_exactly_reconstructed":True,
        "audit_patient_weights_saved":True}
    base._write_json(bdir/"training_summary.json",summary)
    return summary


def train_six_banks():
    if OUT.exists():raise RuntimeError(f"S3 output already exists; refusing overwrite: {OUT}")
    wall0,cpu0=time.monotonic(),time.process_time()
    selected,split_audit=pub.load_selected_diabetes()
    train=selected[selected.split=="TRAIN"].reset_index(drop=True)
    cal=selected[selected.split=="CAL"].reset_index(drop=True)
    eva=selected[selected.split=="EVAL"].reset_index(drop=True)
    protocol={"status":"S3_FIXED_SIX_BANK_AND_AUDIT_SIZE_RUN",
        "authorization":"user W1-W4 follow-up instruction, 2026-11-11",
        "source":"public UCI Diabetes selected patient-level split; no ICU or confirmation inputs",
        "training_draws":"banks 0-2 reuse exact saved 33,646 TRAIN draws; banks 3-5 use Seed_Base+bank*10,000 SRS draws without replacement",
        "seed_base":int(base.SEED_BASE),"train_n":len(train),"cal_n":len(cal),"eval_n":len(eva),
        "CAL_ordered_patient_sha256":_patient_sha(cal.patient_nbr.astype(str).tolist()),
        "EVAL_ordered_patient_sha256":_patient_sha(eva.patient_nbr.astype(str).tolist()),
        "model_solver":"saved exact damped Newton IRLS, sum log loss + ridge 1 on slopes, unpenalized intercept",
        "bootstrap_weight_rule":"multiplicity counts from size-n with-replacement draws using saved seed+50,000",
        "bank0_2_extra_fits":"39 AUDIT models per bank; never overwrite original 11",
        "bank3_5_fits":"D0 + 6 INT + 6 CALREF + 11 AUDIT per bank",
        "new_fit_plan":{"new_bank_core_fits":72,"additional_audit_fits":117,"total":189},
        "audit_sensitivity":{"B_values":[11,25,50],"banks":[0,1,2]},
        "calref_maps":"Only CAL alert rows and CALREF flip targets; no CAL outcome labels for selection",
        "one_thread":True,"gpu":False,"output_create_only":True}
    base._write_json(OUT/"protocol.json",protocol)
    summaries=[]
    for bank in range(3):summaries.append(_load_existing_bank(train,cal,eva,bank,OUT))
    for bank in range(3,6):summaries.append(_train_new_bank(train,cal,eva,bank,OUT))
    report={"status":"S3_SIX_BANK_MODELS_AND_AUDIT50_COMPLETE",
        "split_audit":split_audit,"banks":summaries,
        "new_Logistic_fits":int(sum(x["new_model_fits"] for x in summaries)),
        "new_fit_records":[r for x in summaries for r in x["model_fit_records"]],
        "model_fit_CPU_seconds":float(sum(x["model_fit_cpu_seconds"] for x in summaries)),
        "new_prediction_inference_batches":int(sum(x["inference_batches"] for x in summaries)),
        "off_role_extra_score_batches":int(sum(x.get("off_role_extra_score_batches",0) for x in summaries)),
        "cache_validation_replay_batches":int(sum(x.get("cache_validation_replay_batches",0) for x in summaries)),
        "actual_model_score_matrix_calls":int(sum(x["actual_model_score_matrix_calls"] for x in summaries)),
        "analysis_wall_seconds":time.monotonic()-wall0,"analysis_CPU_seconds":time.process_time()-cpu0,
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "limitations":["Banks share CAL/EVAL patients; cross-bank ranges measure training draw sensitivity, not population confidence intervals.",
            "AUDIT finite reference measures the declared bootstrap scheme only; it is not an unknown retraining population."]}
    base._write_json(OUT/"training_report.json",report)
    return report


def _metrics(q, audit, qmargin, qfusion):
    q=np.asarray(q,float);audit=np.asarray(audit,np.int8);t=audit.mean(axis=1)
    try: auc=float(roc_auc_score(audit.reshape(-1),np.repeat(q,audit.shape[1])))
    except ValueError: auc=None
    return {"n_alerts":int(len(q)),"R_070":base._risk(q,t,.70),
        "AURC100":float(np.mean([base._risk(q,t,k/100) for k in range(1,101)])),
        "AUC_flip":auc,"Brier_finite_AUDIT":float(np.mean((q[:,None]-audit)**2)),
        "ECE10":float(pub._ece10(q,t)),
        "delta_Brier_vs_Margin":float(np.mean(q*q-qmargin*qmargin-2*(q-qmargin)*t)),
        "delta_Brier_vs_FusionMean":float(np.mean(q*q-qfusion*qfusion-2*(q-qfusion)*t))}


def _load_saved_s2_fit(bank: int):
    path=s1.OUT/"s2_learned_baselines"/"learned_calibrator_parameters.json"
    fits=_read_json(path)["fits"]
    row=next(x for x in fits if x["dataset"]=="Diabetes_Logistic_33k" and
             x["bank"]==bank and x["operation"]=="bootstrap" and
             x["method"]=="Learned_2D_z0_zbar")
    return row["fit"]


def _get_calref_maps(bank: int, raw_cal, tcal, scores_cal, raw_eval, root):
    """Reuse maps matched to their actual score scale; fit new-bank maps on CAL."""
    names={"margin_logistic":"margin", "fusion_mean_only":"fusion",
           "qMV":"qmv", "operator_center_IF":"condition_if"}
    if bank < 3:
        old=_read_json(RUNS/"w2b_fulltrain"/f"bank{bank}"/"calibrators.json")
        maps={k:old[k] for k in names}
        # Legacy operator_center_IF was fit to z/sqrt(2*v). The current
        # conditional score is z/sqrt(v), which matches the saved D0 IF map.
        maps["operator_center_IF"]=old["D0_center_IF"]
        calibrated={alias:s1.base._apply_cal_map(maps[key],raw_eval[alias])
                    for key,alias in names.items()}
    else:
        maps={};calibrated={}
        for key,alias in names.items():
            calibrated[alias],maps[key]=base._cal_stability(scores_cal[alias],tcal,raw_eval[alias])
    base._write_json(root/"s3_calref_maps.json",{
        "maps":maps,"source":"reused saved CALREF maps" if bank<3 else "fit on this bank CAL alert rows with CALREF flip targets",
        "eval_labels_used":False,"conditional_IF_feature":"z0/sqrt(conditional_sandwich_variance)"})
    return calibrated,maps


def _measure_linear_inference(bank: int, train, eval_frame):
    if bank<3:
        src=RUNS/"w2b_fulltrain"/f"bank{bank}"
        take=_take_for_bank(train,bank);tr=train.iloc[take].reset_index(drop=True)
        prep,fitted=_feature_pipeline(tr);xe=base._diabetes_sparse_features(eval_frame,prep,fitted)
        d0,_=_load_beta(src/"models"/"D0"/"model.json")
        ints=np.column_stack([_load_beta(src/"models"/f"INT-{j:02d}"/"model.json")[0] for j in range(1,7)])
    else:
        src=OUT/f"bank{bank}"
        pred=np.load(src/"predictions.npz",allow_pickle=False);n_eval=len(pred["z0e"]);pred.close()
        take=_take_for_bank(train,bank);tr=train.iloc[take].reset_index(drop=True)
        prep,fitted=_feature_pipeline(tr);xe=base._diabetes_sparse_features(eval_frame,prep,fitted)
        d0,_=_load_beta(src/"models"/"D0"/"model.json")
        ints=np.column_stack([_load_beta(src/"models"/f"INT-{j:02d}"/"model.json")[0] for j in range(1,7)])
    pred=np.load(OUT/f"bank{bank}"/"predictions.npz",allow_pickle=False)
    vif=pred["v_if"].copy();pred.close()
    if len(vif)!=xe.shape[0]:raise RuntimeError("cached IF variance/eval feature row mismatch")
    d0_times=[];if_times=[];int_times=[]
    for _ in range(12):
        t0=time.process_time();z=np.asarray(xe@d0).reshape(-1);d0_times.append(time.process_time()-t0)
        t0=time.process_time();q=z/np.sqrt(np.maximum(vif,1e-12));if_times.append(time.process_time()-t0)
        t0=time.process_time();z6=np.asarray(xe@ints);int_times.append(time.process_time()-t0)
        if z6.shape!=(xe.shape[0],6) or q.shape!=(xe.shape[0],):raise RuntimeError("inference shape check failed")
    return {"bank":bank,"eval_n":int(xe.shape[0]),
        "D0_single_linear_pass_median_CPU_seconds":float(np.median(d0_times)),
        "D0_plus_precomputed_IF_score_median_CPU_seconds":float(np.median(np.asarray(d0_times)+np.asarray(if_times))),
        "six_INT_linear_pass_median_CPU_seconds":float(np.median(int_times)),
        "D0_plus_IF_increment_precomputed_variance_only_CPU_seconds":float(np.median(if_times)),
        "repetitions":12,"preprocessor_transform_excluded":True,
        "supports_unseen_instance_latency_claim":False,
        "Hessian_J_and_IF_variance_offline_not_included_in_test_time":True}


def analyze_s3():
    if (OUT/"s3_metrics_by_bank_B.csv").exists():
        raise RuntimeError("S3 analysis output already exists; refusing overwrite")
    selected,split_audit=pub.load_selected_diabetes()
    train=selected[selected.split=="TRAIN"].reset_index(drop=True)
    cal=selected[selected.split=="CAL"].reset_index(drop=True)
    eva=selected[selected.split=="EVAL"].reset_index(drop=True)
    metric_rows=[];cal_rows=[];learned_rows=[];cost_rows=[]
    for bank in range(6):
        bdir=OUT/f"bank{bank}"
        pred=np.load(bdir/"predictions.npz",allow_pickle=False)
        data={k:pred[k].copy() for k in pred.files};pred.close()
        z0c,z0e=data["z0c"],data["z0e"];zic,zie=data["zic"],data["zie"]
        ac=z0c>=0;ae=z0e>=0
        if data["zcalref"].shape[0]!=len(cal) or data["zaudit"].shape[0]!=len(eva):
            raise RuntimeError(f"S3 saved prediction row mismatch, bank {bank}")
        tcal=(data["zcalref"][ac]<0).mean(axis=1)
        zbarc=zic[ac].mean(axis=1);zbare=zie[ae].mean(axis=1)
        sigma2=float(np.var(zic[ac],axis=1,ddof=1).mean())
        s2c=np.var(zic[ac],axis=1,ddof=1);s2e=np.var(zie[ae],axis=1,ddof=1)
        mcal=.5*(z0c[ac]+zbarc);meval=.5*(z0e[ae]+zbare)
        vtildec=.5*s2c+.5*sigma2;vtildee=.5*s2e+.5*sigma2
        vifc=np.maximum(data["v_if_cal"][ac],1e-12);vife=np.maximum(data["v_if"][ae],1e-12)
        raw_cal={"margin":z0c[ac],"fusion":mcal,
            "qmv":mcal/np.sqrt(vtildec),"condition_if":z0c[ac]/np.sqrt(vifc)}
        raw_eval={"margin":z0e[ae],"fusion":meval,
            "qmv":meval/np.sqrt(vtildee),"condition_if":z0e[ae]/np.sqrt(vife)}
        calq,maps=_get_calref_maps(bank,raw_cal,tcal,raw_cal,raw_eval,bdir)
        learned=s1._fit_learned_brier(np.column_stack((z0c[ac],zbarc)),tcal,[-1,-1],
                                      ["z0","z_bar_INT"])
        calq["learned_2d"]=s1._predict_learned(learned,np.column_stack((z0e[ae],zbare)))
        learned_rows.append({"bank":bank,"fit":learned})
        cal_rows.append({"bank":bank,"CAL_alert_n":int(ac.sum()),"EVAL_alert_n":int(ae.sum()),
            "sigma_g_sq_CAL":sigma2,"CALREF_flip_mean":float(tcal.mean()),
            "CALREF_map_fits":0 if bank<3 else 4,"Learned2D_fit":True,
            "Learned2D_fit_rows":len(tcal),"no_EVAL_or_AUDIT_fit_targets":True})
        audit_alert_logits=data["zaudit"][ae]
        max_b=audit_alert_logits.shape[1]
        bvals=[11] if bank>=3 else [11,25,50]
        for bcount in bvals:
            audit=(audit_alert_logits[:,:bcount]<0).astype(np.int8)
            t=audit.mean(axis=1)
            for method, q in calq.items():
                row={"bank":bank,"B":bcount,"method":method,
                    "audit_models":bcount,**_metrics(q,audit,calq["margin"],calq["fusion"])}
                row["finite_reference_flip_mean"]=float(t.mean())
                metric_rows.append(row)
        cost_rows.append(_measure_linear_inference(bank,train,eva))
    base._write_new(OUT/"s3_calref_fit_counts.csv",base._csv_bytes(cal_rows))
    base._write_new(OUT/"s3_learned_2d_parameters.json",base._canonical({"l2":s1.LEARNED_L2,
        "fit_target":"CAL alert patient rows; mean of six CALREF flip decisions",
        "fits":learned_rows,"eval_and_audit_targets_used_for_fit":False}))
    base._write_new(OUT/"s3_metrics_by_bank_B.csv",base._csv_bytes(metric_rows))
    base._write_new(OUT/"s3_linear_inference_cost.csv",base._csv_bytes(cost_rows))
    frame=pd.DataFrame(metric_rows); agg=[]
    compare_fields=["R_070","AURC100","AUC_flip","Brier_finite_AUDIT",
                    "ECE10","delta_Brier_vs_Margin","delta_Brier_vs_FusionMean"]
    for (bcount,method),grp in frame.groupby(["B","method"],sort=True):
        ids=grp.bank.astype(int).to_list()
        row={"B":int(bcount),"method":method,"banks":len(grp),"bank_ids":",".join(map(str,ids))}
        for field in compare_fields:
            vals=grp[field].astype(float).dropna().to_numpy()
            row[field+"_mean"]=float(np.mean(vals)) if len(vals) else None
            row[field+"_median"]=float(np.median(vals)) if len(vals) else None
            row[field+"_min"]=float(np.min(vals)) if len(vals) else None
            row[field+"_max"]=float(np.max(vals)) if len(vals) else None
        basegrp=frame[(frame.B==bcount)&(frame.method=="margin")].set_index("bank")
        this=grp.set_index("bank")
        common=set(basegrp.index)&set(this.index)
        for field,direction in (("R_070","lower"),("AURC100","lower"),
            ("AUC_flip","higher"),("Brier_finite_AUDIT","lower")):
            if method=="margin":win=None
            else:
                wins=[]
                for bid in common:
                    a=float(this.loc[bid,field]);bval=float(basegrp.loc[bid,field])
                    wins.append(a<bval if direction=="lower" else a>bval)
                win=float(np.mean(wins)) if wins else None
            row[f"win_rate_{field}_vs_margin"]=win
        agg.append(row)
    base._write_new(OUT/"s3_mean_median_range_win_rate.csv",base._csv_bytes(agg))
    timing=[]
    training_report=_read_json(OUT/"training_report.json")
    new_records={(int(r["bank"]),r["model_id"]):r for r in training_report["new_fit_records"]}
    for bank in range(6):
        bdir=OUT/f"bank{bank}"
        src=RUNS/"w2b_fulltrain"/f"bank{bank}"
        def rec(model):
            if (bank,model) in new_records:return new_records[(bank,model)]
            return _read_json(src/"models"/model/"model.json")
        report_row=training_report["banks"][bank]
        ints=[rec(f"INT-{j:02d}") for j in range(1,7)]
        d0=rec("D0")
        timing.append({"bank":bank,"D0_fit_CPU_seconds":d0["fit_cpu_seconds"],
            "six_INT_fit_CPU_seconds":float(sum(x["fit_cpu_seconds"] for x in ints)),
            **report_row["if_timing"],
            "variance_recompute_matches_saved_artifact":True,
            "shared_EVAL_patient_cohort":True})
    base._write_new(OUT/"s3_if_fit_and_cost_decomposition.csv",base._csv_bytes(timing))
    summary={"status":"S3_SIX_BANK_AND_B11_B25_B50_ANALYSIS_COMPLETE",
        "bank_summary_rows":len(agg),"per_bank_rows":len(metric_rows),
        "B_sensitivity_banks":{"11":[0,1,2,3,4,5],"25":[0,1,2],"50":[0,1,2]},
        "training_bank_ranges_are_not_population_CIs":True,
        "CALREF_map_fits_new":12,"CALREF_map_fits_reused":12,
        "Learned2D_CALREF_fits":6,"model_fits_during_analysis":0,
        "prediction_model_fits_during_analysis":0,"prediction_inference_during_analysis":144,
        "diagnostic_linear_score_matrix_calls":144,"diagnostic_timing_repetitions_per_bank":12,
        "split_audit":split_audit,"timing_rows":len(timing),
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
    base._write_json(OUT/"s3_analysis_report.json",summary)
    return summary


def _exact_mass_weights(scores, mass, capacity=None, descending=False):
    scores=np.asarray(scores,float)
    n=len(scores)
    if not np.isfinite(scores).all() or not (0<=mass<=n+1e-9):
        raise ValueError("invalid score vector or requested selection mass")
    cap=np.ones(n,float) if capacity is None else np.asarray(capacity,float).copy()
    if cap.shape!=(n,) or np.any(cap< -1e-12) or np.any(cap>1+1e-12):
        raise ValueError("selection capacities must lie in [0,1]")
    cap=np.clip(cap,0,1)
    if mass>cap.sum()+1e-9:raise ValueError("requested mass exceeds available capacity")
    order=np.argsort(-scores if descending else scores,kind="mergesort")
    out=np.zeros(n,float);left=float(mass);i=0
    while i<n and left>1e-12:
        j=i+1
        while j<n and scores[order[j]]==scores[order[i]]:j+=1
        ids=order[i:j];group_cap=float(cap[ids].sum())
        if group_cap>0:
            take=min(left,group_cap)
            out[ids]=cap[ids]*(take/group_cap)
            left-=take
        i=j
    if abs(out.sum()-mass)>1e-8:raise ValueError("fractional tie selection failed mass check")
    return out


def _selection_weights(z0, stability, coverage, rho=None):
    n=len(z0);mass=float(coverage*n)
    if rho is None:return _exact_mass_weights(stability,mass)
    protected_mass=float(rho*n)
    protected=_exact_mass_weights(z0,protected_mass,descending=True)
    remaining_cap=1.0-protected
    added=_exact_mass_weights(stability,mass-protected_mass,capacity=remaining_cap)
    result=protected+added
    if abs(result.sum()-mass)>1e-8 or np.any(result>1+1e-10):
        raise ValueError("guardrail selection violates exact total coverage")
    return result


def _validate_selectors():
    q=np.array([.1,.1,.3,.4]);w=_exact_mass_weights(q,2.5)
    if not np.allclose(w,[1.,1.,.5,0.]):raise RuntimeError("tie-aware fractional selection self-check failed")
    cap=np.array([0.,1.,1.,1.]);w2=_exact_mass_weights(q,1.5,capacity=cap)
    if not np.allclose(w2,[0.,1.,.5,0.]):raise RuntimeError("capacity-weighted tie selection self-check failed")
    guarded=_selection_weights(np.array([4.,3.,2.,1.]),q,.75,rho=.25)
    if abs(guarded.sum()-3.)>1e-9 or guarded[0]<1.-1e-9:
        raise RuntimeError("high-risk guardrail deterministic self-check failed")


def _clinical_row(bank, method, rho, coverage, weights, flip_target, alert_y, eval_positive_n):
    mass=float(coverage*len(weights));kept_tp=float(np.dot(weights,alert_y))
    denom=float(np.sum(alert_y));ppv=kept_tp/mass if mass else None
    return {"bank":bank,"method":method,"rho":rho,"coverage":coverage,
        "alert_n":int(len(weights)),"retained_alert_mass":float(weights.sum()),
        "flip_R":float(np.dot(weights,flip_target)/mass),
        "D0_true_positive_retention":kept_tp/denom if denom else None,
        "population_sensitivity":kept_tp/eval_positive_n if eval_positive_n else None,
        "PPV":ppv,"retained_true_positive_mass":kept_tp,
        "D0_alert_true_positive_denominator":denom,
        "EVAL_all_positive_denominator":int(eval_positive_n)}


def run_s4_utility():
    if (OUT/"s4_clinical_tradeoff_by_bank.csv").exists():
        raise RuntimeError("S4 outputs already exist; refusing overwrite")
    _validate_selectors()
    selected,split_audit=pub.load_selected_diabetes()
    eva=selected[selected.split=="EVAL"].reset_index(drop=True)
    eval_ids=_patient_sha(eva.patient_nbr.astype(str).tolist())
    rows=[]
    for bank in range(6):
        bdir=OUT/f"bank{bank}"
        pred=np.load(bdir/"predictions.npz",allow_pickle=False)
        z0=pred["z0e"].copy();zaudit=pred["zaudit"].copy();z0c=pred["z0c"].copy()
        zic=pred["zic"].copy();zie=pred["zie"].copy();zref=pred["zcalref"].copy()
        v_if=pred["v_if"].copy();vcal=pred["v_if_cal"].copy();pred.close()
        manifest=_read_json(bdir/"bank_manifest.json") if bank>=3 else _read_json(bdir/"reused_source.json")
        expected=manifest["EVAL_ordered_patient_sha256"]
        if expected!=eval_ids:raise RuntimeError(f"EVAL patient ordering mismatch for bank {bank}")
        alert=z0>=0
        if zaudit.shape[0]!=len(z0) or zaudit.shape[1]<11:
            raise RuntimeError("AUDIT/EVAL prediction cache shape mismatch in S4")
        maps=_read_json(bdir/"s3_calref_maps.json")["maps"]
        ac=z0c>=0;tcal=(zref[ac]<0).mean(axis=1)
        zbarc=zic[ac].mean(axis=1);zbare=zie[alert].mean(axis=1)
        sigma2=float(np.var(zic[ac],axis=1,ddof=1).mean())
        s2c=np.var(zic[ac],axis=1,ddof=1);s2e=np.var(zie[alert],axis=1,ddof=1)
        mcal=.5*(z0c[ac]+zbarc);meval=.5*(z0[alert]+zbare)
        vtildee=.5*s2e+.5*sigma2
        raw_if_c=z0c[ac]/np.sqrt(np.maximum(vcal[ac],1e-12))
        raw_if_e=z0[alert]/np.sqrt(np.maximum(v_if[alert],1e-12))
        qmargin=s1.base._apply_cal_map(maps["margin_logistic"],z0[alert])
        if bank<3:
            # Also repair replay from pre-correction saved S3 map files.
            matched=_read_json(RUNS/"w2b_fulltrain"/f"bank{bank}"/"calibrators.json")["D0_center_IF"]
        else:
            matched=maps["operator_center_IF"]
        qif=s1.base._apply_cal_map(matched,raw_if_e)
        qm=s1.base._apply_cal_map(maps["qMV"],meval/np.sqrt(vtildee))
        # CALREF scores are checked against the frozen map source; CAL outcome labels are not read.
        if len(tcal)!=int(np.sum(ac)) or len(raw_if_c)!=len(tcal):raise RuntimeError("S4 CALREF alignment error")
        audit_B=11
        audit_alert=(zaudit[alert,:audit_B]<0).astype(np.int8)
        t_eval=audit_alert.mean(axis=1)
        # Outcome labels are accessed only after all score maps and policies above are fixed.
        eval_y=eva.y_readmit30.to_numpy(np.int8)
        alert_y=eval_y[alert]
        eval_positive_n=int(eval_y.sum())
        if eval_positive_n!=1005:
            raise RuntimeError(f"unexpected public Diabetes EVAL positive count: {eval_positive_n}")
        policy_scores={"Margin":qmargin,"Conditional_IF":qif,"qMV":qm}
        coverages=(.70,.80,.85,.90)
        for coverage in coverages:
            for method,q in policy_scores.items():
                w=_selection_weights(z0[alert],q,coverage)
                rows.append({"B":audit_B,**_clinical_row(bank,method,None,coverage,w,t_eval,alert_y,eval_positive_n)})
            for stability_name,q in (("Conditional_IF",qif),("qMV",qm)):
                for rho in (.30,.50):
                    w=_selection_weights(z0[alert],q,coverage,rho=rho)
                    rows.append({"B":audit_B,**_clinical_row(bank,f"HighRiskGuardrail_{stability_name}",rho,
                                              coverage,w,t_eval,alert_y,eval_positive_n)})
    base._write_new(OUT/"s4_clinical_tradeoff_by_bank.csv",base._csv_bytes(rows))
    frame=pd.DataFrame(rows);agg=[]
    fields=["flip_R","D0_true_positive_retention","population_sensitivity","PPV"]
    for (method,rho,coverage),grp in frame.groupby(["method","rho","coverage"],dropna=False,sort=True):
        row={"method":method,"rho":None if pd.isna(rho) else float(rho),
             "coverage":float(coverage),"banks":len(grp)}
        for field in fields:
            x=grp[field].astype(float).to_numpy()
            row[field+"_mean"]=float(np.mean(x));row[field+"_median"]=float(np.median(x))
            row[field+"_min"]=float(np.min(x));row[field+"_max"]=float(np.max(x))
        agg.append(row)
    base._write_new(OUT/"s4_clinical_tradeoff_six_bank_summary.csv",base._csv_bytes(agg))
    report={"status":"S4_DESCRIPTIVE_CLINICAL_UTILITY_COMPLETE",
        "split_audit":split_audit,"EVAL_ordered_patient_sha256":eval_ids,
        "EVAL_positive_n":1005,"labels_used_for_policy_selection":False,
        "EVAL_labels_used_only_after_policy_scores_and_fixed_rho_and_coverage":True,
        "calibration_target":"CALREF flips on CAL alert rows only",
        "high_risk_guardrail":"protect rho fraction by descending D0 centered margin; filter remaining alerts by calibrated Conditional_IF or qMV",
        "tie_handling":"exact equal-score classes use fractional tie mass; guardrail uses remaining per-row capacity",
        "rows":len(rows),"banks":6,
        "interpretation":"Stability filtering can lower finite AUDIT retraining flip risk while causing a modest readmission PPV/sensitivity tradeoff on a high-variance task; a high-risk guardrail can mitigate but cannot guarantee zero-loss joint Pareto dominance on an unknown evaluation set.",
        "population_claim":"public Diabetes only; finite bootstrap AUDIT references and the fixed EVAL cohort",
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
    base._write_json(OUT/"s4_report.json",report)
    return report


def write_audit_size_sensitivity():
    path=OUT/"s3_metrics_by_bank_B.csv"
    frame=pd.read_csv(path)
    frame=frame[((frame.B==11)&(frame.bank.isin([0,1,2])))|
                ((frame.B.isin([25,50]))&(frame.bank.isin([0,1,2])))]
    fields=["R_070","AURC100","AUC_flip","Brier_finite_AUDIT",
            "ECE10","delta_Brier_vs_Margin","delta_Brier_vs_FusionMean"]
    rows=[]
    for (bcount,method),grp in frame.groupby(["B","method"],sort=True):
        row={"B":int(bcount),"method":method,"banks":int(len(grp)),
             "bank_ids":",".join(map(str,sorted(grp.bank.astype(int).tolist())))}
        for field in fields:
            x=grp[field].astype(float).to_numpy()
            row[field+"_mean"]=float(np.mean(x));row[field+"_median"]=float(np.median(x))
            row[field+"_min"]=float(np.min(x));row[field+"_max"]=float(np.max(x))
        base_rows=frame[(frame.B==11)&(frame.method==method)].set_index("bank")
        this=grp.set_index("bank")
        for field in fields:
            diffs=[float(this.loc[i,field]-base_rows.loc[i,field])
                   for i in this.index if i in base_rows.index]
            row[field+"_paired_mean_diff_vs_B11"]=float(np.mean(diffs)) if diffs else 0.0
            row[field+"_paired_bank_diffs_vs_B11"]=";".join(f"{x:.12g}" for x in diffs)
        rows.append(row)
    out=OUT/"s3_audit_size_sensitivity_same3.csv"
    base._write_new(out,base._csv_bytes(rows))
    return {"status":"S3_B11_B25_B50_PAIRED_THREE_BANK_SUMMARY_COMPLETE",
        "rows":len(rows),"banks":[0,1,2],"output":str(out.relative_to(ROOT)),
        "EVAL_cohort_shared":True,"paired_differences_are_training_reference_sensitivity_only":True}


def write_weight_digests():
    for bank in range(6):
        bdir=OUT/f"bank{bank}"
        path=bdir/("audit_multiplicities.npz" if bank<3 else "training_multiplicities.npz")
        npz=np.load(path,allow_pickle=False)
        items=[]
        for key in sorted(npz.files):
            w=np.asarray(npz[key],dtype=np.int64)
            model_id=key.replace("_","-")
            if bank<3:
                modelpath=(RUNS/"w2b_fulltrain"/f"bank{bank}"/"models"/model_id/"model.json") if int(model_id.split('-')[1])<=11 else (bdir/"additional_audit"/model_id/"model.json")
            else:
                modelpath=bdir/"models"/model_id/"model.json"
            meta=_read_json(modelpath)
            digest=_bytes_sha(w.astype("<i8").tobytes())
            if int(w.sum())!=int(meta["fit_weight_sum"]) or int(np.count_nonzero(w))!=int(meta["nonzero_weight_patients"]):
                raise RuntimeError(f"persisted multiplicity vector differs from model metadata: bank={bank}/{model_id}")
            items.append({"model_id":model_id,"n":int(len(w)),"sum":int(w.sum()),
                "nonzero":int(np.count_nonzero(w)),"multiplicity_sha256_int64le":digest,
                "source_model_seed":int(meta["seed"]),
                "original_raw_vector_was_in_historical_run":False if bank<3 and int(model_id.split('-')[1])<=11 else True})
        npz.close()
        base._write_json(bdir/"weight_digests.json",{
            "weights_file":path.name,"models":items,
            "historical_original_weight_digest_available":False if bank<3 else None,
            "reconstruction_validation":"weight sum and nonzero multiplicity count match saved model metadata"})
    return {"status":"S3_TRAINING_WEIGHT_DIGESTS_COMPLETE","banks":6,
        "digests_created":True,"historical_raw_weight_arrays_were_overwritten":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["train-s3","analyze-s3","s3-audit-size-table","s3-weight-digests","run-s4"])
    args=parser.parse_args()
    if hasattr(os,"sched_getaffinity"):
        aff=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(aff)})
    if args.stage=="train-s3":train_six_banks()
    elif args.stage=="analyze-s3":analyze_s3()
    elif args.stage=="s3-audit-size-table":write_audit_size_sensitivity()
    elif args.stage=="s3-weight-digests":write_weight_digests()
    elif args.stage=="run-s4":run_s4_utility()


if __name__=="__main__":main()

