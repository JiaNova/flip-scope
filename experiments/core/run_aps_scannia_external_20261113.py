#!/usr/bin/env python3
"""Fixed-protocol APS Scania external benchmark (Logistic + LightGBM).

The protocol and official source files are create-only inputs. EVAL outcomes are
not accessed until all method scores are fit and materialized. This is a new
public-data experiment; it does not read project clinical or confirmation data.
"""
from __future__ import annotations
import csv, hashlib, json, os, resource, signal, sys, time
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core import run_w1_w4 as base
from experiments.core import run_u1_u3 as u1
from experiments.core import run_c1_c4_v28 as c1

WORK = ROOT / "reports/core/STEP3_APS_20261113"
RAW = ROOT / "data/public/aps"
OUT = WORK / "run_v1_1"
PROTOCOL = WORK / "protocol_v1_1.json"
TRAIN_PATH = RAW / "aps_failure_training_set.csv"
TEST_PATH = RAW / "aps_failure_test_set.csv"
THREADS = 1
N_BANKS = 6
N_MODEL = 16000
OPS = ("Bootstrap", "Half_Replace", "Full_Replace")
LAMBDA_GRID = (0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.)
ROLE_SPECS = [("INT", i) for i in range(1, 7)] + [("CALREF", i) for i in range(1, 7)] + [("AUDIT", i) for i in range(1, 12)]
METHODS_LOGIT = ("Margin", "OpCenter_Mean", "Conditional_IF", "Center_EmpVar", "Fused_Center_Only", "Variance_Only_qV", "qMV", "Tuned_INT6_3D", "Tuned_INT4_Ratio")
METHODS_LGB = ("Margin", "OpCenter_Mean", "Center_EmpVar", "Fused_Center_Only", "Variance_Only_qV", "qMV", "Tuned_INT6_3D", "Tuned_INT4_Ratio")


class BudgetTimeout(TimeoutError):
    pass


def _timeout(_signum, _frame):
    raise BudgetTimeout("APS run reached fixed 21600-second wall-clock limit")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unsafe/missing input: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed(*parts) -> int:
    raw = ("APS_SCANNIA_EXTERNAL_V1_1_20261113|" + "|".join(map(str, parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32 - 1)


def atomic_bytes(path: Path, raw: bytes, *, replace: bool = False) -> None:
    if path.is_symlink() or (path.exists() and not replace):
        raise FileExistsError(f"refuse overwrite or symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if tmp.exists() or tmp.is_symlink():
        raise FileExistsError(tmp)
    with tmp.open("xb") as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, obj, *, replace: bool = False) -> None:
    atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(), replace=replace)


def write_csv(path: Path, rows, *, replace: bool = False) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else []
    from io import StringIO
    s = StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=fields); w.writeheader(); w.writerows(rows)
    atomic_bytes(path, s.getvalue().encode(), replace=replace)


def write_npz(path: Path, *, replace: bool = False, **arrays) -> None:
    from io import BytesIO
    b = BytesIO(); np.savez_compressed(b, **arrays)
    atomic_bytes(path, b.getvalue(), replace=replace)


def update_state(path: Path, obj) -> None:
    """Atomic status checkpoint inside this newly-created run directory."""
    write_json(path, obj, replace=path.exists())


def append_jsonl(path: Path, obj) -> None:
    if path.is_symlink(): raise RuntimeError(f"refuse symlink fit ledger: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())


def parse_data():
    """Parse official CSVs and make exact hash-ranked, fixed-size partitions."""
    lines_train = TRAIN_PATH.read_bytes().splitlines()
    lines_test = TEST_PATH.read_bytes().splitlines()
    if len(lines_train) < 21 or len(lines_test) < 21:
        raise RuntimeError("APS source is shorter than the 20-line copyright preamble and header")
    htrain = next(csv.reader([lines_train[20].decode("utf-8-sig")]))
    htest = next(csv.reader([lines_test[20].decode("utf-8-sig")]))
    if htrain != htest or htrain[0] != "class" or len(htrain) != 171:
        raise RuntimeError(f"APS schema mismatch: train={len(htrain)} test={len(htest)} columns")
    feat = htrain[1:]
    train_lines = [x.rstrip(b"\r\n") for x in lines_train[21:]]
    test_lines = [x.rstrip(b"\r\n") for x in lines_test[21:]]
    if len(train_lines) != 60000 or len(test_lines) != 16000:
        raise RuntimeError(f"APS row count mismatch: train={len(train_lines)} test={len(test_lines)}")
    train_predictor_keys=[]
    for line in train_lines:
        row=next(csv.reader([line.decode("utf-8-sig")]))
        if len(row)!=171:raise RuntimeError("raw APS row width mismatch")
        train_predictor_keys.append(",".join(row[1:]).encode("utf-8"))
    train = pd.read_csv(TRAIN_PATH, skiprows=20, na_values=["na"], low_memory=False)
    # Load EVAL predictors only. The target column is parsed after all predictions are frozen.
    test = pd.read_csv(TEST_PATH, skiprows=20, usecols=feat, na_values=["na"], low_memory=False)
    if list(train.columns) != htrain or set(test.columns) != set(feat) or len(test) != 16000:
        raise RuntimeError("pandas parser column order differs from raw header")
    if not set(train["class"].astype(str).unique()) <= {"neg", "pos"}:
        raise RuntimeError("unexpected APS training target labels")
    for col in feat:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
    if train[feat].notna().to_numpy().sum() + train[feat].isna().to_numpy().sum() != 60000 * 170:
        raise RuntimeError("APS training numeric coercion lost cells")

    train_order = sorted(range(60000), key=lambda i: (hashlib.sha256(train_predictor_keys[i]).digest(), i))
    tr_idx = np.asarray(train_order[:48000], np.int64); ca_idx = np.asarray(train_order[48000:], np.int64)
    pool_order = sorted(tr_idx.tolist(), key=lambda i: (hashlib.sha256(b"APS_POOL_A_B_V1_1:" + train_predictor_keys[i]).digest(), i))
    a_idx = np.asarray(pool_order[:24000], np.int64); b_idx = np.asarray(pool_order[24000:], np.int64)
    if len(set(tr_idx) & set(ca_idx)) or len(set(a_idx) & set(b_idx)) or len(a_idx) != 24000 or len(b_idx) != 24000:
        raise RuntimeError("hash partition overlap/cardinality check failed")
    # Check exact duplicate full records crossing roles; no customer ID exists.
    role_by_row = np.full(60000, "", dtype="U5"); role_by_row[tr_idx] = "TRAIN"; role_by_row[ca_idx] = "CAL"
    duplicate_groups = {}
    for i, key in enumerate(train_predictor_keys):
        duplicate_groups.setdefault(sha(key), []).append(i)
    duplicate_cross = sum(1 for ix in duplicate_groups.values() if len({role_by_row[i] for i in ix}) > 1)
    missing = train[feat].isna().sum()
    audit = {
        "status": "PREFLIGHT_VALIDATED_NO_MODEL_FIT",
        "protocol_id": "APS_SCANNIA_EXTERNAL_V1_1_20261113",
        "train_rows": len(train), "test_rows": len(test), "predictor_columns": len(feat),
        "feature_names": feat,
        "header_sha256": sha(",".join(htrain).encode()),
        "official_train_label_counts": {str(k): int(v) for k, v in train["class"].value_counts().sort_index().items()},
        "official_test_label_counts_materialized_for_final_metric_only": False,
        "TRAIN_n": int(len(tr_idx)), "CAL_n": int(len(ca_idx)), "PoolA_n": int(len(a_idx)), "PoolB_n": int(len(b_idx)),
        "TRAIN_index_sha256": sha(tr_idx.astype("<i8").tobytes()), "CAL_index_sha256": sha(ca_idx.astype("<i8").tobytes()),
        "PoolA_index_sha256": sha(a_idx.astype("<i8").tobytes()), "PoolB_index_sha256": sha(b_idx.astype("<i8").tobytes()),
        "official_test_index_sha256": sha(np.arange(16000, dtype="<i8").tobytes()),
        "missing_values_by_feature": {str(k): int(v) for k, v in missing.items() if int(v) > 0},
        "missing_cell_count": int(missing.sum()),
        "identical_predictor_groups_crossing_TRAIN_CAL": int(duplicate_cross),
        "partition_key": "canonical predictor values only; outcome column excluded; official row index breaks duplicate-key ties",
        "row_identity_limitation": "the public files have no customer/truck ID; row identity is the official source row index",
        "train_csv_sha256": file_sha(TRAIN_PATH), "test_csv_sha256": file_sha(TEST_PATH),
    }
    return train, test, feat, tr_idx, ca_idx, a_idx, b_idx, audit


def fit_preprocessor(train_a: pd.DataFrame, feat: list[str]):
    n = len(train_a); raw = train_a[feat].to_numpy(float)
    cont_cols=[]; lows=[]; highs=[]; medians=[]; means=[]; scales=[]; dropped=[]
    missing_cols=[]; missing_support={}
    for j, name in enumerate(feat):
        col = raw[:, j]; observed = col[np.isfinite(col)]
        if not len(observed):
            dropped.append({"feature": name, "reason": "all_missing"}); continue
        lo, hi = np.quantile(observed, [0.005, 0.995])
        clipped = np.clip(col, lo, hi); med = float(np.nanmedian(clipped)); filled = np.where(np.isfinite(clipped), clipped, med)
        mu = float(np.mean(filled)); sd = float(np.std(filled, ddof=0))
        support = int(np.count_nonzero(~np.isfinite(col))); missing_support[name] = support
        if sd <= 1e-12 or not np.isfinite(sd):
            dropped.append({"feature": name, "reason": "constant_after_winsorization"})
        else:
            cont_cols.append(name); lows.append(float(lo)); highs.append(float(hi)); medians.append(med); means.append(mu); scales.append(sd)
        # A sparse positive or negative side under 50 is treated as a rare indicator and dropped.
        if support > 0 and min(support, n-support) >= 50:
            missing_cols.append(name)
    if not cont_cols:
        raise RuntimeError("preprocessing removed all APS continuous features")
    return {"raw_feature_columns": list(feat), "continuous_columns": cont_cols, "winsor_low": lows, "winsor_high": highs,
            "medians": medians, "means": means, "scales": scales,
            "missing_indicator_columns": missing_cols, "missing_support": missing_support,
            "dropped_continuous": dropped,
            "rare_missing_indicators_dropped": [k for k,v in missing_support.items() if 0 < v < 50 or n-50 < v < n],
            "fit_n": n, "fit_pool": "Pool_A"}


def transform(frame: pd.DataFrame, prep):
    cols = prep["continuous_columns"]
    raw_all = frame[prep["raw_feature_columns"]].to_numpy(float)
    raw = frame[cols].to_numpy(float)
    lows=np.asarray(prep["winsor_low"]); highs=np.asarray(prep["winsor_high"])
    med=np.asarray(prep["medians"]); means=np.asarray(prep["means"]); scales=np.asarray(prep["scales"])
    missing=~np.isfinite(raw); vals=np.minimum(np.maximum(raw,lows),highs); vals=np.where(missing,med,vals)
    cont=(vals-means)/scales
    pieces=[cont]
    if prep["missing_indicator_columns"]:
        ix=[prep["raw_feature_columns"].index(c) for c in prep["missing_indicator_columns"]]
        pieces.append((~np.isfinite(raw_all[:,ix])).astype(float))
    out=np.column_stack(pieces).astype(np.float64,copy=False)
    if not np.isfinite(out).all(): raise RuntimeError("nonfinite APS transformed feature")
    return out


def with_intercept(x: np.ndarray):
    return sparse.hstack([sparse.csr_matrix(np.ones((len(x),1))), sparse.csr_matrix(x)], format="csr")


def fit_logit(x, y, weights, beta_path: Path):
    beta, info = base._fit_irls(with_intercept(x), np.asarray(y,np.int8), np.asarray(weights,float), ridge=1.)
    info = {**info, "beta_sha256": sha(beta.astype("<f8").tobytes())}
    write_json(beta_path, {"fit_family":"regularized_logistic_IRLS","ridge":1.,"fit_info":info,
        "beta_sha256":info["beta_sha256"],"beta":beta.tolist()})
    return beta, info


def lgb_params(seed_value: int):
    return {"objective":"binary","metric":"binary_logloss","learning_rate":.05,"num_leaves":15,"max_depth":5,
        "min_data_in_leaf":40,"lambda_l2":1.,"feature_fraction":1.,"bagging_fraction":1.,"bagging_freq":0,
        "deterministic":True,"force_col_wise":True,"num_threads":1,"device_type":"cpu","verbosity":-1,
        "seed":int(seed_value),"data_random_seed":int(seed_value),"feature_fraction_seed":int(seed_value),
        "bagging_seed":int(seed_value),"drop_seed":int(seed_value)}


def fit_lgb(x, y, weights, seed_value: int, model_path: Path):
    ds=lgb.Dataset(np.asarray(x), label=np.asarray(y,np.int8), weight=np.asarray(weights,float), free_raw_data=True)
    model=lgb.train(lgb_params(seed_value), ds, num_boost_round=100)
    raw=model.model_to_string().encode()
    atomic_bytes(model_path,raw)
    return model, {"model_sha256":sha(raw),"seed":int(seed_value),"num_boost_round":100,"lightgbm_version":lgb.__version__}


def margins(model, family: str, x):
    if family == "Logistic":
        return np.asarray(with_intercept(np.asarray(x)) @ model).reshape(-1)
    return np.asarray(model.predict(np.asarray(x), raw_score=True, num_threads=1),float).reshape(-1)


def _fit_predict_model(family, x, y, weights, seed_value, path):
    t0=time.process_time();w0=time.monotonic()
    if family == "Logistic":
        model, info=fit_logit(x,y,weights,path.with_suffix(".json"))
    else:
        model, info=fit_lgb(x,y,weights,seed_value,path.with_suffix(".txt"))
    info={**info,"fit_cpu_seconds":time.process_time()-t0,"fit_wall_seconds":time.monotonic()-w0}
    return model,info


def _metrics(q, audit):
    q=np.asarray(q,float); audit=np.asarray(audit,np.int8); t=audit.mean(axis=1)
    return {"n_alerts":int(len(q)),"R_070":base._risk(q,t,.70),
        "AURC100":float(np.mean([base._risk(q,t,j/100.) for j in range(1,101)])),
        "AUC_flip":float(roc_auc_score(audit.ravel(),np.repeat(q,audit.shape[1]))),
        "Brier":float(np.mean((q[:,None]-audit)**2)),"ECE10":c1._ece10(q,t),"mean_flip_rate":float(t.mean())}


def _prediction_metrics(q, y):
    q=np.asarray(q,float); y=np.asarray(y,np.int8)
    return {"EVAL_n":int(len(y)),"EVAL_positive_n":int(y.sum()),"AUROC":float(roc_auc_score(y,q)),
        "AUPRC":float(average_precision_score(y,q)),"Brier":float(np.mean((q-y)**2))}


def _metrics_for_bank(bank: int, family: str, op: str, scores_cal, scores_eval, audit, tcal, z0c, z0e,
                      zRc, zRe, zIC, zIE, vIFc, vIFe, fitdir: Path, cvseed: int):
    zbarc=zIC.mean(axis=1); zbare=zIE.mean(axis=1)
    s2c=np.var(zIC,axis=1,ddof=1); s2e=np.var(zIE,axis=1,ddof=1)
    sigma2=float(s2c.mean()); vtc=.5*s2c+.5*sigma2; vte=.5*s2e+.5*sigma2
    mc=.5*(z0c+zbarc); me=.5*(z0e+zbare)
    rawcal={"Margin":z0c,"OpCenter_Mean":zRc,"Center_EmpVar":zRc/np.sqrt(vtc),
        "Fused_Center_Only":mc,"Variance_Only_qV":z0c/np.sqrt(vtc),"qMV":mc/np.sqrt(vtc)}
    rawev={"Margin":z0e,"OpCenter_Mean":zRe,"Center_EmpVar":zRe/np.sqrt(vte),
        "Fused_Center_Only":me,"Variance_Only_qV":z0e/np.sqrt(vte),"qMV":me/np.sqrt(vte)}
    if vIFc is not None:
        rawcal["Conditional_IF"]=zRc/np.sqrt(np.maximum(vIFc,1e-12));rawev["Conditional_IF"]=zRe/np.sqrt(np.maximum(vIFe,1e-12))
    out={};maps={}
    for name, sc in rawcal.items():
        mp=c1._fit_score_map(sc,tcal);maps[name]=mp;out[name]=c1._apply_score_map(mp,rawev[name])
    x3c=np.column_stack((z0c,zbarc,np.log(np.maximum(vtc,1e-300))))
    x3e=np.column_stack((z0e,zbare,np.log(np.maximum(vte,1e-300))))
    fit3,q3,fold3=u1._cv_brier("Tuned_INT6_3D",x3c,tcal,x3e,bank,cvseed);out["Tuned_INT6_3D"]=q3
    x4c=np.column_stack((z0c,zbarc,np.log(np.maximum(vtc,1e-300)),mc/np.sqrt(vtc)))
    x4e=np.column_stack((z0e,zbare,np.log(np.maximum(vte,1e-300)),me/np.sqrt(vte)))
    fit4,q4,fold4=u1._cv_brier("Tuned_INT4_Ratio",x4c,tcal,x4e,bank,cvseed+1);out["Tuned_INT4_Ratio"]=q4
    cvrows=[{"bank":bank,"family":family,"operation":op,**r} for r in fold3+fold4]
    write_json(fitdir/"calibration_and_cv.json",{"1d_maps":maps,"Tuned_INT6_3D":fit3,"Tuned_INT4_Ratio":fit4,
        "CALREF_target_sha256":sha(tcal.astype("<f8").tobytes()),"lambda_grid":list(LAMBDA_GRID),"eval_used_for_selection":False})
    methodrows=[]
    for name,q in out.items():
        methodrows.append({"bank":bank,"family":family,"operation":op,"method":name,**_metrics(q,audit),
            "sigma_g2_cal":sigma2,"cv_lambda":fit4["lambda"] if name=="Tuned_INT4_Ratio" else fit3["lambda"] if name=="Tuned_INT6_3D" else None})
    diag={"bank":bank,"family":family,"operation":op,"eval_alert_n":len(audit),"sigma_g2_cal":sigma2,
        "mean_audit_flip_rate":float(audit.mean()),"mean_INT_variance_eval":float(s2e.mean()),
        "INT_variance_calibration_sample_n":6,"finite_reference_AUDIT_n":11}
    if vIFe is not None:
        avar=np.var(scores_eval["AUDIT"],axis=1,ddof=1); vm=float(np.mean(vIFe));am=float(avar.mean())
        diag.update({"mean_IF_variance":vm,"mean_AUDIT_logit_variance":am,
            "AUDIT_over_IF_mean_ratio":am/vm if vm>0 else None,
            "instance_corr_IF_vs_AUDIT_variance":float(np.corrcoef(vIFe,avar)[0,1]) if np.std(vIFe)>0 and np.std(avar)>0 else None})
    write_npz(fitdir/"predictions.npz",z0_cal=z0c,z0_eval=z0e,zR_cal=zRc,zR_eval=zRe,
        zINT_cal=zIC,zINT_eval=zIE,zCALREF_cal=scores_cal["CALREF"],zAUDIT_eval=scores_eval["AUDIT"],
        audit_eval=(scores_eval["AUDIT"]<0).astype(np.int8),vtilde_cal=vtc,vtilde_eval=vte,
        vIF_cal=np.asarray([] if vIFc is None else vIFc),vIF_eval=np.asarray([] if vIFe is None else vIFe),
        method_names=np.asarray(sorted(out),dtype="U64"),q_eval=np.column_stack([out[k] for k in sorted(out)]))
    return methodrows,cvrows,diag,{"fits": {"Tuned_INT6_3D_lambda":fit3["lambda"],"Tuned_INT4_Ratio_lambda":fit4["lambda"]},"cal_maps":len(maps)}


def main(preflight_only=False):
    if WORK.is_symlink() or RAW.is_symlink() or PROTOCOL.is_symlink(): raise RuntimeError("refuse symlink protocol/work paths")
    protocol=json.loads(PROTOCOL.read_text())
    if protocol.get("protocol_id")!="APS_SCANNIA_EXTERNAL_V1_1_20261113" or protocol.get("status")!="FIXED_BEFORE_STABILITY_SCORE_INSPECTION":
        raise RuntimeError("APS protocol is absent or not fixed")
    if file_sha(TRAIN_PATH)!="bb484302e3a3a1c8ef5e1f0129c4dc7cbd58f350867f95b575461ca21ab6b9da" or file_sha(TEST_PATH)!="2cdf6f7661c7b4c63333c93cdec36a3a82350176b604a2312cf82799fb2712f3":
        raise RuntimeError("official APS source digest differs from downloaded source")
    train,test,feat,tr_idx,ca_idx,a_idx,b_idx,audit=parse_data()
    if preflight_only:
        p=WORK/"preflight_v1_1_corrected"/"input_audit.json"
        pp=WORK/"preflight_v1_1_corrected"/"preprocessing_audit.json"
        if p.exists() or p.is_symlink() or pp.exists() or pp.is_symlink(): raise FileExistsError(p if p.exists() else pp)
        prep=fit_preprocessor(train.iloc[a_idx].reset_index(drop=True),feat)
        xa=transform(train.iloc[a_idx],prep);xc=transform(train.iloc[ca_idx],prep);xe=transform(test[feat],prep)
        audit={**audit,"dropped_continuous":prep["dropped_continuous"],"continuous_features_after_filter":len(prep["continuous_columns"]),
            "unscaled_missing_indicators":len(prep["missing_indicator_columns"]),"rare_missing_indicators_dropped":prep["rare_missing_indicators_dropped"],
            "preflight_preprocessing_fit_scope":"Pool A only","preflight_transformed_shapes":{"PoolA":list(xa.shape),"CAL":list(xc.shape),"EVAL_predictors_only":list(xe.shape)},
            "preflight_all_transforms_finite":bool(np.isfinite(xa).all() and np.isfinite(xc).all() and np.isfinite(xe).all())}
        write_json(p,audit)
        write_json(pp,{"preprocessor":prep,"feature_dimension":int(xa.shape[1]),"all_finite":True,
            "EVAL_label_values_accessed":False,"CAL_outcome_labels_used":False})
        print(json.dumps(audit,indent=2))
        return
    if OUT.exists() or OUT.is_symlink(): raise FileExistsError(f"create-only APS model output exists: {OUT}")
    OUT.mkdir(parents=True,exist_ok=False)
    startw=time.monotonic();startc=time.process_time()
    # Fixed resource limits: one CPU thread, 16 GiB address space, 6 h process/wall caps.
    if hasattr(resource,"RLIMIT_AS"):
        soft,hard=resource.getrlimit(resource.RLIMIT_AS); cap=16*(1024**3)
        resource.setrlimit(resource.RLIMIT_AS,(min(cap,hard) if hard!=-1 else cap,min(cap,hard) if hard!=-1 else cap))
    if hasattr(resource,"RLIMIT_CPU"): resource.setrlimit(resource.RLIMIT_CPU,(21600,21630))
    signal.signal(signal.SIGALRM,_timeout);signal.alarm(21600)
    state={"status":"RUNNING","protocol_id":protocol["protocol_id"],"completed_bank_cells":[],"cpu_threads":1,"gpu":False}
    update_state(OUT/"execution_state.json",state)
    atomic_bytes(OUT/"fit_ledger.jsonl",b"")
    all_method=[];all_cv=[];var_rows=[];fit_rows=[];task_rows=[];threshold_rows=[];event_rows=[]
    pending_task={}
    cal_frame=train.iloc[ca_idx].reset_index(drop=True)
    poolA=train.iloc[a_idx].reset_index(drop=True);poolB=train.iloc[b_idx].reset_index(drop=True)
    yA=(poolA["class"].astype(str).to_numpy()=="pos").astype(np.int8);yB=(poolB["class"].astype(str).to_numpy()=="pos").astype(np.int8)
    for bank in range(N_BANKS):
        bdir=OUT/f"bank{bank}";bdir.mkdir()
        prep=fit_preprocessor(poolA,feat);xA=transform(poolA,prep);xB=transform(poolB,prep);xcal_all=transform(cal_frame,prep);xeval_all=transform(test[feat],prep)
        write_json(bdir/"preprocessor.json",prep)
        D0rng=np.random.default_rng(seed(bank,"D0","PoolA_SRSWOR"));d0_idx=np.sort(D0rng.choice(len(poolA),N_MODEL,replace=False))
        bank_task={};bank_threshold={}
        for family in ("Logistic","LightGBM"):
            fdir=bdir/family;fdir.mkdir()
            if family=="Logistic":
                D0,info0=_fit_predict_model(family,xA[d0_idx],yA[d0_idx],np.ones(N_MODEL),seed(bank,"D0",family),fdir/"D0")
            else:
                D0,info0=_fit_predict_model(family,xA[d0_idx],yA[d0_idx],np.ones(N_MODEL),seed(bank,"D0",family),fdir/"D0")
            fit_rows.append({"bank":bank,"family":family,"operation":"shared","role":"D0","train_n":N_MODEL,**info0})
            append_jsonl(OUT/"fit_ledger.jsonl",fit_rows[-1])
            z0cal_all=margins(D0,family,xcal_all);p0cal=expit(np.clip(z0cal_all,-40,40));tau=float(np.quantile(p0cal,.90));gamma=float(logit(np.clip(tau,1e-12,1-1e-12)))
            z0cal_all-=gamma;z0eval_all=margins(D0,family,xeval_all)-gamma
            ac=z0cal_all>=0;ae=z0eval_all>=0;cal_idx=np.flatnonzero(ac);eval_idx=np.flatnonzero(ae)
            z0c=z0cal_all[ac];z0e=z0eval_all[ae]
            bank_threshold[family]={"tau0":tau,"gamma0":gamma,"cal_alert_n":int(ac.sum()),"eval_alert_n":int(ae.sum()),
                "eval_alert_rate":float(ae.mean()),"D0_score_sha256_cal":sha(z0c.astype("<f8").tobytes()),
                "D0_score_sha256_eval":sha(z0e.astype("<f8").tobytes())}
            # Same input order is frozen for every model in this bank/family/op.
            xc_alert=xcal_all[ac];xe_alert=xeval_all[ae]
            if family=="Logistic":
                xD0=with_intercept(xA[d0_idx])
            for op in OPS:
                odir=fdir/op;odir.mkdir()
                if op=="Bootstrap":
                    center=D0;center_info={"operator_center":"D0; bootstrap expected center","new_center_fit":False}
                    k,l=N_MODEL,0
                else:
                    k,l=(8000,8000) if op=="Half_Replace" else (0,16000)
                    if op=="Half_Replace":
                        cx=np.vstack((xA,xB));cy=np.r_[yA,yB];cw=np.r_[np.full(len(xA),k/len(xA)),np.full(len(xB),l/len(xB))]
                    else:
                        cx=xB;cy=yB;cw=np.full(len(yB),l/len(xB))
                    center,center_info=_fit_predict_model(family,cx,cy,cw,seed(bank,op,"operator_center",family),odir/"operator_center")
                    center_info={**center_info,"operator_center":"expected weighted empirical-risk center fit","expected_k":k,"expected_l":l}
                    fit_rows.append({"bank":bank,"family":family,"operation":op,"role":"OperatorCenter","train_n":int(round(cw.sum())),**center_info})
                    append_jsonl(OUT/"fit_ledger.jsonl",fit_rows[-1])
                zRc_all=margins(center,family,xcal_all)-gamma;zRe_all=margins(center,family,xeval_all)-gamma
                zRc=zRc_all[ac];zRe=zRe_all[ae]
                zINTc=[];zINTe=[];zCALREFc=[];zAUDITe=[];sample_arrays={"D0_PoolA_index":d0_idx.astype(np.int32)};op_fit_count=0;op_fit_hashes=[]
                for role,j in ROLE_SPECS:
                    drawseed=seed(bank,op,role,j)
                    rng=np.random.default_rng(drawseed)
                    if op=="Bootstrap":
                        take=rng.integers(0,len(d0_idx),size=N_MODEL);ia=d0_idx[take];ib=np.empty(0,np.int64)
                        xx=xA[ia];yy=yA[ia];ww=np.ones(N_MODEL,float)
                    elif op=="Half_Replace":
                        ia=np.sort(rng.choice(len(xA),8000,replace=False));ib=np.sort(rng.choice(len(xB),8000,replace=False))
                        xx=np.vstack((xA[ia],xB[ib]));yy=np.r_[yA[ia],yB[ib]];ww=np.ones(N_MODEL,float)
                    else:
                        ia=np.empty(0,np.int64);ib=np.sort(rng.choice(len(xB),N_MODEL,replace=False))
                        xx=xB[ib];yy=yB[ib];ww=np.ones(N_MODEL,float)
                    model,mi=_fit_predict_model(family,xx,yy,ww,drawseed,odir/f"{role}_{j:02d}");op_fit_count+=1
                    fit_rows.append({"bank":bank,"family":family,"operation":op,"role":role,"replicate":j,
                        "train_n":N_MODEL,"seed":drawseed,"PoolA_draw_n":len(ia),"PoolB_draw_n":len(ib),**mi})
                    append_jsonl(OUT/"fit_ledger.jsonl",fit_rows[-1])
                    op_fit_hashes.append(mi.get("beta_sha256",mi.get("model_sha256")))
                    sample_arrays[f"{role}_{j:02d}_PoolA_index"]=ia.astype(np.int32);sample_arrays[f"{role}_{j:02d}_PoolB_index"]=ib.astype(np.int32)
                    zc=margins(model,family,xc_alert)-gamma;ze=margins(model,family,xe_alert)-gamma
                    if role=="INT":zINTc.append(zc);zINTe.append(ze)
                    elif role=="CALREF":zCALREFc.append(zc)
                    else:zAUDITe.append(ze)
                zIC=np.column_stack(zINTc);zIE=np.column_stack(zINTe);zCC=np.column_stack(zCALREFc);zAE=np.column_stack(zAUDITe)
                audit=(zAE<0).astype(np.int8);tcal=(zCC<0).mean(axis=1)
                vIFc=vIFe=None;vIF_meta=None
                if family=="Logistic":
                    if op=="Bootstrap":
                        vIFc,vIFe,_,vIF_meta=u1._fit_if(xD0,yA[d0_idx],D0,with_intercept(xc_alert),with_intercept(xe_alert))
                    else:
                        vIFc,vIFe,_,vIF_meta=u1._finite_population_if(with_intercept(xA),yA,with_intercept(xB),yB,center,k,l,
                            with_intercept(xc_alert),with_intercept(xe_alert))
                scorecal={"CALREF":zCC};scoreeval={"AUDIT":zAE}
                methodrows,cvrows,vdiag,fitdiag=_metrics_for_bank(bank,family,op,scorecal,scoreeval,audit,tcal,z0c,z0e,zRc,zRe,
                    zIC,zIE,vIFc,vIFe,odir,seed(bank,op,"CV","INT"))
                all_method.extend(methodrows);all_cv.extend(cvrows);var_rows.append({**vdiag,"IF_status":"COMPUTED" if vIFe is not None else "NOT_AVAILABLE_FOR_TREE_ENSEMBLE",
                    "finite_population_if_seconds":None if vIF_meta is None else vIF_meta.get("total_cpu_seconds"),
                    "role_fit_count":op_fit_count,"operator_center_fit":center_info})
                event_rows.append({"bank":bank,"family":family,"operation":op,"audit_models":11,"alert_n":len(audit),
                    "observed_flip_events":int(audit.sum()),"flip_trials":int(audit.size),"zero_flip_alerts":int(np.sum(audit.sum(axis=1)==0))})
                write_npz(odir/"sampling_indices.npz",**sample_arrays)
                write_json(odir/"operation_manifest.json",{"bank":bank,"family":family,"operation":op,"train_n_per_role":N_MODEL,
                    "D0_alert_threshold_quantile":.90,"D0_alert_threshold_tau":tau,"gamma0":gamma,"CAL_alert_n":int(ac.sum()),
                    "EVAL_alert_n":int(ae.sum()),"role_model_count":23,"unique_model_hash_count":None,
                    "operator_center_fit":center_info,"unique_role_model_hashes":len(set(op_fit_hashes)),
                    "calref_target_definition":"mean of six CALREF margins below zero",
                    "audit_definition":"eleven AUDIT margins below zero","eval_outcomes_used_for_fitting":False,
                    "CV_patient_level":False,"CV_unit":"row; dataset provides no entity key","selected_lambdas":fitdiag["fits"]})
            bank_task[family]={"z0_eval_all":z0eval_all,"p0_eval_all":expit(np.clip(z0eval_all+gamma,-40,40)),"tau":tau}
        # EVAL labels are intentionally accessed after all model/method scores for this bank are fixed.
        for family,vals in bank_task.items():
            pending_task[(bank,family)]={"tau0":vals["tau"],"alert_rate":float(np.mean(vals["z0_eval_all"]>=0)),"p0_eval":vals["p0_eval_all"]}
            threshold_rows.append({"bank":bank,"family":family,**bank_threshold[family]})
        state["completed_bank_cells"].append(bank);state["last_completed_bank"]={"bank":bank,"time_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())};update_state(OUT/"execution_state.json",state)
        print(f"APS bank {bank} complete: alerts={bank_threshold['Logistic']['eval_alert_n']} logistic, {bank_threshold['LightGBM']['eval_alert_n']} lightgbm",flush=True)
        del xA,xB,xcal_all,xeval_all
    # Only now parse EVAL labels. They cannot influence any fit, threshold, CV, or calibration.
    eval_target_frame=pd.read_csv(TEST_PATH,skiprows=20,usecols=["class"],low_memory=False)
    if len(eval_target_frame)!=16000 or not set(eval_target_frame["class"].astype(str).unique())<={"neg","pos"}:
        raise RuntimeError("post-scoring EVAL target schema/value validation failed")
    y_eval=(eval_target_frame["class"].astype(str).to_numpy()=="pos").astype(np.int8)
    for (bank,family),vals in sorted(pending_task.items()):
        task_rows.append({"bank":bank,"family":family,"tau0":vals["tau0"],"D0_eval_alert_rate":vals["alert_rate"],
            **_prediction_metrics(vals["p0_eval"],y_eval)})
    # Main summary: paired six-bank mean/range; ranges are training-bank sensitivity, not population CIs.
    summaries=[]
    for family in ("Logistic","LightGBM"):
        for op in OPS:
            methods=sorted({r["method"] for r in all_method if r["family"]==family and r["operation"]==op})
            for method in methods:
                rr=[r for r in all_method if r["family"]==family and r["operation"]==op and r["method"]==method]
                row={"family":family,"operation":op,"method":method,"banks_n":len(rr)}
                for met in ("R_070","AURC100","AUC_flip","Brier","ECE10"):
                    vv=np.asarray([r[met] for r in rr],float);row[met+"_mean"]=float(vv.mean());row[met+"_min"]=float(vv.min());row[met+"_max"]=float(vv.max())
                ref=[r for r in all_method if r["family"]==family and r["operation"]==op and r["method"]=="Margin"]
                if method!="Margin":
                    d={r["bank"]:r for r in rr};m={r["bank"]:r for r in ref}
                    for met in ("R_070","AURC100","Brier","ECE10"):
                        dif=np.asarray([d[b][met]-m[b][met] for b in range(6)])
                        row[f"delta_{met}_vs_Margin_mean"]=float(dif.mean());row[f"wins_vs_Margin_{met}_out_of_6"]=int(np.sum(dif<0))
                summaries.append(row)
    write_csv(OUT/"model_fit_records.csv",fit_rows)
    write_csv(OUT/"per_bank_method_metrics.csv",all_method);write_csv(OUT/"six_bank_method_summary.csv",summaries)
    write_csv(OUT/"cv_fold_scores.csv",all_cv);write_csv(OUT/"variance_closure.csv",var_rows)
    write_csv(OUT/"D0_task_metrics.csv",task_rows);write_csv(OUT/"thresholds_and_alerts.csv",threshold_rows);write_csv(OUT/"audit_event_counts.csv",event_rows)
    elapsed_cpu=time.process_time()-startc;elapsed_wall=time.monotonic()-startw
    manifest={"status":"COMPLETED","protocol_id":protocol["protocol_id"],"dataset":"UCI APS Failure at Scania Trucks",
        "source_sha256":{"train_csv":file_sha(TRAIN_PATH),"test_csv":file_sha(TEST_PATH)},
        "banks":6,"families":["Logistic","LightGBM"],"operations":list(OPS),"role_models_per_cell":23,
        "D0_models":12,"operator_center_fits":24,"role_model_fits":6*2*3*23,"total_prediction_model_fits":12+24+6*2*3*23,
        "one_dim_calibration_maps":sum(1 for _ in []),"cv_learners_per_cell":2,"cv_fit_count":len(all_cv)+6*2*3*2,
        "one_dim_calibration_maps_count":sum(1 for r in all_method if r["method"] not in ("Tuned_INT6_3D","Tuned_INT4_Ratio")),
        "CAL_outcome_labels_used":False,"EVAL_outcomes_used_for_fit_or_selection":False,"EVAL_outcomes_used_for_final_D0_quality_metrics":True,
        "eval_alert_threshold_quantile":.90,"cv_lambda_grid":list(LAMBDA_GRID),"cpu_threads":1,"gpu":False,
        "cpu_seconds":elapsed_cpu,"wall_seconds":elapsed_wall,"peak_rss_platform_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_units":"KiB on Linux","resource_caps":{"wall_seconds":21600,"address_space_gib":16},
        "cross_bank_range_interpretation":"training-bank sensitivity only; shared official EVAL population, not independent population CI",
        "logistic_if_for_tree":"not applicable; no differentiable parameter-space Sandwich IF was asserted for LightGBM"}
    write_json(OUT/"execution_audit.json",manifest)
    state.update({"status":"COMPLETED","cpu_seconds":elapsed_cpu,"wall_seconds":elapsed_wall});update_state(OUT/"execution_state.json",state)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    try:
        main("--preflight-only" in sys.argv)
    except BaseException as exc:
        out=OUT
        if out.exists() and out.is_dir() and not out.is_symlink():
            try: update_state(out/"execution_state.json",{"status":"INCOMPLETE","exception_type":type(exc).__name__,"exception":str(exc),"time_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
            except Exception: pass
        raise
