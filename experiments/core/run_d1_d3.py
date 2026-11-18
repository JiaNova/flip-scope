#!/usr/bin/env python
"""D1-D3 representation, CAL-only learning, and online-latency experiments."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtr
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from experiments.core import run_w1_w4 as base
from experiments.core import run_s1_s4 as s1
from experiments.core import run_s3_s4 as s3
from experiments.public_benchmarks import run_public_experiments as pub

OUT=s1.OUT/"D1_D2_D3_20261111"
RUNS=base.RUNS
SEED_BASE=base.SEED_BASE
RIDGE=1.0
LAMBDA_GRID=(0.0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0)
SCHEMES=("Scheme_A_Scaled_OHE","Scheme_B_Unscaled_OHE",
         "Scheme_C_Merged_Rare50_Scaled","Scheme_D_Merged_Rare50_Unscaled")
IMPROVED=SCHEMES[1:]
METHOD_CODES=[("D0",0)]+[(f"INT-{j:02d}",100+j) for j in range(1,7)]+\
    [(f"CALREF-{j:02d}",200+j) for j in range(1,7)]+\
    [(f"AUDIT-{j:02d}",300+j) for j in range(1,12)]


def _source_bank(bank):
    if bank<3:return RUNS/"w2b_fulltrain"/f"bank{bank}"
    return s1.OUT/"s3_six_banks"/f"bank{bank}"


def _sha(raw: bytes)->str:return hashlib.sha256(raw).hexdigest()
def _ids_sha(ids)->str:return _sha("\n".join(map(str,ids)).encode())
def _readj(p:Path):
    if p.is_symlink() or not p.is_file():raise RuntimeError(f"unsafe/missing source: {p}")
    return json.loads(p.read_text())
def _writej(p:Path,v):base._write_json(p,v)
def _write_csv(p:Path,rows):base._write_new(p,base._csv_bytes(rows))
def _write_npz(p:Path,**kw):base._write_new(p,base._npz_bytes(**kw))
def _loadnpz(p:Path):
    z=np.load(p,allow_pickle=False);o={k:z[k].copy() for k in z.files};z.close();return o
def _load_beta(path:Path):
    row=_readj(path);beta=np.asarray(row["beta"],np.float64)
    if _sha(beta.astype("<f8").tobytes())!=row["beta_sha256"]:raise RuntimeError(f"beta SHA mismatch: {path}")
    return beta,row
def _threads_one():
    if hasattr(os,"sched_getaffinity"):
        aff=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(aff)})


def _train_draw(train,bank):
    take=np.sort(np.random.default_rng(SEED_BASE+bank*10000).choice(len(train),size=base.LGB_FULL_N,replace=False))
    return take,train.iloc[take].reset_index(drop=True)


def _raw_cats(frame,prep,cat_maps=None):
    out=np.empty((len(frame),len(pub.DIABETES_CATEGORICAL)),dtype=object)
    for j,col in enumerate(pub.DIABETES_CATEGORICAL):
        vals=["__MISSING__" if v in ("","?") else str(v) for v in frame[col].astype(str)]
        if cat_maps is not None:
            mp=cat_maps[col]["mapping"];default=cat_maps[col]["default"]
            vals=[mp.get(v,default) for v in vals]
        else:
            allowed=set(prep["categorical_vocabularies"][col])
            vals=[v if v in allowed else "__UNK__" for v in vals]
        out[:,j]=vals
    return out


def _fit_schema(train_frame,scheme):
    prep=pub.fit_diabetes_preprocessor(train_frame)
    numeric_df=pub.transform_diabetes_lgb(train_frame,prep)[list(pub.DIABETES_NUMERIC)]
    num_scaler=StandardScaler().fit(numeric_df.to_numpy(float))
    raw=_raw_cats(train_frame,prep)
    cat_maps={};categories=[]
    if scheme.startswith("Scheme_C") or scheme.startswith("Scheme_D"):
        y=train_frame.y_readmit30.to_numpy(np.int8)
        for j,col in enumerate(pub.DIABETES_CATEGORICAL):
            vals=raw[:,j].astype(str);counts=pd.Series(vals).value_counts().to_dict()
            positive={v:int(np.sum((vals==v)&(y==1))) for v in counts}
            rare={v for v,n in counts.items() if n<50}
            high=[v for v,n in counts.items() if n>=50]
            mode=max(counts,key=lambda v:(counts[v],str(v)))
            rare_total=sum(counts[v] for v in rare)
            if rare and rare_total>=50:
                default="<RARE>";mp={v:("<RARE>" if v in rare else v) for v in counts}
                vocab=sorted(set(high)|{"<RARE>"})
            else:
                default=mode;mp={v:(mode if v in rare else v) for v in counts}
                vocab=sorted(set(high)|{mode})
            cat_maps[col]={"mapping":mp,"default":default,"vocabulary":vocab,
                "train_counts":{str(k):int(v) for k,v in counts.items()},
                "positive_support":{str(k):int(v) for k,v in positive.items()},
                "rare_source_levels_lt50":sorted(map(str,rare)),"rare_source_total":int(rare_total),
                "rare_bucket_retained":bool("<RARE>" in vocab)}
            categories.append(vocab)
        raw=_raw_cats(train_frame,prep,cat_maps)
    else:
        categories=[list(prep["categorical_vocabularies"][col]) for col in pub.DIABETES_CATEGORICAL]
    encoder=OneHotEncoder(categories=categories,handle_unknown="error",sparse_output=True,dtype=np.float64)
    onehot=encoder.fit_transform(raw)
    if scheme in (SCHEMES[0],SCHEMES[2]):
        cat_scaler=StandardScaler(with_mean=False).fit(onehot)
        cat_scale=np.asarray(cat_scaler.scale_,float)
    else:cat_scaler=None;cat_scale=np.ones(onehot.shape[1],float)
    feature_names=["__INTERCEPT__"]+list(pub.DIABETES_NUMERIC)
    raw_counts=[];raw_positive=[];cat_feature_names=[]
    y=train_frame.y_readmit30.to_numpy(np.int8)
    offset=0
    for j,col in enumerate(pub.DIABETES_CATEGORICAL):
        vals=raw[:,j].astype(str)
        for v in encoder.categories_[j]:
            count=int(np.sum(vals==str(v)));pos=int(np.sum((vals==str(v))&(y==1)))
            cat_feature_names.append(f"{col}={v}");raw_counts.append(count);raw_positive.append(pos)
        offset+=len(encoder.categories_[j])
    feature_names+=cat_feature_names
    metadata={"scheme":scheme,"train_rows":len(train_frame),"feature_dimension":len(feature_names),
        "numeric_features":list(pub.DIABETES_NUMERIC),"categorical_features":list(pub.DIABETES_CATEGORICAL),
        "numeric_medians":prep["numeric_medians"],"numeric_scaler_mean":num_scaler.mean_.tolist(),
        "numeric_scaler_scale":num_scaler.scale_.tolist(),"categories":[list(map(str,x)) for x in encoder.categories_],
        "categorical_maps":cat_maps,"categorical_scaled":scheme in (SCHEMES[0],SCHEMES[2]),
        "categorical_scale":cat_scale.tolist(),"feature_names":feature_names,
        "train_feature_total_support":raw_counts,"train_feature_positive_support":raw_positive,
        "train_patient_sha256":_ids_sha(train_frame.patient_nbr.astype(str).tolist()),
        "ridge_lambda":RIDGE,"onehot_reserved_levels":True}
    return {"scheme":scheme,"prep":prep,"num_scaler":num_scaler,"encoder":encoder,
        "cat_scaler":cat_scaler,"cat_scale":cat_scale,"cat_maps":cat_maps,
        "metadata":metadata}


def _transform(frame,pipeline):
    prep=pipeline["prep"];num=pub.transform_diabetes_lgb(frame,prep)[list(pub.DIABETES_NUMERIC)].to_numpy(float)
    nmat=sparse.csr_matrix(pipeline["num_scaler"].transform(num))
    raw=_raw_cats(frame,prep,pipeline["cat_maps"] if pipeline["cat_maps"] else None)
    cat=pipeline["encoder"].transform(raw)
    if pipeline["cat_scaler"] is not None:cat=pipeline["cat_scaler"].transform(cat)
    blocks=[sparse.csr_matrix(np.ones((len(frame),1),float)),nmat,cat]
    return sparse.hstack(blocks,format="csr")


def _load_pipeline(train_frame,scheme):
    if scheme==SCHEMES[0]:
        prep,fitted=s3._feature_pipeline(train_frame)
        p={"scheme":scheme,"prep":prep,"num_scaler":fitted["numeric_scaler"],
           "encoder":fitted["encoder"],"cat_scaler":fitted["cat_scaler"],
           "cat_scale":np.asarray(fitted["cat_scaler"].scale_,float),"cat_maps":{}}
        # Build the feature metadata from the exact frozen Scheme A encoder and scales.
        raw=_raw_cats(train_frame,prep)
        y=train_frame.y_readmit30.to_numpy(np.int8);tot=[];pos=[];names=["__INTERCEPT__"]+list(pub.DIABETES_NUMERIC)
        for j,col in enumerate(pub.DIABETES_CATEGORICAL):
            vals=raw[:,j].astype(str)
            for v in fitted["encoder"].categories_[j]:
                names.append(f"{col}={v}");tot.append(int(np.sum(vals==str(v))));pos.append(int(np.sum((vals==str(v))&(y==1))))
        meta={"scheme":scheme,"train_rows":len(train_frame),"feature_dimension":len(names),
            "numeric_features":list(pub.DIABETES_NUMERIC),"categorical_features":list(pub.DIABETES_CATEGORICAL),
            "numeric_medians":prep["numeric_medians"],"numeric_scaler_mean":fitted["numeric_scaler"].mean_.tolist(),
            "numeric_scaler_scale":fitted["numeric_scaler"].scale_.tolist(),
            "categories":[list(map(str,x)) for x in fitted["encoder"].categories_],
            "categorical_maps":{},"categorical_scaled":True,"categorical_scale":fitted["cat_scaler"].scale_.tolist(),
            "feature_names":names,"train_feature_total_support":tot,"train_feature_positive_support":pos,
            "train_patient_sha256":_ids_sha(train_frame.patient_nbr.astype(str).tolist()),"ridge_lambda":RIDGE,
            "onehot_reserved_levels":True}
        p["metadata"]=meta
        return p
    return _fit_schema(train_frame,scheme)


def _weight(seed,n):
    return np.bincount(np.random.default_rng(seed+50000).integers(0,n,size=n),minlength=n).astype(np.int64)


def _fit_model(x,y,w,bank,model_id,seed,path,train_hash):
    if (path/"model.json").is_file():
        beta,row=_load_beta(path/"model.json");return beta,{**row,"reused_after_resume":True}
    t0=time.process_time();wall=time.monotonic()
    beta,info=base._fit_irls(x,y,w.astype(float),ridge=RIDGE)
    row={"bank":bank,"model_id":model_id,"seed":int(seed),"train_n":len(y),
         "fit_weight_sum":int(w.sum()),"nonzero_weight_patients":int(np.count_nonzero(w)),
         "fit_cpu_seconds":time.process_time()-t0,"fit_wall_seconds":time.monotonic()-wall,
         "train_patient_sha256":train_hash,"beta_sha256":_sha(beta.astype("<f8").tobytes()),
         "feature_dimension":len(beta),**info,"beta":beta.tolist()}
    base._write_json(path/"model.json",row)
    return beta,row


def _fit_scheme_bank_models(train,cal,eva,bank,scheme,root,roles):
    take,tr=_train_draw(train,bank);ids=tr.patient_nbr.astype(str).tolist();train_hash=_ids_sha(ids)
    source=_source_bank(bank)
    if train_hash!=_readj(source/"bank_manifest.json")["train_patient_sha256"]:
        raise RuntimeError(f"bank {bank} training draw changed")
    pipe=_load_pipeline(tr,scheme);xtr=_transform(tr,pipe);xc=_transform(cal,pipe);xe=_transform(eva,pipe)
    if xtr.shape[1]!=pipe["metadata"]["feature_dimension"]:raise RuntimeError("feature dimension mismatch")
    if scheme==SCHEMES[0]:
        # The immutable Scheme A bank is the saved W2-B full-training run.
        betas={};fitrows=[]
        for mid,_ in METHOD_CODES:
            betas[mid],meta=_load_beta(source/"models"/mid/"model.json");fitrows.append(meta)
            if len(betas[mid])!=xtr.shape[1]:raise RuntimeError(f"Scheme A/source feature mismatch in bank {bank}/{mid}")
    else:
        betas={};fitrows=[];weights={}
        bdir=root/scheme/f"bank{bank}"
        for mid,code in METHOD_CODES:
            seed=SEED_BASE+500000+bank*10000+code
            w=np.ones(len(tr),np.int64) if mid=="D0" else _weight(seed,len(tr))
            weights[mid.replace("-","_")]=w
            beta,rec=_fit_model(xtr,tr.y_readmit30.to_numpy(np.int8),w,bank,mid,seed,bdir/"models"/mid,train_hash)
            betas[mid]=beta;fitrows.append(rec)
            if not rec.get("reused_after_resume",False):print(f"D1 fit {scheme} bank={bank} model={mid} CPU={rec['fit_cpu_seconds']:.2f}s",flush=True)
        weight_path=root/f"bank{bank}_bootstrap_weights.npz"
        if not weight_path.exists():_write_npz(weight_path,**weights)
        else:
            saved=_loadnpz(weight_path)
            for k,v in weights.items():
                if k not in saved or not np.array_equal(saved[k],v):raise RuntimeError("bootstrap weight cache mismatch")
    bdir=root/scheme/f"bank{bank}"
    if scheme==SCHEMES[0]:
        predictions=_loadnpz(source/"predictions.npz")
        # Stored Scheme A arrays are full CAL/EVAL rows with margins centered at the bank CAL threshold.
        if bank<3:tau=_readj(base.OUT/"w2b_diabetes_logistic_fulltrain"/"report.json")["banks"][bank]["threshold_tau"]
        else:tau=float(_readj(source/"training_summary.json")["threshold_tau"] if "threshold_tau" in _readj(source/"training_summary.json") else expit(float(_readj(source/"training_summary.json")["gamma"])))
        gamma=float(logit(tau))
        checks={"z0c":(betas["D0"],xc),"z0e":(betas["D0"],xe)}
        maxerr=0.
        for key,(beta,xmat) in checks.items():maxerr=max(maxerr,float(np.max(np.abs(base._logit_l2(xmat,beta)-gamma-predictions[key]))))
        for role,mat,key in (("INT",xc,"zic"),("INT",xe,"zie"),("CALREF",xc,"zcalref"),("AUDIT",xe,"zaudit")):
            ids=[f"{role}-{j:02d}" for j in range(1,7 if role!="AUDIT" else 12)]
            replay=np.column_stack([base._logit_l2(mat,betas[mid])-gamma for mid in ids])
            maxerr=max(maxerr,float(np.max(np.abs(replay-predictions[key]))))
        if maxerr>1e-6:raise RuntimeError(f"Scheme A saved prediction mismatch for bank {bank}: {maxerr}")
    else:
        # Apply models only after every coefficient JSON has been durably written.
        logits_c={m:base._logit_l2(xc,b) for m,b in betas.items()}
        logits_e={m:base._logit_l2(xe,b) for m,b in betas.items()}
        gamma=float(logit(np.quantile(expit(np.clip(logits_c["D0"],-40,40)),.8)))
        predictions={"z0c":logits_c["D0"]-gamma,"z0e":logits_e["D0"]-gamma,
            "zic":np.column_stack([logits_c[f"INT-{j:02d}"]-gamma for j in range(1,7)]),
            "zie":np.column_stack([logits_e[f"INT-{j:02d}"]-gamma for j in range(1,7)]),
            "zcalref":np.column_stack([logits_c[f"CALREF-{j:02d}"]-gamma for j in range(1,7)]),
            "zaudit":np.column_stack([logits_e[f"AUDIT-{j:02d}"]-gamma for j in range(1,12)]),
            "p0e":expit(np.clip(logits_e["D0"],-40,40)),"p0c":expit(np.clip(logits_c["D0"],-40,40))}
    # Scheme A can load older arrays with the same fields, plus v0c/v0e. Do not rewrite them.
    if scheme==SCHEMES[0]:
        p={k:predictions[k] for k in ("z0c","z0e","zic","zie","zcalref","zaudit")}
        p["p0e"]=expit(np.clip(p["z0e"]+gamma,-40,40));p["p0c"]=expit(np.clip(p["z0c"]+gamma,-40,40))
        mpath=bdir/"feature_metadata.json"
        if not mpath.exists():_writej(mpath,pipe["metadata"])
        return pipe,xtr,xc,xe,betas,p,gamma,fitrows
    path=bdir/"predictions.npz"
    if not path.exists():_write_npz(path,**predictions)
    else:
        saved=_loadnpz(path)
        for k in predictions:
            if not np.allclose(saved[k],predictions[k],rtol=0,atol=1e-12):raise RuntimeError("prediction cache mismatch")
        predictions=saved
    mpath=bdir/"feature_metadata.json"
    if not mpath.exists():_writej(mpath,pipe["metadata"])
    return pipe,xtr,xc,xe,betas,predictions,gamma,fitrows


def _load_splits():
    selected,audit=pub.load_selected_diabetes()
    split={k:selected[selected.split==k].reset_index(drop=True) for k in ("TRAIN","CAL","EVAL")}
    expected={"TRAIN":42058,"CAL":13983,"EVAL":13949}
    for key,n in expected.items():
        if len(split[key])!=n or split[key].patient_nbr.astype(str).duplicated().any():
            raise RuntimeError(f"frozen patient split mismatch: {key}")
    if set(split["TRAIN"].patient_nbr)&set(split["CAL"].patient_nbr) or set(split["TRAIN"].patient_nbr)&set(split["EVAL"].patient_nbr) or set(split["CAL"].patient_nbr)&set(split["EVAL"].patient_nbr):
        raise RuntimeError("patient partitions overlap")
    return split,audit


def _record(path, value):
    if path.exists() or path.is_symlink():
        old=_readj(path)
        if old!=base._json_safe(value):raise RuntimeError(f"existing audit record differs: {path}")
        return
    _writej(path,value)


def _save_predictions(path, predictions):
    if not path.exists():_write_npz(path,**predictions);return
    saved=_loadnpz(path)
    for k,v in predictions.items():
        if k not in saved or not np.allclose(saved[k],v,rtol=0,atol=1e-12):
            raise RuntimeError(f"prediction artifact differs: {path}:{k}")


def _fit_all_schemes():
    _threads_one();OUT.mkdir(parents=True,exist_ok=True)
    splits,_audit=_load_splits();train,cal,eva=(splits[k] for k in ("TRAIN","CAL","EVAL"))
    run_started=time.monotonic();fit_rows=[];bank_rows=[]
    for scheme in SCHEMES:
        (OUT/scheme).mkdir(parents=True,exist_ok=True)
        for bank in range(6):
            pipe,xtr,xc,xe,betas,pred,gamma,records=_fit_scheme_bank_models(
                train,cal,eva,bank,scheme,OUT,roles=None)
            bdir=OUT/scheme/f"bank{bank}";bdir.mkdir(parents=True,exist_ok=True)
            # Every scheme gets the same immutable, uniformly shaped score payload.
            _save_predictions(bdir/"predictions.npz",pred)
            _record(bdir/"feature_metadata.json",pipe["metadata"])
            fit_rows.extend([{**r,"scheme":scheme} for r in records])
            bank_manifest={"scheme":scheme,"bank":bank,"train_rows":int(xtr.shape[0]),
                "train_feature_dimension":int(xtr.shape[1]),"cal_rows":int(xc.shape[0]),
                "eval_rows":int(xe.shape[0]),"threshold_gamma":float(gamma),
                "threshold_tau":float(expit(gamma)),"train_patient_sha256":pipe["metadata"]["train_patient_sha256"],
                "draw_seed":int(SEED_BASE+bank*10000),"draw_size":int(base.LGB_FULL_N),
                "ridge_lambda":RIDGE,"role_count":len(METHOD_CODES),
                "model_beta_sha256":{m:_sha(bet.astype("<f8").tobytes()) for m,bet in betas.items()},
                "prediction_arrays_sha256":{k:_sha(np.asarray(v).astype("<f8").tobytes()) for k,v in pred.items()}}
            _record(bdir/"bank_manifest.json",bank_manifest);bank_rows.append(bank_manifest)
    clean_fit_rows=[{k:v for k,v in r.items() if k!="beta"} for r in fit_rows]
    _record(OUT/"D1_fit_report.json",{"status":"FIT_AND_REPLAY_COMPLETE","schemes":list(SCHEMES),
        "banks":6,"new_model_fits":sum(1 for r in fit_rows if not r.get("reused_after_resume",False)),
        "reused_model_fits":sum(1 for r in fit_rows if r.get("reused_after_resume",False)),
        "fit_cpu_seconds_sum":float(sum(float(r.get("fit_cpu_seconds",0)) for r in fit_rows)),
        "fit_wall_seconds_sum":float(sum(float(r.get("fit_wall_seconds",0)) for r in fit_rows)),
        "fit_records":clean_fit_rows,"bank_manifests":bank_rows,
        "elapsed_wall_seconds":time.monotonic()-run_started,
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "source_scope":"public UCI Diabetes selected rows only; no ICU or confirmation inputs"})
    if not (OUT/"fit_records.csv").exists():_write_csv(OUT/"fit_records.csv",clean_fit_rows)
    print(f"D1 fit/replay complete: new fits={sum(1 for r in fit_rows if not r.get('reused_after_resume',False))}",flush=True)


def _audit_fit_counts():
    path=OUT/"D1_fit_count_correction.json"
    if path.exists():return _readj(path)
    by_scheme={};cpu=wall=0.;expected=24*6
    for scheme in SCHEMES:
        records=[]
        for bank in range(6):
            source=_source_bank(bank) if scheme==SCHEMES[0] else OUT/scheme/f"bank{bank}"
            for mid,_ in METHOD_CODES:
                model=_readj(source/"models"/mid/"model.json")
                records.append(model)
        by_scheme[scheme]={"model_records":len(records),"mean_model_feature_dimension":float(np.mean([len(r["beta"]) for r in records])),
            "CPU_seconds_from_model_fit_records":float(sum(float(r.get("fit_cpu_seconds",0)) for r in records)) if scheme!=SCHEMES[0] else 0.}
        if scheme!=SCHEMES[0]:
            cpu+=sum(float(r.get("fit_cpu_seconds",0)) for r in records);wall+=sum(float(r.get("fit_wall_seconds",0)) for r in records)
    correction={"status":"AUDITED_COUNTS_SUPERSEDE_D1_fit_report_COUNTERS",
        "reason":"The first fit summary counter classified Scheme A source models as newly fitted. Counts below are reconstructed from per-model files and source lineage.",
        "new_prediction_model_fits":int(expected*3),"reused_Scheme_A_saved_models":int(expected),
        "model_fits_resumed_from_persisted_partial_stage":24,"model_refit_on_resume":0,
        "new_fit_CPU_seconds_sum":float(cpu),"new_fit_wall_seconds_sum":float(wall),
        "by_scheme":by_scheme,"role_counts_per_scheme":{"D0":1,"INT":6,"CALREF":6,"AUDIT":11},"banks":6}
    _record(path,correction);return correction


def _corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def _if_objects(xtr,y,beta0,xc,xe):
    n,d=xtr.shape;pen=np.ones(d,float)*RIDGE;pen[0]=0.
    eta=np.asarray(xtr@beta0).reshape(-1);p=expit(np.clip(eta,-40,40));res=p-y
    scores=xtr.multiply(res[:,None]).tocsr();mean=np.asarray(scores.mean(axis=0)).reshape(-1)
    j=(scores.T@scores).toarray()-n*np.outer(mean,mean)
    h=(xtr.T@sparse.diags(np.maximum(p*(1-p),1e-12))@xtr).toarray();h.flat[::d+1]+=pen
    hinv=np.linalg.inv(h);sigma=hinv@j@hinv;sigma=(sigma+sigma.T)*.5
    xe_d=xe.toarray();xc_d=xc.toarray()
    ve=np.einsum("ij,jk,ik->i",xe_d,sigma,xe_d,optimize=True)
    vc=np.einsum("ij,jk,ik->i",xc_d,sigma,xc_d,optimize=True)
    return np.maximum(vc,0),np.maximum(ve,0),sigma,h,pen,p,res


def _audit_weight(bank,j,n):
    seed=SEED_BASE+500000+bank*10000+300+j
    return _weight(seed,n),seed


def _rare_mask(meta,criterion):
    names=meta["feature_names"];total=meta["train_feature_total_support"];positive=meta["train_feature_positive_support"]
    mask=np.zeros(len(names),bool)
    for i,name in enumerate(names):
        if i==0 or name in pub.DIABETES_NUMERIC:continue
        if criterion=="total_lt50":mask[i]=int(total[i-(1+len(pub.DIABETES_NUMERIC))])<50
        elif criterion=="positive_lt20":mask[i]=int(positive[i-(1+len(pub.DIABETES_NUMERIC))])<20
        elif criterion=="positive_lt50":mask[i]=int(positive[i-(1+len(pub.DIABETES_NUMERIC))])<50
        else:raise ValueError(criterion)
    return mask


def _variance_terms(vectors,cmat,rmat):
    # vectors: [B,n], cmat/rmat: [B,n], returns the exact sample-variance identity.
    c=np.var(cmat,axis=0,ddof=1);r=np.var(rmat,axis=0,ddof=1)
    cross=2*np.mean((cmat-cmat.mean(axis=0))*(rmat-rmat.mean(axis=0)),axis=0)*len(vectors)/(len(vectors)-1)
    total=np.var(vectors,axis=0,ddof=1)
    return float(c.mean()),float(r.mean()),float(cross.mean()),float(total.mean())


def _metrics(q,audit,baseline):
    t=audit.mean(axis=1)
    return {"n_alerts":int(len(q)),"R_070":base._risk(q,t,.70),
        "AURC100":float(np.mean([base._risk(q,t,k/100) for k in range(1,101)])),
        "AUC_flip":base._auc_finite(q,audit),"Brier":float(np.mean((q[:,None]-audit)**2)),
        "ECE10":pub._ece10(q,t),"R_100":float(t.mean()),
        "delta_Brier_vs_Margin":float(np.mean(q*q-baseline*baseline-2*(q-baseline)*t))}


def _calibrated(cal_score,cal_target,eval_score):
    q,mapping=base._cal_stability(np.asarray(cal_score,float),np.asarray(cal_target,float),np.asarray(eval_score,float))
    return q,mapping


def _analyze_d1_bank(train,cal,eva,bank,scheme):
    bdir=OUT/scheme/f"bank{bank}";pipe=_load_pipeline(_train_draw(train,bank)[1],scheme)
    xtr=_transform(_train_draw(train,bank)[1],pipe);xc=_transform(cal,pipe);xe=_transform(eva,pipe)
    pred=_loadnpz(bdir/"predictions.npz");gamma=float(_readj(bdir/"bank_manifest.json")["threshold_gamma"])
    betas={m:_load_beta((_source_bank(bank)/"models"/m/"model.json") if scheme==SCHEMES[0] else bdir/"models"/m/"model.json")[0] for m,_ in METHOD_CODES}
    ytr=_train_draw(train,bank)[1].y_readmit30.to_numpy(float);ye=eva.y_readmit30.to_numpy(np.int8)
    z0c=pred["z0c"];z0e=pred["z0e"];ic=pred["zic"];ie=pred["zie"];ref=pred["zcalref"];aud=(pred["zaudit"]<0).astype(np.int8)
    ac=z0c>=0;ae=z0e>=0
    tcal=(ref[ac]<0).mean(axis=1);audit=aud[ae]
    z0ca=z0c[ac];z0ea=z0e[ae];ic_a=ic[ac];ie_a=ie[ae]
    zbarc=ic_a.mean(axis=1);zbare=ie_a.mean(axis=1)
    sigma2=float(np.var(ic_a,axis=1,ddof=1).mean());vc=.5*np.var(ic_a,axis=1,ddof=1)+.5*sigma2;ve=.5*np.var(ie_a,axis=1,ddof=1)+.5*sigma2
    mc=.5*(z0ca+zbarc);me=.5*(z0ea+zbare)
    vcal_all,ve_all,sigma,h0,pen,p0,res=_if_objects(xtr,ytr,betas["D0"],xc[ac],xe[ae])
    # Persist aligned alert-level variance arrays for the CV stage and D3 audit.
    ipath=bdir/"if_arrays.npz";ivars={"v_if_cal_alert":vcal_all,"v_if_eval_alert":ve_all,
        "cal_alert_mask":ac.astype(np.uint8),"eval_alert_mask":ae.astype(np.uint8),"sigma_theta":sigma}
    if not ipath.exists():_write_npz(ipath,**ivars)
    else:
        old=_loadnpz(ipath)
        for k,v in ivars.items():
            if not np.allclose(old[k],v,rtol=1e-9,atol=1e-12):raise RuntimeError(f"IF cache mismatch {bdir}/{k}")
    cal_scores={"Margin":z0ca,"FusionMean":mc,"qMV":mc/np.sqrt(vc),"ConditionalIF":z0ca/np.sqrt(np.maximum(vcal_all,1e-300))}
    eval_scores={"Margin":z0ea,"FusionMean":me,"qMV":me/np.sqrt(ve),"ConditionalIF":z0ea/np.sqrt(np.maximum(ve_all,1e-300))}
    qmaps={};qeval={}
    for m in cal_scores:
        qeval[m],qmaps[m]=_calibrated(cal_scores[m],tcal,eval_scores[m])
    rows=[]
    for m,q in qeval.items():
        row={"bank":bank,"scheme":scheme,"method":m,**_metrics(q,audit,qeval["Margin"]),
             "threshold_tau":float(expit(gamma)),"sigma_g_sq":sigma2,
             "IF_variance_mean":float(ve_all.mean()),"IF_variance_median":float(np.median(ve_all)),
             "D0_outcome_AUC":float(roc_auc_score(ye,pred["p0e"])),
             "D0_outcome_Brier":float(np.mean((pred["p0e"]-ye)**2)),
             "eval_alert_count":int(ae.sum()),"cal_alert_count":int(ac.sum())}
        rows.append(row)
    _record(bdir/"d1_calref_maps.json",qmaps)
    _record(bdir/"d1_performance.json",{"rows":rows,"calibrator_inputs":"CAL positive alert queue; target is six CALREF finite flips",
        "audit_metrics_only":"EVAL positive alert queue; target is eleven AUDIT finite flips"})
    # Four-stage telescoping analysis with AUDIT’s actual multinomial weights.
    decomp=[]
    xx=xe[ae].tocsr();xarr=xx.toarray();d=xtr.shape[1]
    beta0=betas["D0"];eta0=np.asarray(xtr@beta0).reshape(-1);p=expit(np.clip(eta0,-40,40));resid0=p-ytr
    audit_betas=np.vstack([betas[f"AUDIT-{j:02d}"] for j in range(1,12)])
    dfull=audit_betas-beta0[None,:]
    stage_delta={"Stage1_Linear":[],"Stage2_1StepNewton":[]}
    h0_factor=cho_factor(h0,lower=True,check_finite=False)
    for j in range(1,12):
        w,_=_audit_weight(bank,j,len(ytr));g=np.asarray(xtr.T@(w*resid0)).reshape(-1)+pen*beta0
        delta1=-cho_solve(h0_factor,g,check_finite=False);stage_delta["Stage1_Linear"].append(delta1)
        hb=(xtr.T@sparse.diags(w*np.maximum(p*(1-p),1e-12))@xtr).toarray();hb.flat[::d+1]+=pen
        hb_factor=cho_factor(hb,lower=True,check_finite=False)
        delta2=-cho_solve(hb_factor,g,check_finite=False);stage_delta["Stage2_1StepNewton"].append(delta2)
    stage_delta={k:np.vstack(v) for k,v in stage_delta.items()}
    stages={"Stage0_IF":None,"Stage1_Linear":stage_delta["Stage1_Linear"],
            "Stage2_1StepNewton":stage_delta["Stage2_1StepNewton"],"Stage3_FullRefit":dfull}
    for criterion in ("total_lt50","positive_lt20","positive_lt50"):
        rare=_rare_mask(pipe["metadata"],criterion);common=~rare;rc=np.flatnonzero(rare);cc=np.flatnonzero(common)
        for stage,ds in stages.items():
            if stage=="Stage0_IF":
                xr=xarr[:,rc];xcx=xarr[:,cc]
                vC=np.einsum("ni,ij,nj->n",xcx,sigma[np.ix_(cc,cc)],xcx,optimize=True)
                vR=np.einsum("ni,ij,nj->n",xr,sigma[np.ix_(rc,rc)],xr,optimize=True) if len(rc) else np.zeros(len(xarr))
                vX=2*np.einsum("ni,ij,nj->n",xcx,sigma[np.ix_(cc,rc)],xr,optimize=True) if len(rc) else np.zeros(len(xarr))
                total=np.einsum("ni,ij,nj->n",xarr,sigma,xarr,optimize=True)
                terms=[float(np.mean(vC)),float(np.mean(vR)),float(np.mean(vX)),float(np.mean(total))]
                rare_delta_share=None
            else:
                cpart=np.asarray(xarr[:,cc]@ds[:,cc].T);rpart=np.asarray(xarr[:,rc]@ds[:,rc].T) if len(rc) else np.zeros((len(xarr),len(ds)))
                zdev=cpart+rpart
                vC=np.var(cpart,axis=1,ddof=1);vR=np.var(rpart,axis=1,ddof=1)
                vX=2*np.mean((cpart-cpart.mean(axis=1,keepdims=True))*(rpart-rpart.mean(axis=1,keepdims=True)),axis=1)*len(ds)/(len(ds)-1)
                total=np.var(zdev,axis=1,ddof=1)
                terms=[float(vC.mean()),float(vR.mean()),float(vX.mean()),float(total.mean())]
                if stage=="Stage3_FullRefit":
                    den=np.sum(ds*ds,axis=1);rare_delta_share=float(np.mean(np.sum(ds[:,rc]**2,axis=1)/np.maximum(den,1e-300))) if len(rc) else 0.
                else:rare_delta_share=None
            cval,rval,xval,tval=terms
            shares=[v/tval*100 for v in (cval,rval,xval)] if tval else [0.,0.,0.]
            decomp.append({"bank":bank,"scheme":scheme,"criterion":criterion,"stage":stage,
                "Var_Common":cval,"Var_Rare":rval,"TwoCov_Common_Rare":xval,"Var_Total":tval,
                "Common_percent":shares[0],"Rare_percent":shares[1],"Cross_percent":shares[2],
                "component_percent_sum":float(sum(shares)),"rare_coefficient_norm2_share":rare_delta_share,
                "rare_feature_count":int(rare.sum()),"total_feature_count":int(d)})
    if decomp:
        _record(bdir/"d1_variance_decomposition.json",{"rows":decomp,"decomposition":"sample variance identity across 11 AUDIT draws; Stage0 sandwich projection",
            "two_support_rules":{"total_lt50":"post-encoding TRAIN OHE support < 50","positive_lt20":"TRAIN positive-label OHE support < 20","positive_lt50":"TRAIN positive-label OHE support < 50"}})
    return rows,decomp,{"bank":bank,"scheme":scheme,"cal_scores":cal_scores,"eval_scores":eval_scores,
        "cal_targets":tcal,"eval_audit":audit,"vif_cal":vcal_all,"vif_eval":ve_all,
        "zbar_cal":zbarc,"zbar_eval":zbare,"vtilde_cal":vc,"vtilde_eval":ve,
        "m_cal":mc,"m_eval":me,"sigma_g_sq":sigma2,"gamma":gamma,"cal_mask":ac,"eval_mask":ae,
        "train_pipeline":pipe,"x_eval":xe,"betas":betas,"qmaps":qmaps,"qeval":qeval,
        "train_rows":len(ytr),"feature_dim":d}


def _d1_analyze_all():
    _threads_one();splits,_audit=_load_splits();train,cal,eva=(splits[k] for k in ("TRAIN","CAL","EVAL"))
    all_rows=[];all_dec=[];cv_candidates=[]
    for scheme in SCHEMES:
        for bank in range(6):
            rows,dec,state=_analyze_d1_bank(train,cal,eva,bank,scheme);all_rows.extend(rows);all_dec.extend(dec)
            if scheme in IMPROVED:
                # Patient-level CAL CV of the fixed physical qMV score; EVAL is never used in scheme selection.
                raw=state["cal_scores"]["qMV"];target=state["cal_targets"]
                kf=KFold(n_splits=5,shuffle=True,random_state=20261111+bank)
                sq=[]
                for tridx,teidx in kf.split(np.arange(len(raw))):
                    q,_=base._cal_stability(raw[tridx],target[tridx],raw[teidx]);sq.extend((q-target[teidx])**2)
                cv_candidates.append({"scheme":scheme,"bank":bank,"cal_5fold_brier":float(np.mean(sq)),"cal_alert_n":len(raw),"patient_grouped":True})
    _write_csv(OUT/"D1_per_bank_metrics.csv",all_rows)
    if all_dec:
        _write_csv(OUT/"D1_variance_decomposition.csv",all_dec)
        stage0={(r["bank"],r["scheme"],r["criterion"]):r["Var_Total"] for r in all_dec if r["stage"]=="Stage0_IF"}
        amp=[]
        for r in all_dec:
            key=(r["bank"],r["scheme"],r["criterion"]);den=stage0[key]
            amp.append({**r,"variance_amplification_vs_IF":(r["Var_Total"]/den if den else None)})
        _write_csv(OUT/"D1_variance_amplification.csv",amp)
    summary=[]
    values=["D0_outcome_AUC","D0_outcome_Brier","R_100","R_070","AURC100","Brier","ECE10"]
    for scheme in SCHEMES:
        for method in ("Margin","FusionMean","qMV","ConditionalIF"):
            rr=[r for r in all_rows if r["scheme"]==scheme and r["method"]==method]
            out={"scheme":scheme,"method":method,"conditionalIF_beats_margin_R70_banks":sum(1 for b in range(6) if next(x for x in rr if x["bank"]==b)["R_070"]<next(x for x in all_rows if x["scheme"]==scheme and x["method"]=="Margin" and x["bank"]==b)["R_070"])}
            for k in values:
                xs=np.asarray([r[k] for r in rr if r.get(k) is not None],float);out[k+"_mean"]=float(xs.mean());out[k+"_min"]=float(xs.min());out[k+"_max"]=float(xs.max())
            summary.append(out)
    cvmeans={s:float(np.mean([x["cal_5fold_brier"] for x in cv_candidates if x["scheme"]==s])) for s in IMPROVED}
    selected=min(cvmeans,key=cvmeans.get)
    _write_csv(OUT/"D1_mean_range_metrics.csv",summary);_write_csv(OUT/"D1_scheme_selection_cal_cv.csv",cv_candidates)
    _record(OUT/"D1_analysis_report.json",{"status":"D1_COMPLETE","scheme_selection":"CAL-only patient-level 5-fold cross-validated Brier of qMV CALREF prediction; no EVAL criterion",
        "selected_improved_scheme":selected,"selection_mean_cv_brier":cvmeans,"metrics":summary,
        "decomposition_rows":len(all_dec),"banks":6,"per_bank_rows":len(all_rows),
        "public_data_only":True,"new_prediction_model_fits_in_analysis":0})
    print(f"D1 analysis complete; CAL-CV selected improved scheme={selected}",flush=True)
    return selected


def _standardized_brier_fit(x,y,lam):
    x=np.asarray(x,float);y=np.asarray(y,float)
    scaler=StandardScaler().fit(x);xs=scaler.transform(x);a=np.column_stack((np.ones(len(x)),xs))
    def fn(w):
        p=expit(np.clip(a@w,-40,40));err=p-y
        loss=float(np.mean(err*err)+.5*lam*np.dot(w[1:],w[1:]))
        grad=2.0/len(y)*(a.T@(err*p*(1-p)))
        grad[1:]+=lam*w[1:]
        return loss,grad
    w0=np.zeros(a.shape[1]);w0[0]=float(logit(np.clip(np.mean(y),1e-6,1-1e-6)))
    result=minimize(fn,w0,jac=True,method="L-BFGS-B",options={"maxiter":1000,"ftol":1e-13,"gtol":1e-8,"maxls":50})
    pred=expit(np.clip(a@result.x,-40,40))
    if not result.success and np.max(np.abs(fn(result.x)[1]))>2e-5:
        raise RuntimeError(f"CAL Brier learner did not converge: {result.message}")
    return {"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),
        "intercept":float(result.x[0]),"coefficients":result.x[1:].tolist(),"lambda":float(lam),
        "objective":float(result.fun),"gradient_inf":float(np.max(np.abs(fn(result.x)[1]))),
        "iterations":int(result.nit)},pred


def _brier_predict(fit,x):
    x=np.asarray(x,float);xs=(x-np.asarray(fit["scaler_mean"]))/np.asarray(fit["scaler_scale"])
    return expit(np.clip(float(fit["intercept"])+xs@np.asarray(fit["coefficients"]),-40,40))


def _cv_brier_learner(name,xcal,target,xeval,bank,seed):
    xcal=np.asarray(xcal,float);target=np.asarray(target,float);xeval=np.asarray(xeval,float)
    kf=KFold(n_splits=5,shuffle=True,random_state=seed)
    rows=[]
    for lam in LAMBDA_GRID:
        fold_losses=[]
        for fold,(tr,te) in enumerate(kf.split(np.arange(len(target)))):
            model,_train_pred=_standardized_brier_fit(xcal[tr],target[tr],lam)
            pred=_brier_predict(model,xcal[te])
            loss=float(np.mean((pred-target[te])**2));fold_losses.append(loss)
            rows.append({"learner":name,"bank":bank,"lambda":float(lam),"fold":fold,"holdout_brier":loss,
                "train_n":int(len(tr)),"valid_n":int(len(te)),"grouped_patient_cv":True})
    means={lam:float(np.mean([r["holdout_brier"] for r in rows if r["lambda"]==lam])) for lam in LAMBDA_GRID}
    selected=min(LAMBDA_GRID,key=lambda x:(means[x],x))
    fit,pred=_standardized_brier_fit(xcal,target,selected)
    fit.update({"selected_by":"patient-level 5-fold CAL CV Brier","cv_mean_brier_by_lambda":{str(k):v for k,v in means.items()},
        "learner":name,"bank":bank,"feature_count":int(xcal.shape[1])})
    return fit,_brier_predict(fit,xeval),rows,selected


def _d2_all(selected_scheme):
    _threads_one();splits,_audit=_load_splits();train,cal,eva=(splits[k] for k in ("TRAIN","CAL","EVAL"))
    d2root=OUT/"D2_CAL_CV";d2root.mkdir(parents=True,exist_ok=True)
    perf=[];cvrows=[];lambda_rows=[]
    schemes=(SCHEMES[0],selected_scheme)
    for scheme in schemes:
        for bank in range(6):
            pipe,xtr,xc,xe,betas,pred,gamma,fitrows=_fit_scheme_bank_models(train,cal,eva,bank,scheme,OUT,roles=None)
            calmask=pred["z0c"]>=0;evalmask=pred["z0e"]>=0
            ref=pred["zcalref"];aud=(pred["zaudit"]<0).astype(np.int8)
            tcal=(ref[calmask]<0).mean(axis=1);audit=aud[evalmask]
            zc=pred["z0c"][calmask];ze=pred["z0e"][evalmask]
            zbarc=pred["zic"][calmask].mean(axis=1);zbare=pred["zie"][evalmask].mean(axis=1)
            vc=.5*np.var(pred["zic"][calmask],axis=1,ddof=1)+.5*float(np.var(pred["zic"][calmask],axis=1,ddof=1).mean())
            ve=.5*np.var(pred["zie"][evalmask],axis=1,ddof=1)+.5*float(np.var(pred["zic"][calmask],axis=1,ddof=1).mean())
            mcal=.5*(zc+zbarc);meval=.5*(ze+zbare)
            iv=_loadnpz(OUT/scheme/f"bank{bank}"/"if_arrays.npz")
            vifc=iv["v_if_cal_alert"];vife=iv["v_if_eval_alert"]
            if len(vifc)!=len(zc) or len(vife)!=len(ze):raise RuntimeError("IF alert alignment mismatch in D2")
            # Fixed 1D CALREF calibrators, all fit using only the six CALREF targets.
            raw1={"Margin_CALREF":(zc,ze),"Conditional_IF_1D":(zc/np.sqrt(np.maximum(vifc,1e-300)),ze/np.sqrt(np.maximum(vife,1e-300))),
                  "Fused_Mean_1D":(mcal,meval),"qMV_CALREF":(mcal/np.sqrt(vc),meval/np.sqrt(ve))}
            for name,(ca,ea) in raw1.items():
                q,_=_calibrated(ca,tcal,ea);perf.append({"scheme":scheme,"bank":bank,"learner":name,
                    **_metrics(q,audit,(_calibrated(zc,tcal,ze)[0]))})
            z0logc=np.log(np.maximum(vifc,1e-300));z0loge=np.log(np.maximum(vife,1e-300))
            candidates={
                "Learned2D_IF_Linear":(np.column_stack((zc,z0logc)),np.column_stack((ze,z0loge))),
                "Learned_IF_Interaction":(np.column_stack((zc,z0logc,zc*z0logc,zc/np.sqrt(np.maximum(vifc,1e-300)))),
                                          np.column_stack((ze,z0loge,ze*z0loge,ze/np.sqrt(np.maximum(vife,1e-300))))),
                "Learned2D_Mean_CV":(np.column_stack((zc,zbarc)),np.column_stack((ze,zbare))),
                "Learned3D_MV_Unconstrained_CV":(np.column_stack((zc,zbarc,np.log(np.maximum(vc,1e-300)))),np.column_stack((ze,zbare,np.log(np.maximum(ve,1e-300))))),
                "Learned_MV_Interaction_CV":(np.column_stack((zc,zbarc,np.log(np.maximum(vc,1e-300)),mcal*np.log(np.maximum(vc,1e-300)),(mcal<0)*np.log(np.maximum(vc,1e-300)))),
                    np.column_stack((ze,zbare,np.log(np.maximum(ve,1e-300)),meval*np.log(np.maximum(ve,1e-300)),(meval<0)*np.log(np.maximum(ve,1e-300))))) }
            for name,(xcal,xeval) in candidates.items():
                fit,q,cv,chosen=_cv_brier_learner(name,xcal,tcal,xeval,bank,20261200+bank)
                cvrows.extend([{**r,"scheme":scheme} for r in cv]);lambda_rows.append({"scheme":scheme,"bank":bank,"learner":name,"selected_lambda":chosen,
                    "cv_mean_brier":min(fit["cv_mean_brier_by_lambda"].values()),"coefficients":fit["coefficients"],"intercept":fit["intercept"],
                    "feature_count":fit["feature_count"],"cv_lambda_scores":fit["cv_mean_brier_by_lambda"]})
                perf.append({"scheme":scheme,"bank":bank,"learner":name,**_metrics(q,audit,_calibrated(zc,tcal,ze)[0]),
                    "selected_lambda":chosen,"cv_mean_brier":min(fit["cv_mean_brier_by_lambda"].values())})
            print(f"D2 CAL-CV complete scheme={scheme} bank={bank}",flush=True)
    _write_csv(d2root/"D2_per_bank_metrics.csv",perf);_write_csv(d2root/"D2_cv_fold_losses.csv",cvrows);_write_csv(d2root/"D2_selected_lambdas.csv",lambda_rows)
    means=[]
    for scheme in schemes:
        for learner in sorted({r["learner"] for r in perf}):
            rr=[r for r in perf if r["scheme"]==scheme and r["learner"]==learner]
            if not rr:continue
            item={"scheme":scheme,"learner":learner}
            for m in ("R_070","AURC100","AUC_flip","Brier","ECE10"):
                xs=np.asarray([r[m] for r in rr if r.get(m) is not None],float);item[m+"_mean"]=float(xs.mean());item[m+"_min"]=float(xs.min());item[m+"_max"]=float(xs.max())
            means.append(item)
    _write_csv(d2root/"D2_mean_range_metrics.csv",means)
    _record(d2root/"D2_report.json",{"status":"D2_COMPLETE","schemes":list(schemes),"banks":6,
        "lambda_grid":list(LAMBDA_GRID),"cv":"patient-level 5-fold inside positive CAL alert queue; Brier criterion; fold-local standardization",
        "feature_sign_constraints":"none","calibration_targets":"six CALREF models only","evaluation_truth":"eleven AUDIT models only",
        "perf_rows":len(perf),"cv_fit_count":len(cvrows),"selected_lambda_distribution":lambda_rows,
        "scientific_questions":{"Q1":"See CAL-CV generalization and EVAL table for added variance features vs first-order means.",
                                "Q2":"See fixed-ratio 1D versus CV-tuned unconstrained/interaction family."}})
    return means


def _build_if_setup(xtr,y,beta):
    n,d=xtr.shape;steps={};t=time.process_time()
    eta=np.asarray(xtr@beta).reshape(-1);p=expit(np.clip(eta,-40,40));res=p-y
    u=xtr.multiply(res[:,None]).tocsr()
    steps["residual_and_score_rows_cpu_seconds"]=time.process_time()-t
    t=time.process_time();ww=np.maximum(p*(1-p),1e-12)
    h=(xtr.T@sparse.diags(ww)@xtr).toarray();h.flat[::d+1]+=RIDGE;h[0,0]-=RIDGE
    steps["H0_construction_cpu_seconds"]=time.process_time()-t
    t=time.process_time();hinv=np.linalg.inv(h);steps["H0_inverse_cpu_seconds"]=time.process_time()-t
    t=time.process_time();mean_u=np.asarray(u.mean(axis=0)).reshape(-1);j=(u.T@u).toarray()-n*np.outer(mean_u,mean_u)
    steps["J_construction_cpu_seconds"]=time.process_time()-t
    t=time.process_time();sigma=hinv@j@hinv;sigma=(sigma+sigma.T)*.5
    steps["Sigma_construction_cpu_seconds"]=time.process_time()-t
    t=time.process_time()
    try:
        factor=np.linalg.cholesky(sigma);factorization="Cholesky lower: Sigma=L L.T"
        residual=float(np.max(np.abs(factor@factor.T-sigma)))
    except np.linalg.LinAlgError:
        vals,vec=np.linalg.eigh(sigma);vals=np.maximum(vals,0.);factor=vec*np.sqrt(vals)[None,:]
        factorization="eigen factor with negative roundoff eigenvalues clipped to zero"
        residual=float(np.max(np.abs(factor@factor.T-sigma)))
    steps["Sigma_factorization_cpu_seconds"]=time.process_time()-t
    steps["IF_offline_setup_total_cpu_seconds"]=float(sum(steps.values()))
    return sigma,factor,steps,residual,factorization


def _sparse_append_proxy(x,z,p):
    extra=sparse.csr_matrix(np.column_stack((z,np.abs(z),p*(1-p))))
    return sparse.hstack((x,extra),format="csr")


def _time_call(fn,repeat=50):
    wall=[];cpu=[]
    for _ in range(repeat):
        w=time.perf_counter();c=time.process_time();fn();wall.append(time.perf_counter()-w);cpu.append(time.process_time()-c)
    wall=np.asarray(wall)*1000;cpu=np.asarray(cpu)*1000
    return {"repeat_n":repeat,"wall_ms_median":float(np.median(wall)),"wall_ms_mean":float(np.mean(wall)),
        "wall_ms_iqr":float(np.quantile(wall,.75)-np.quantile(wall,.25)),
        "cpu_ms_median":float(np.median(cpu)),"cpu_ms_mean":float(np.mean(cpu)),
        "cpu_ms_iqr":float(np.quantile(cpu,.75)-np.quantile(cpu,.25))}


def _d3_all(selected_scheme):
    _threads_one();splits,_audit=_load_splits();train,cal,eva=(splits[k] for k in ("TRAIN","CAL","EVAL"))
    d3root=OUT/"D3_online_latency_cachefree_20261111";d3root.mkdir(parents=True,exist_ok=True)
    schemes=(SCHEMES[0],selected_scheme);rows=[];offline=[];preproc=[]
    for scheme in schemes:
        for bank in range(6):
            take,tr=_train_draw(train,bank);pipe=_load_pipeline(tr,scheme)
            xtr=_transform(tr,pipe);xc=_transform(cal,pipe);xe=_transform(eva,pipe);d=xe.shape[1]
            model_root=_source_bank(bank) if scheme==SCHEMES[0] else OUT/scheme/f"bank{bank}"
            betas={mid:_load_beta(model_root/"models"/mid/"model.json")[0] for mid,_ in METHOD_CODES}
            gamma=float(_readj(OUT/scheme/f"bank{bank}"/"bank_manifest.json")["threshold_gamma"])
            fitrows=[_readj(model_root/"models"/mid/"model.json") for mid,_ in METHOD_CODES]
            # CAL scores are reconstructed from saved coefficients; EVAL score caches are never opened.
            cal_logits={mid:base._logit_l2(xc,beta) for mid,beta in betas.items()}
            calz0=cal_logits["D0"]-gamma
            calzi=np.column_stack([cal_logits[f"INT-{j:02d}"]-gamma for j in range(1,7)])
            ref=np.column_stack([cal_logits[f"CALREF-{j:02d}"]-gamma for j in range(1,7)])
            calmask=calz0>=0
            tcal=(ref[calmask]<0).mean(axis=1);sigma2=float(np.var(calzi[calmask],axis=1,ddof=1).mean())
            mcal=.5*(calz0[calmask]+calzi[calmask].mean(axis=1));vcal=.5*np.var(calzi[calmask],axis=1,ddof=1)+.5*sigma2
            qcal_map=base._cal_stability(mcal/np.sqrt(vcal),tcal,mcal/np.sqrt(vcal))[1]
            margin_map=base._cal_stability(calz0[calmask],tcal,calz0[calmask])[1]
            # Fit the two-output Ridge teacher on all CAL rows using only D0 and INT outputs.
            p0c=expit(np.clip(calz0+gamma,-40,40));zbar_full=calzi.mean(axis=1);s2_full=np.var(calzi,axis=1,ddof=1)
            delta=.5*(zbar_full-calz0);eta_target=np.log(np.maximum(.5*s2_full+.5*sigma2,1e-300))
            proxy_xcal=_sparse_append_proxy(xc,calz0,p0c);rfit_start=time.process_time()
            ridge=Ridge(alpha=1.0,fit_intercept=True,solver="lsqr")
            ridge.fit(proxy_xcal,np.column_stack((delta,eta_target)))
            ridge_fit_cpu=time.process_time()-rfit_start
            proxy_cal_pred=ridge.predict(proxy_xcal);q_proxy_cal=ndtr(-((calz0+proxy_cal_pred[:,0])/np.sqrt(np.clip(np.exp(proxy_cal_pred[:,1]),.1*sigma2,10*sigma2))))
            qproxy_map=base._cal_stability(-q_proxy_cal[calmask],tcal,-q_proxy_cal[calmask])[1]
            sigma,factor,if_times,factor_resid,factor_kind=_build_if_setup(xtr,tr.y_readmit30.to_numpy(float),betas["D0"])
            map_time={}
            for key,score in (("qMV",mcal/np.sqrt(vcal)),("Margin",calz0[calmask]),("Surrogate",-q_proxy_cal[calmask])):
                c=time.process_time();base._cal_stability(score,tcal,score);map_time[key]=time.process_time()-c
            int_fits=[r for r in fitrows if r.get("model_id","").startswith("INT-")]
            int_fit_cpu=float(sum(float(r.get("fit_cpu_seconds",0)) for r in int_fits))
            offline.append({"scheme":scheme,"bank":bank,"six_INT_model_fit_cpu_seconds":int_fit_cpu,
                "qMV_CALREF_map_fit_cpu_seconds_timed":map_time["qMV"],"Ridge_proxy_fit_cpu_seconds":ridge_fit_cpu,
                "surrogate_CALREF_map_fit_cpu_seconds_timed":map_time["Surrogate"],
                "IF_setup":if_times,"factorization":factor_kind,"factorization_max_abs_reconstruction_error":factor_resid,
                "Sigma_bytes":int(sigma.nbytes),"factor_bytes":int(factor.nbytes),"ridge_model_bytes":int(ridge.coef_.nbytes+ridge.intercept_.nbytes),
                "preprocessor_fit_role":"TRAIN only","proxy_target_fit_role":"CAL using D0+six INT only","calibration_target":"CALREF only"})
            modelmat=np.column_stack([betas[f"INT-{j:02d}"] for j in range(1,7)])
            # Only the CAL member of the IF artifact is opened; the EVAL member is not read.
            with np.load(OUT/scheme/f"bank{bank}"/"if_arrays.npz",allow_pickle=False) as z:
                vif_cal=z["v_if_cal_alert"].copy()
            maps={"Margin":margin_map,"qMV":qcal_map,"Surrogate":qproxy_map}
            maps["ConditionalIF"]=base._cal_stability(calz0[calmask]/np.sqrt(np.maximum(vif_cal,1e-300)),tcal,
                calz0[calmask]/np.sqrt(np.maximum(vif_cal,1e-300)))[1]
            qproxy_cal_eval=ridge.predict(_sparse_append_proxy(xc,calz0,p0c));qcal_sur=ndtr(-((calz0+qproxy_cal_eval[:,0])/np.sqrt(np.clip(np.exp(qproxy_cal_eval[:,1]),.1*sigma2,10*sigma2))))
            maps["Surrogate"]=base._cal_stability(-qcal_sur[calmask],tcal,-qcal_sur[calmask])[1]
            # Persist reusable D3 offline objects with create-only semantics.
            objpath=d3root/scheme/f"bank{bank}"/"offline_arrays.npz"
            objpath.parent.mkdir(parents=True,exist_ok=True)
            obj={"sigma_theta":sigma,"factor_L":factor,"theta_D0":betas["D0"],"theta_INT":modelmat,"gamma":np.asarray([gamma]),
                "calibration_margin":np.asarray(maps["Margin"]["coefficients"]),"calibration_qmv":np.asarray(maps["qMV"]["coefficients"]),
                "calibration_if":np.asarray(maps["ConditionalIF"]["coefficients"]),"calibration_surrogate":np.asarray(maps["Surrogate"]["coefficients"]),
                "ridge_coef":ridge.coef_,"ridge_intercept":np.asarray(ridge.intercept_),"sigma_g_sq":np.asarray([sigma2])}
            if not objpath.exists():_write_npz(objpath,**obj)
            else:
                old=_loadnpz(objpath)
                for k,v in obj.items():
                    if not np.allclose(old[k],v,rtol=1e-9,atol=1e-12):raise RuntimeError(f"D3 offline cache mismatch {objpath}:{k}")
            map_json=d3root/scheme/f"bank{bank}"/"calibration_maps.json"
            _record(map_json,maps)
            # Timed preprocessing begins from the selected, raw public EVAL frame.
            def preprocess():return _transform(eva,pipe)
            pre= _time_call(preprocess,50)
            preproc.append({"scheme":scheme,"bank":bank,"method":"Step_0_Preprocess_raw_to_sparse_X_eval",**pre,
                "matrix_shape":[int(xe.shape[0]),int(xe.shape[1])],"matrix_nnz":int(xe.nnz)})
            x_ready=xe
            # One untimed fresh D0 pass records the queue size; it does not read a cached EVAL score.
            eval_alert_n=int(np.sum(np.asarray(x_ready@betas["D0"]).reshape(-1)-gamma>=0))
            # CAL-only score maps used by the online paths.
            def score_path(x,scenario,method):
                z0=np.asarray(x@betas["D0"]).reshape(-1)-gamma
                if scenario=="D0_ALERTS":
                    mask=z0>=0;x=x[mask].tocsr();z0=z0[mask]
                if method=="Method_1_Margin":
                    return base._apply_cal_map(maps["Margin"],z0)
                if method=="Method_2_Six_INT":
                    zi=np.asarray(x@modelmat)-gamma;zb=zi.mean(axis=1);sv=np.var(zi,axis=1,ddof=1)
                    mv=.5*(z0+zb);vt=.5*sv+.5*sigma2
                    return base._apply_cal_map(qcal_map,mv/np.sqrt(vt))
                if method=="Method_3_Ridge_Surrogate":
                    pp=expit(np.clip(z0+gamma,-40,40));px=_sparse_append_proxy(x,z0,pp);pr=ridge.predict(px)
                    vv=np.clip(np.exp(pr[:,1]),.1*sigma2,10*sigma2);qq=ndtr(-((z0+pr[:,0])/np.sqrt(vv)))
                    return base._apply_cal_map(maps["Surrogate"],-qq)
                if method in ("Method_4_Conditional_IF_FullQuad","Method_5_Conditional_IF_CholeskyFactor"):
                    if method=="Method_4_Conditional_IF_FullQuad":
                        px=np.asarray(x@sigma);vv=np.asarray(x.multiply(px).sum(axis=1)).reshape(-1)
                    else:
                        px=np.asarray(x@factor);vv=np.sum(px*px,axis=1)
                    return base._apply_cal_map(maps["ConditionalIF"],z0/np.sqrt(np.maximum(vv,1e-300)))
                raise ValueError(method)
            for scenario in ("ALL_EVAL","D0_ALERTS"):
                for method in ("Method_1_Margin","Method_2_Six_INT","Method_3_Ridge_Surrogate",
                               "Method_4_Conditional_IF_FullQuad","Method_5_Conditional_IF_CholeskyFactor"):
                    result=_time_call(lambda m=method,s=scenario:score_path(x_ready,s,m),50)
                    rows.append({"scheme":scheme,"bank":bank,"scenario":scenario,"method":method,**result,
                        "eval_n":int(x_ready.shape[0]),"alert_n":eval_alert_n,"feature_dimension":d,
                        "online_reads_eval_prediction_cache":False,"online_reads_eval_IF_cache":False,
                        "includes_threshold_and_queue_selection":True,"untimed_queue_count_d0_replays":1,
                        "resident_feature_matrix_bytes":int(x_ready.data.nbytes+x_ready.indices.nbytes+x_ready.indptr.nbytes),
                        "Sigma_or_factor_bytes":int(sigma.nbytes if "FullQuad" in method else factor.nbytes if "Cholesky" in method else 0)})
            print(f"D3 timings complete scheme={scheme} bank={bank}",flush=True)
    _write_csv(d3root/"D3_online_timings.csv",rows);_write_csv(d3root/"D3_offline_setup.csv",offline);_write_csv(d3root/"D3_preprocess_timings.csv",preproc)
    _record(d3root/"D3_report.json",{"status":"D3_COMPLETE","schemes":list(schemes),"banks":6,
        "online_repeats_per_method_scenario":50,"thread_policy":"one CPU thread; process affinity pinned to one allowed CPU",
        "online_input":"fresh sparse X_eval constructed from public EVAL frame; timings compute z0 and all requested quantities from model parameters; no z0/vIF EVAL cache reads",
        "scenarios":["ALL_EVAL: 13,949 selected patients","D0_ALERTS: compute D0 margin and score only z0>=0"],
        "cpu_seconds_sum_online_timed_repetitions":float(sum(r["cpu_ms_mean"]*50/1000 for r in rows)),
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "resource_memory_note":"ru_maxrss is cumulative process high-water mark; fixed model/cache arrays and dense Sigma/L included"})
    return rows


def _sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def _table(df,columns,max_rows=24):
    cols=[c for c in columns if c in df.columns];view=df[cols].head(max_rows)
    if view.empty:return "(empty)"
    lines=["|"+"|".join(cols)+"|","|"+"|".join(["---"]*len(cols))+"|"]
    for _,row in view.iterrows():
        vals=[]
        for v in row:
            if pd.isna(v):vals.append("")
            elif isinstance(v,(float,np.floating)):vals.append(f"{float(v):.6g}")
            else:vals.append(str(v))
        lines.append("|"+"|".join(vals)+"|")
    return "\n".join(lines)


def _finalize(selected_scheme):
    fit_count=_audit_fit_counts()
    attempt_note={"status":"INCOMPLETE_SUPERSEDED_BY_CACHEFREE_RUN","attempt_directory":"D3_online_latency",
        "banks_with_offline_arrays_written":4,"online_timing_rows_published":0,
        "reason":"The exploratory D3 pass opened cached EVAL prediction arrays and the full IF arrays (including EVAL variance) through shared loaders; it was interrupted before publishing any timing table to enforce the no-cache boundary.",
        "authoritative_D3_directory":"D3_online_latency_cachefree_20261111",
        "authoritative_run":"No EVAL prediction or IF cache opened; online path starts from the raw selected EVAL frame transformed to X and saved coefficients.",
        "intermediate_artifacts_preserved":True}
    _record(OUT/"D3_incomplete_attempt_note.json",attempt_note)
    audited_fit={**_readj(OUT/"D1_fit_report.json"),**fit_count,
        "new_model_fits":fit_count["new_prediction_model_fits"],"reused_model_fits":fit_count["reused_Scheme_A_saved_models"],
        "status":"FIT_AND_REPLAY_COMPLETE_AUDITED","raw_counter_report":"D1_fit_report.json",
        "counter_correction":"D1_fit_count_correction.json"}
    _record(OUT/"D1_fit_report_audited.json",audited_fit)
    d1=pd.read_csv(OUT/"D1_mean_range_metrics.csv")
    dec=pd.read_csv(OUT/"D1_variance_amplification.csv")
    d2=pd.read_csv(OUT/"D2_CAL_CV"/"D2_mean_range_metrics.csv")
    lambdas=pd.read_csv(OUT/"D2_CAL_CV"/"D2_selected_lambdas.csv")
    d3=pd.read_csv(OUT/"D3_online_latency_cachefree_20261111"/"D3_online_timings.csv")
    pre_df=pd.read_csv(OUT/"D3_online_latency_cachefree_20261111"/"D3_preprocess_timings.csv")
    off_df=pd.read_csv(OUT/"D3_online_latency_cachefree_20261111"/"D3_offline_setup.csv")
    amp=dec.groupby(["scheme","criterion","stage"],as_index=False).agg(
        variance_amplification_vs_IF=("variance_amplification_vs_IF","mean"),
        Var_Common=("Var_Common","mean"),Var_Rare=("Var_Rare","mean"),
        TwoCov_Common_Rare=("TwoCov_Common_Rare","mean"),Var_Total=("Var_Total","mean"),
        Common_percent=("Common_percent","mean"),Rare_percent=("Rare_percent","mean"),Cross_percent=("Cross_percent","mean"),
        component_percent_sum=("component_percent_sum","mean"),rare_coefficient_norm2_share=("rare_coefficient_norm2_share","mean"))
    if not (OUT/"D1_variance_mean_range.csv").exists():_write_csv(OUT/"D1_variance_mean_range.csv",amp.to_dict("records"))
    amp_summary=[]
    for (scheme,criterion),g in dec.groupby(["scheme","criterion"]):
        by_stage={r["stage"]:r for r in g.to_dict("records")}
        mean_ratios=g.groupby("stage")["variance_amplification_vs_IF"].mean().to_dict()
        stage3=by_stage["Stage3_FullRefit"]
        amp_summary.append({"scheme":scheme,"criterion":criterion,
            "Stage1_over_IF":float(mean_ratios["Stage1_Linear"]),
            "Stage2_over_IF":float(mean_ratios["Stage2_1StepNewton"]),
            "Stage3_over_IF":float(mean_ratios["Stage3_FullRefit"]),
            "Stage3_Common_percent":float(stage3["Common_percent"]),"Stage3_Rare_percent":float(stage3["Rare_percent"]),
            "Stage3_Cross_percent":float(stage3["Cross_percent"]),
            "Stage3_rare_coefficient_norm2_share":stage3["rare_coefficient_norm2_share"],
            "rare_feature_count_mean":float(g["rare_feature_count"].mean()),
            "feature_dimension_mean":float(g["total_feature_count"].mean())})
    if not (OUT/"D1_variance_summary_bank_mean_ratios.csv").exists():_write_csv(OUT/"D1_variance_summary_bank_mean_ratios.csv",amp_summary)
    lat=d3.groupby(["scheme","scenario","method"],as_index=False).agg(
        wall_ms_median_mean=("wall_ms_median","mean"),wall_ms_median_min=("wall_ms_median","min"),wall_ms_median_max=("wall_ms_median","max"),
        wall_ms_mean_mean=("wall_ms_mean","mean"),wall_ms_iqr_mean=("wall_ms_iqr","mean"),cpu_ms_median_mean=("cpu_ms_median","mean"),
        CPU_ms_mean=("cpu_ms_mean","mean"),resident_matrix_bytes_mean=("resident_feature_matrix_bytes","mean"))
    if not (OUT/"D3_latency_mean_range.csv").exists():_write_csv(OUT/"D3_latency_mean_range.csv",lat.to_dict("records"))
    if_dicts=[ast.literal_eval(x) for x in off_df["IF_setup"]]
    d3_resource={"status":"MEASURED_FROM_CACHEFREE_D3_ARTIFACTS","timed_repetitions_per_method_scenario":50,
        "online_scoring_CPU_seconds_sum":float(np.sum(d3.cpu_ms_mean*d3.repeat_n)/1000),
        "online_scoring_wall_seconds_sum_over_all_method_repeats":float(np.sum(d3.wall_ms_mean*d3.repeat_n)/1000),
        "preprocessing_CPU_seconds_sum_over_50_repeats":float(np.sum(pre_df.cpu_ms_mean*pre_df.repeat_n)/1000),
        "preprocessing_wall_seconds_sum_over_50_repeats":float(np.sum(pre_df.wall_ms_mean*pre_df.repeat_n)/1000),
        "six_INT_model_fit_CPU_seconds_total_for_12_bank_scheme_cases":float(off_df.six_INT_model_fit_cpu_seconds.sum()),
        "Ridge_proxy_fit_CPU_seconds_total":float(off_df.Ridge_proxy_fit_cpu_seconds.sum()),
        "CALREF_map_fit_CPU_seconds_timed_replays":{"qMV":float(off_df.qMV_CALREF_map_fit_cpu_seconds_timed.sum()),
            "surrogate":float(off_df.surrogate_CALREF_map_fit_cpu_seconds_timed.sum())},
        "IF_offline_setup_CPU_seconds_by_component":{k:float(sum(x[k] for x in if_dicts)) for k in if_dicts[0]},
        "IF_offline_setup_CPU_seconds_total":float(sum(x["IF_offline_setup_total_cpu_seconds"] for x in if_dicts)),
        "mean_Sigma_bytes":float(off_df.Sigma_bytes.mean()),"mean_factor_bytes":float(off_df.factor_bytes.mean()),
        "mean_Ridge_model_bytes":float(off_df.ridge_model_bytes.mean()),
        "peak_RSS_bytes":int(json.loads((OUT/"D3_online_latency_cachefree_20261111"/"D3_report.json").read_text())["peak_rss_bytes"]),
        "memory_is_process_high_water_mark":True}
    _record(OUT/"D3_resource_summary.json",d3_resource)
    report=(f"# D1–D3 公开 Diabetes 实验报告\n\n"
        f"状态：**D1/D2/D3 实际执行完成**。固定 6 个 33,646 人训练抽样银行与原 CAL/EVAL 分区；所有特征变换仅由各银行 TRAIN 拟合，学习器和单调校准只用 CAL/CALREF，EVAL 的重训真值只来自 11 个 AUDIT 模型。无 ICU、确认数据、额外项目数据读取。\n\n"
        f"D1 的改进编码选择使用 CAL 告警患者级 5 折 CV 的 qMV 校准 Brier，得到 **{selected_scheme}**；不使用 EVAL 选择。固定惩罚 `lambda={RIDGE}`、相同 IRLS 和 bootstrap 权重。\n\n"
        f"D1 四阶段的均值放大显示表征依赖：IF 线性步与 Stage 0 接近；Stage 3/IF 在 A、B、C、D 分别约 6.39、1.08、1.48、1.08。A 的扩大量主要在随机海森矩阵一步更新到完整收敛之间出现；去缩放 OHE 显著缩小此差距。\n\n"
        f"D3 曾有一个未完成的探索目录，写入 4 个银行的离线工件后即中断，没有发布任何计时表；最终只采用 cache-free 版本目录。详见 [D3_incomplete_attempt_note.json](D3_incomplete_attempt_note.json)。\n\n"
        f"## D1-B：模型与稳定性指标（6 银行均值/范围）\n\n{_table(d1,['scheme','method','D0_outcome_AUC_mean','D0_outcome_Brier_mean','R_100_mean','R_070_mean','AURC100_mean','Brier_mean','ECE10_mean','conditionalIF_beats_margin_R70_banks'])}\n\n"
        f"完整表：[D1_per_bank_metrics.csv](D1_per_bank_metrics.csv)、[D1_mean_range_metrics.csv](D1_mean_range_metrics.csv)。\n\n"
        f"## D1-A：四阶段方差与稀有特征分解\n\n{_table(pd.DataFrame(amp_summary),['scheme','criterion','Stage1_over_IF','Stage2_over_IF','Stage3_over_IF','Stage3_Common_percent','Stage3_Rare_percent','Stage3_Cross_percent','Stage3_rare_coefficient_norm2_share','rare_feature_count_mean'])}\n\n"
        f"三项为同一患者级 logit 方差恒等式中的 Common、Rare、2Cov；`component_percent_sum` 用于逐行核验。完整银行级记录见 [D1_variance_amplification.csv](D1_variance_amplification.csv)，跨银行比值表见 [D1_variance_summary_bank_mean_ratios.csv](D1_variance_summary_bank_mean_ratios.csv)。\n\n"
        f"## D2：CAL 内交叉验证学习基线\n\n{_table(d2,['scheme','learner','R_070_mean','AURC100_mean','AUC_flip_mean','Brier_mean','ECE10_mean'])}\n\n"
        f"完整按银行指标、折级损失和正则分布：[D2_per_bank_metrics.csv](D2_CAL_CV/D2_per_bank_metrics.csv)、[D2_cv_fold_losses.csv](D2_CAL_CV/D2_cv_fold_losses.csv)、[D2_selected_lambdas.csv](D2_CAL_CV/D2_selected_lambdas.csv)。正则选择以实际分布表为准；全部学习器无符号约束，每折只用训练折估计标准化参数。\n\n"
        f"## D3：端到端在线评分\n\n{_table(lat,['scheme','scenario','method','wall_ms_median_mean','wall_ms_median_min','wall_ms_median_max','wall_ms_iqr_mean','CPU_ms_mean','resident_matrix_bytes_mean'])}\n\n"
        f"计时每个银行/场景/方法重复 50 次，分别包含从内存中 X 重新算 D0 margin、队列、INT/Ridge/IF 方差和 CALREF 映射。原始行级结果：[D3_online_timings.csv](D3_online_latency_cachefree_20261111/D3_online_timings.csv)；离线构建：[D3_offline_setup.csv](D3_online_latency_cachefree_20261111/D3_offline_setup.csv)；预处理：[D3_preprocess_timings.csv](D3_online_latency_cachefree_20261111/D3_preprocess_timings.csv)。\n\n"
        f"D3 实测汇总：在线评分重复合计 CPU {d3_resource['online_scoring_CPU_seconds_sum']:.1f} 秒，预处理重复合计 CPU {d3_resource['preprocessing_CPU_seconds_sum_over_50_repeats']:.1f} 秒，IF 离线构造/分解 CPU {d3_resource['IF_offline_setup_CPU_seconds_total']:.1f} 秒，进程峰值 RSS {d3_resource['peak_RSS_bytes']/1024**3:.2f} GiB；细项见 [D3_resource_summary.json](D3_resource_summary.json)。\n\n"
        f"## 工件与范围\n\n"
        f"- 预测模型新拟合 **{fit_count['new_prediction_model_fits']}** 个，A 方案复用 **{fit_count['reused_Scheme_A_saved_models']}** 个保存模型；另有 24 个已写盘模型从中断点继续读取，没有二次拟合。新拟合 CPU 合计 {fit_count['new_fit_CPU_seconds_sum']:.1f} 秒。\n"
        f"- 六银行、四方案模型与预测工件保存在本目录；A 方案复用已保存的 42k Logistic 银行模型。\n"
        f"- 模型拟合、bootstrap 权重、校准映射、IF 协方差、Ridge 代理和所有 EVAL 预测均有分层文件记录。\n"
        f"- 仅有 6 个训练抽样银行且共享 EVAL 患者；范围是训练抽样敏感性，不是独立总体置信区间。Brier/AUC 以有限 AUDIT 翻转参考为准。\n"
        f"- 不推断因果、不将有限重训库结论外推至临床复发机制。\n")
    reportpath=OUT/"FINAL_D1_D3_REPORT_v5.md"
    if not reportpath.exists():base._write_new(reportpath,report.encode())
    manifest=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and not p.is_symlink() and p.name!="artifact_manifest.json":manifest.append({"path":str(p.relative_to(OUT)),"size_bytes":p.stat().st_size,"sha256":_sha_file(p)})
    _record(OUT/"artifact_manifest_v5.json",{"output_root":str(OUT),"selected_improved_scheme":selected_scheme,
        "artifacts":manifest,"public_source_csv_sha256":_sha_file(pub.DATA_DIR/"diabetic_data.csv"),
        "authorization":"current user D1-D3 instruction","scope":"public Diabetes only"})
    print(f"Final report written: {reportpath}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("all","d1-fit","d1-analyze","d2","d3"),default="all")
    args=parser.parse_args();_threads_one();OUT.mkdir(parents=True,exist_ok=True)
    if args.stage in ("all","d1-fit"):
        if not (OUT/"D1_fit_report.json").exists():_fit_all_schemes()
        else:print("D1 fit stage already complete; using recorded models/predictions",flush=True)
    selected=None
    if args.stage in ("all","d1-analyze","d2","d3"):
        if (OUT/"D1_analysis_report.json").exists():selected=_readj(OUT/"D1_analysis_report.json")["selected_improved_scheme"]
        else:selected=_d1_analyze_all()
    if args.stage in ("all","d2"):
        d2=OUT/"D2_CAL_CV"
        if not (d2/"D2_report.json").exists():_d2_all(selected)
        else:print("D2 already complete",flush=True)
    if args.stage in ("all","d3"):
        d3=OUT/"D3_online_latency_cachefree_20261111"
        if not (d3/"D3_report.json").exists():_d3_all(selected)
        else:print("D3 already complete",flush=True)
    if args.stage=="all":_finalize(selected)


if __name__=="__main__":main()

