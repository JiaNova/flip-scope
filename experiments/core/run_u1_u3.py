#!/usr/bin/env python
"""Pool-A/Pool-B update experiment for U1-U3; public Diabetes only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for _thread_var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var,"1")
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from experiments.core import run_d1_d3 as d1
from experiments.core import run_w1_w4 as base
from experiments.public_benchmarks import run_public_experiments as pub

OUT=d1.OUT.parent/"U1_POOL_UPDATE_20261112"
POOL_N=21029
FIT_N=14000
K_HALF=L_HALF=7000
N_FULL=L_FULL=14000
RIDGE=1.0
SEED_BASE=2026111200
LAMBDA_GRID=(0.0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0)
OPS={"Op_Half_Replace":{"k":K_HALF,"l":L_HALF,"code":1000},
     "Op_Full_Replace":{"k":0,"l":L_FULL,"code":2000}}
ROLE_SPECS=[("INT",j) for j in range(1,7)]+[("CALREF",j) for j in range(1,7)]+[("AUDIT",j) for j in range(1,12)]
ROLES=[f"{r}-{j:02d}" for r,j in ROLE_SPECS]


def _sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def _patient_sha(ids)->str:return _sha("\n".join(map(str,ids)).encode())
def _file_sha(p):return _sha(p.read_bytes())
def _readj(p):
    if p.is_symlink() or not p.is_file():raise RuntimeError(f"unsafe/missing JSON: {p}")
    return json.loads(p.read_text())
def _writej(p,v):base._write_json(p,v)
def _writecsv(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row)) if rows else None
    raw=base._csv_bytes(rows,fields)
    if p.exists() or p.is_symlink():
        if p.is_symlink() or p.read_bytes()!=raw:raise RuntimeError(f"create-only CSV differs: {p}")
        return
    base._write_new(p,raw)
def _writenpz(p,**arrays):base._write_new(p,base._npz_bytes(**arrays))
def _loadnpz(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def _threads_one():
    if hasattr(os,"sched_getaffinity"):
        aff=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(aff)})


def _splits():
    splits,audit=d1._load_splits()
    if tuple(len(splits[k]) for k in ("TRAIN","CAL","EVAL"))!=(42058,13983,13949):raise RuntimeError("frozen Diabetes split changed")
    return splits,audit


def _pool_indices(bank,n):
    perm=np.random.default_rng(SEED_BASE+9000000+bank*1000).permutation(n)
    a=np.sort(perm[:POOL_N]);b=np.sort(perm[POOL_N:])
    if (len(a)!=POOL_N or len(b)!=POOL_N or np.intersect1d(a,b).size or
        not np.array_equal(np.sort(np.r_[a,b]),np.arange(n,dtype=np.int64))):raise RuntimeError("Pool A/B split invalid")
    return a,b


def _sample(seed,n,k):
    if int(n)!=n or int(k)!=k or n<0 or not 0<=k<=n:
        raise ValueError("SRSWOR requires integer 0 <= sample size <= population size")
    ids=np.sort(np.random.default_rng(seed).choice(n,size=k,replace=False))
    if len(ids)!=k or len(np.unique(ids))!=k:raise RuntimeError("within-model sampling is not without replacement")
    return ids


def _op_seed(bank,op,role,idx):
    rc={"CENTER":300,"INT":400,"CALREF":500,"AUDIT":600}[role]
    return int(SEED_BASE+bank*1000000+OPS[op]["code"]+rc*100+idx)


def _poola_seed(bank,role,idx):
    rc={"D0":10,"INT":20,"CALREF":30}[role]
    return int(SEED_BASE+bank*1000000+900000+rc*100+idx)


def _fit_saved(x,y,weights,model_id,bank,seed,train_ids_sha,path,fit_scope,train_n):
    weight_sha=_sha(np.asarray(weights,dtype="<f8").tobytes())
    if (path/"model.json").exists():
        row=_readj(path/"model.json");beta=np.asarray(row["beta"],float)
        if _sha(beta.astype("<f8").tobytes())!=row["beta_sha256"]:raise RuntimeError(f"model hash mismatch: {path}")
        expected=(model_id,int(bank),int(seed),train_ids_sha,weight_sha,fit_scope)
        actual=(row.get("model_id"),int(row.get("bank",-1)),int(row.get("seed",-1)),row.get("train_patient_sha256"),row.get("sample_weight_sha256"),row.get("fit_scope"))
        if actual!=expected:raise RuntimeError(f"cached fit provenance mismatch: {path}")
        return beta,row
    started_cpu=time.process_time();started_wall=time.monotonic()
    beta,info=base._fit_irls(x,np.asarray(y,np.int8),np.asarray(weights,float),ridge=RIDGE)
    wh=float(np.sum(weights));nonzero=int(np.count_nonzero(weights))
    row={"model_id":model_id,"bank":int(bank),"seed":int(seed),"train_n":int(train_n),
        "fit_weight_sum_float":wh,"nonzero_weight_rows":nonzero,"fit_cpu_seconds":time.process_time()-started_cpu,
        "fit_wall_seconds":time.monotonic()-started_wall,"train_patient_sha256":train_ids_sha,
        "sample_weight_sha256":weight_sha,
        "feature_dimension":int(len(beta)),"beta_sha256":_sha(beta.astype("<f8").tobytes()),
        "fit_scope":fit_scope,**info,"beta":beta.tolist()}
    _writej(path/"model.json",row)
    print(f"U1 fit bank={bank} model={model_id} d={len(beta)} cpu={row['fit_cpu_seconds']:.2f}s",flush=True)
    return beta,row


def _weight_for_center(nA,nB,k,l):
    return np.r_[np.full(nA,k/nA,float),np.full(nB,l/nB,float)]


def _fit_if(xtrain,ytrain,beta,xcal,xeval):
    started=time.process_time();n,d=xtrain.shape;pen=np.ones(d,float)*RIDGE;pen[0]=0.
    p=expit(np.clip(np.asarray(xtrain@beta).reshape(-1),-40,40));u=xtrain.multiply((p-ytrain)[:,None]).tocsr()
    mean_u=np.asarray(u.mean(axis=0)).reshape(-1);j=(u.T@u).toarray()-n*np.outer(mean_u,mean_u)
    h=(xtrain.T@sparse.diags(np.maximum(p*(1-p),1e-12))@xtrain).toarray();h.flat[::d+1]+=pen
    hi=np.linalg.inv(h);sigma=(hi@j@hi);sigma=(sigma+sigma.T)*.5
    vc=np.maximum(np.einsum("ij,jk,ik->i",xcal.toarray(),sigma,xcal.toarray(),optimize=True),0.)
    ve=np.maximum(np.einsum("ij,jk,ik->i",xeval.toarray(),sigma,xeval.toarray(),optimize=True),0.)
    return vc,ve,sigma,{"total_cpu_seconds":time.process_time()-started,"J_scope":"Pool-A D0 multinomial bootstrap score covariance"}


def _finite_population_if(xA,yA,xB,yB,beta,k,l,xcal,xeval):
    total0=time.process_time();sigma0=time.process_time()
    nA,d=xA.shape;nB=xB.shape[0];pen=np.ones(d,float)*RIDGE;pen[0]=0.
    pA=expit(np.clip(np.asarray(xA@beta).reshape(-1),-40,40));pB=expit(np.clip(np.asarray(xB@beta).reshape(-1),-40,40))
    uA=xA.multiply((pA-yA)[:,None]).tocsr();uB=xB.multiply((pB-yB)[:,None]).tocsr()
    mA=np.asarray(uA.mean(axis=0)).reshape(-1);mB=np.asarray(uB.mean(axis=0)).reshape(-1)
    j=_srs_score_sum_covariance(nA,k,(uA.T@uA).toarray(),mA)+_srs_score_sum_covariance(nB,l,(uB.T@uB).toarray(),mB)
    curvA=np.maximum(pA*(1-pA),1e-12)*(k/nA);curvB=np.maximum(pB*(1-pB),1e-12)*(l/nB)
    h=(xA.T@sparse.diags(curvA)@xA+xB.T@sparse.diags(curvB)@xB).toarray();h.flat[::d+1]+=pen
    hi=np.linalg.inv(h);sigma=hi@j@hi;sigma=(sigma+sigma.T)*.5
    sigma_cpu=time.process_time()-sigma0;xc=xcal.toarray();xe=xeval.toarray()
    proj0=time.process_time();vc=np.maximum(np.einsum("ij,jk,ik->i",xc,sigma,xc,optimize=True),0.)
    cal_projection_cpu=time.process_time()-proj0
    proj0=time.process_time();ve=np.maximum(np.einsum("ij,jk,ik->i",xe,sigma,xe,optimize=True),0.)
    eval_projection_cpu=time.process_time()-proj0
    return vc,ve,sigma,{"poolA_covariance_denominator":int(nA-1),"poolB_covariance_denominator":int(nB-1),
        "finite_population_formula":"k(1-k/N_A) S_A + l(1-l/N_B) S_B",
        "hessian_scale":"k/N_A times A summed Hessian + l/N_B times B summed Hessian + ridge slopes",
        "Sigma_build_cpu_seconds":float(sigma_cpu),"CAL_projection_cpu_seconds":float(cal_projection_cpu),
        "EVAL_projection_cpu_seconds_not_required_for_online_route":float(eval_projection_cpu),
        "total_cpu_seconds":float(time.process_time()-total0)}


def _srs_score_sum_covariance(n,k,score_crossproduct,score_mean):
    """Exact covariance of an SRSWOR sum at a fixed linearization point."""
    if int(n)!=n or int(k)!=k or n<1 or not 0<=k<=n:
        raise ValueError("finite-population covariance requires 0 <= k <= n and n >= 1")
    cross=np.asarray(score_crossproduct,float);mean=np.asarray(score_mean,float)
    if cross.shape!=(len(mean),len(mean)) or not np.isfinite(cross).all() or not np.isfinite(mean).all():
        raise ValueError("invalid score covariance inputs")
    if k==0 or k==n:
        return np.zeros_like(cross)
    return (k*(1-k/n)/(n-1))*(cross-n*np.outer(mean,mean))


def _fit_brier(x,y,lam):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if (x.ndim!=2 or len(x)==0 or y.shape!=(len(x),) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or np.any((y<0)|(y>1)) or not np.isfinite(lam) or lam<0):
        raise ValueError("CAL Brier fit requires finite features, probability targets and nonnegative regularization")
    sc=StandardScaler().fit(x);xs=sc.transform(x);a=np.column_stack((np.ones(len(x)),xs))
    def fn(w):
        p=expit(np.clip(a@w,-40,40));e=p-y;loss=float(np.mean(e*e)+.5*lam*np.dot(w[1:],w[1:]))
        grad=2/len(y)*(a.T@(e*p*(1-p)));grad[1:]+=lam*w[1:]
        return loss,grad
    w0=np.zeros(a.shape[1]);w0[0]=float(logit(np.clip(np.mean(y),1e-6,1-1e-6)))
    res=minimize(fn,w0,jac=True,method="L-BFGS-B",options={"maxiter":1000,"ftol":1e-13,"gtol":1e-8,"maxls":50})
    if not np.isfinite(res.x).all() or not np.isfinite(res.fun) or not np.isfinite(fn(res.x)[1]).all():
        raise RuntimeError("CAL Brier fit produced nonfinite parameters, objective or gradient")
    if not res.success and np.max(np.abs(fn(res.x)[1]))>2e-5:raise RuntimeError(f"CAL Brier fit failed: {res.message}")
    return {"scaler_mean":sc.mean_.tolist(),"scaler_scale":sc.scale_.tolist(),"intercept":float(res.x[0]),
        "coefficients":res.x[1:].tolist(),"lambda":float(lam),"gradient_inf":float(np.max(np.abs(fn(res.x)[1]))),
        "iterations":int(res.nit)}


def _predict_brier(model,x):
    x=np.asarray(x,float);xs=(x-np.asarray(model["scaler_mean"]))/np.asarray(model["scaler_scale"])
    return expit(np.clip(model["intercept"]+xs@np.asarray(model["coefficients"]),-40,40))


def _cv_brier(name,x,y,eval_features,bank,seed):
    x=np.asarray(x,float);y=np.asarray(y,float);eval_features=np.asarray(eval_features,float)
    kf=KFold(n_splits=5,shuffle=True,random_state=seed);rows=[];means={}
    for lam in LAMBDA_GRID:
        losses=[]
        for fold,(tr,te) in enumerate(kf.split(np.arange(len(y)))):
            m=_fit_brier(x[tr],y[tr],lam);pred=_predict_brier(m,x[te]);loss=float(np.mean((pred-y[te])**2));losses.append(loss)
            rows.append({"learner":name,"bank":bank,"lambda":lam,"fold":fold,"holdout_brier":loss,"train_n":len(tr),"valid_n":len(te),"patient_level":True})
        means[str(lam)]=float(np.mean(losses))
    chosen=min(LAMBDA_GRID,key=lambda v:(means[str(v)],v));fit=_fit_brier(x,y,chosen)
    fit.update({"learner":name,"bank":bank,"selected_by":"5-fold patient-level CAL Brier","cv_mean_brier_by_lambda":means,
        "feature_count":x.shape[1]})
    return fit,_predict_brier(fit,eval_features),rows


def _metrics(q,audit):
    q=np.asarray(q,float);audit=np.asarray(audit)
    if (q.ndim!=1 or len(q)==0 or audit.ndim!=2 or audit.shape[0]!=len(q) or audit.shape[1]<1
            or not np.isfinite(q).all() or np.any((q<0)|(q>1)) or not np.isin(audit,[0,1]).all()):
        raise ValueError("metrics require paired probability scores and binary AUDIT decisions")
    audit=audit.astype(np.int8);t=audit.mean(axis=1)
    auc=base._auc_finite(q,audit)
    return {"n_alerts":int(len(q)),"R_070":base._risk(q,t,.70),
        "AURC100":float(np.mean([base._risk(q,t,j/100) for j in range(1,101)])),
        "AUC":auc,"AUC_flip":auc,"Brier":float(np.mean((q[:,None]-audit)**2)),
        "ECE10":pub._ece10(q,t),"R_100":float(t.mean())}


def _cov(v,audit):
    v=np.asarray(v,float);audit=np.asarray(audit,float)
    if (v.ndim!=1 or len(v)==0 or audit.ndim!=2 or audit.shape[0]!=len(v) or audit.shape[1]<2
            or not np.isfinite(v).all() or not np.isfinite(audit).all() or np.any(v<0)):
        raise ValueError("variance diagnostic requires nonnegative predictions and paired finite logits from >=2 AUDIT models")
    s=np.var(audit,axis=1,ddof=1)
    if np.std(v)==0 or np.std(s)==0:corr=None
    else:corr=float(np.corrcoef(v,s)[0,1])
    vm,sm=float(v.mean()),float(s.mean())
    relation=("BOTH_ZERO" if vm==sm==0 else "IF_OVERDISPERSED" if vm>sm else "IF_UNDERDISPERSED" if vm<sm else "MEANS_EQUAL")
    return {"mean_IF_variance":vm,"mean_AUDIT_margin_sample_variance":sm,
        "AUDIT_over_IF_mean_ratio":sm/vm if vm>0 else None,"IF_over_AUDIT_mean_ratio":vm/sm if sm>0 else None,
        "mean_dispersion_relation":relation,"instance_correlation_IF_vs_AUDIT_variance":corr,
        "IF_variance_median":float(np.median(v)),"AUDIT_variance_median":float(np.median(s))}


def _score_map(c,e,target):
    q,m=d1._calibrated(c,target,e);return q,m


def _draw_update(xA,yA,xB,yB,bank,op,role,idx):
    k,l=OPS[op]["k"],OPS[op]["l"]
    seed=_op_seed(bank,op,role,idx)
    ia=_sample(seed+17,len(yA),k) if k else np.empty(0,dtype=np.int64)
    ib=_sample(seed+43,len(yB),l) if l else np.empty(0,dtype=np.int64)
    xx=sparse.vstack((xA[ia],xB[ib]),format="csr")
    yy=np.r_[yA[ia],yB[ib]];w=np.ones(len(yy),float)
    return xx,yy,w,ia,ib,seed


def _record(path,value):
    if path.exists() or path.is_symlink():
        old=_readj(path)
        if old!=base._json_safe(value):raise RuntimeError(f"resume record mismatch: {path}")
        return
    _writej(path,value)


def _save_npz(path,**arrays):
    if path.exists() or path.is_symlink():
        old=_loadnpz(path)
        if set(old)!=set(arrays):raise RuntimeError(f"NPZ resume keys differ: {path}")
        for k,v in arrays.items():
            if not np.array_equal(old[k],np.asarray(v)):
                raise RuntimeError(f"NPZ resume payload differs: {path}:{k}")
        return
    _writenpz(path,**arrays)


def _fit_role(xA,yA,xB,yB,bank,op,role,idx,model_root):
    xx,yy,w,ia,ib,seed=_draw_update(xA,yA,xB,yB,bank,op,role,idx)
    ids=np.r_[ia,ib]
    # Patient identity is unique in the frozen TRAIN partition. Cross-model
    # overlap is retained as repeated sampling from finite pools and audited.
    tokens=[f"PoolA_global_{int(v)}" for v in ia]+[f"PoolB_global_{int(v)}" for v in ib]
    training_sha=_patient_sha(tokens)
    mid=f"{role}-{idx:02d}"
    path=model_root/mid;path.mkdir(parents=True,exist_ok=True)
    _save_npz(path/"sampling.npz",poolA_relative=ia.astype(np.int32),poolB_relative=ib.astype(np.int32),
        seed=np.asarray([seed],np.int64),weights=w.astype(np.float64))
    beta,row=_fit_saved(xx,yy,w,mid,bank,seed,training_sha,path,"operation_without_replacement",len(yy))
    return beta,row,ia,ib


def _fit_poola_bootstrap(xD0,yD0,bank,role,idx,model_root,patient_ids):
    seed=_poola_seed(bank,role,idx)
    weights=np.random.default_rng(seed).multinomial(len(yD0),np.full(len(yD0),1/len(yD0))).astype(np.int64)
    if int(weights.sum())!=len(yD0) or np.any(weights<0):
        raise RuntimeError("invalid Pool-A multinomial bootstrap weights")
    mid=f"{role}-{idx:02d}";path=model_root/mid;path.mkdir(parents=True,exist_ok=True)
    _save_npz(path/"sampling.npz",source_D0_row=np.arange(len(yD0),dtype=np.int32),weights=weights,seed=np.asarray([seed],np.int64))
    # The row ids are the D0 Pool-A sample; multiplicities implement bootstrap.
    ids_sha=_patient_sha(list(map(str,patient_ids)))
    beta,row=_fit_saved(xD0,yD0,weights,mid,bank,seed,ids_sha,path,"Pool-A-only_multinomial_bootstrap",len(yD0))
    return beta,row,weights


def _calibrate_1d(name,cal_score,tcal,eval_score,bank,folder):
    path=folder/f"calibration_{name}.json"
    cal_score=np.asarray(cal_score,float);eval_score=np.asarray(eval_score,float);tcal=np.asarray(tcal,float)
    cal_hash=_sha(cal_score.astype("<f8").tobytes());target_hash=_sha(tcal.astype("<f8").tobytes());eval_hash=_sha(eval_score.astype("<f8").tobytes())
    if path.exists() or path.is_symlink():
        old=_readj(path)
        if (old.get("cal_score_sha256"),old.get("cal_target_sha256"),old.get("eval_score_sha256"))!=(cal_hash,target_hash,eval_hash):
            raise RuntimeError(f"CALREF mapping input provenance changed: {path}")
        return base._apply_cal_map(old["mapping"],eval_score),old
    c0=time.process_time();w0=time.monotonic();q,mapping=_score_map(cal_score,eval_score,tcal)
    payload={"learner":name,"bank":int(bank),"method":"shared two-parameter nonincreasing logistic-Brier CALREF mapping",
        "cal_score_sha256":cal_hash,"cal_target_sha256":target_hash,"eval_score_sha256":eval_hash,"mapping":mapping,
        "fit_cpu_seconds":time.process_time()-c0,"fit_wall_seconds":time.monotonic()-w0}
    _record(path,payload)
    return q,payload


def _cv_or_replay(name,xcal,tcal,xeval,bank,seed,folder):
    ehash=_sha(np.asarray(xeval,dtype="<f8").tobytes())
    cv_path=folder/f"cv_{name}.json"
    if cv_path.exists() or cv_path.is_symlink():
        saved=_readj(cv_path)
        if saved.get("xcal_sha256")!=_sha(np.asarray(xcal,dtype="<f8").tobytes()) or saved.get("xeval_sha256")!=ehash or saved.get("target_sha256")!=_sha(np.asarray(tcal,dtype="<f8").tobytes()):
            raise RuntimeError(f"CV input provenance changed: {name} bank={bank}")
        fit=saved["fit"]
        pred=_predict_brier(fit,xeval)
        return pred,fit,saved["folds"],saved["selected_lambda"]
    fit,pred,folds=_cv_brier(name,xcal,tcal,xeval,bank,seed)
    selected=float(fit["lambda"])
    payload={"learner":name,"bank":int(bank),"seed":int(seed),"xcal_sha256":_sha(np.asarray(xcal,dtype="<f8").tobytes()),"xeval_sha256":ehash,
        "target_sha256":_sha(np.asarray(tcal,dtype="<f8").tobytes()),"selected_lambda":selected,
        "fit":fit,"folds":folds,"lambda_grid":list(LAMBDA_GRID),"patient_level_cv":True}
    _record(cv_path,payload)
    return pred,fit,folds,selected


def _weighted_center(xA,yA,xB,yB,bank,op,folder):
    k,l=OPS[op]["k"],OPS[op]["l"]
    w=_weight_for_center(len(yA),len(yB),k,l)
    x=sparse.vstack((xA,xB),format="csr");y=np.r_[yA,yB]
    model_root=folder/"models";model_root.mkdir(parents=True,exist_ok=True)
    path=model_root/"OP_CENTER";path.mkdir(parents=True,exist_ok=True)
    train_sha=_patient_sha(["A:"+str(i) for i in range(len(yA))]+["B:"+str(i) for i in range(len(yB))])
    _save_npz(path/"sampling.npz",poolA_weight=np.full(len(yA),k/len(yA)),poolB_weight=np.full(len(yB),l/len(yB)))
    beta,row=_fit_saved(x,y,w,"OP_CENTER",bank,_op_seed(bank,op,"CENTER",0),train_sha,path,
        "expected_operator_center_fractional_pool_weights",len(y))
    return beta,row


def _bank_poola_stage(bank,train,cal,eva):
    bdir=OUT/f"bank{bank}";bdir.mkdir(parents=True,exist_ok=True)
    aidx,bidx=_pool_indices(bank,len(train));adf=train.iloc[aidx].reset_index(drop=True)
    pipe=d1._fit_schema(adf,d1.SCHEMES[3]);xA=d1._transform(adf,pipe)
    xcal=d1._transform(cal,pipe);xeval=d1._transform(eva,pipe)
    yA=adf.y_readmit30.to_numpy(np.int8)
    _record(bdir/"feature_metadata.json",pipe["metadata"])
    _save_npz(bdir/"pool_split.npz",poolA_global_indices=aidx.astype(np.int32),poolB_global_indices=bidx.astype(np.int32))
    split_record={"bank":bank,"scheme":d1.SCHEMES[3],"poolA_n":len(aidx),"poolB_n":len(bidx),
        "poolA_patient_sha256":_patient_sha(adf.patient_nbr.astype(str).tolist()),
        "pool_disjoint":not bool(set(aidx.tolist())&set(bidx.tolist())),"feature_schema_fit_rows":"Pool A only",
        "feature_metadata_sha256":_file_sha(bdir/"feature_metadata.json"),"split_seed":SEED_BASE+9000000+bank*1000,
        "poolB_raw_rows_materialized_by_shared_split_loader":True,
        "poolB_used_to_fit_prospective_model_or_calibration":False}
    _record(bdir/"pool_manifest.json",split_record)
    d0_ids=_sample(SEED_BASE+bank*1000000+10000,len(yA),FIT_N)
    xD0=xA[d0_ids];yD0=yA[d0_ids]
    d0dir=bdir/"poolA_only"/"models";d0dir.mkdir(parents=True,exist_ok=True)
    p=d0dir/"D0";p.mkdir(parents=True,exist_ok=True)
    _save_npz(p/"sampling.npz",poolA_relative=d0_ids.astype(np.int32),weights=np.ones(FIT_N),
        seed=np.asarray([SEED_BASE+bank*1000000+10000],np.int64))
    d0,_d0row=_fit_saved(xD0,yD0,np.ones(FIT_N),"D0",bank,SEED_BASE+bank*1000000+10000,
        _patient_sha(adf.iloc[d0_ids].patient_nbr.astype(str).tolist()),p,"Pool-A_D0_simple_random_sample",FIT_N)
    z0c=np.asarray(xcal@d0).reshape(-1);z0e=np.asarray(xeval@d0).reshape(-1)
    pc=expit(np.clip(z0c,-40,40));gamma=float(logit(np.quantile(pc,.8)))
    z0c=z0c-gamma;z0e=z0e-gamma
    ac=z0c>=0;ae=z0e>=0
    if not np.any(ac) or not np.any(ae):raise RuntimeError(f"bank {bank} has empty alert queue")
    _record(bdir/"D0_threshold.json",{"bank":bank,"tau0":float(expit(gamma)),"gamma0":gamma,
        "cal_alerts":int(ac.sum()),"eval_alerts":int(ae.sum()),"quantile":0.8,"threshold_source":"D0 CAL only"})

    # Pool-A-only bootstrap bank is completed and recorded before any Pool-B
    # feature rows are transformed or any Pool-B prediction is computed.
    aroot=bdir/"prospective";amodels=aroot/"models";amodels.mkdir(parents=True,exist_ok=True)
    int_betas=[];cal_betas=[];a_fit_rows=[]
    for role,count,dest in (("INT",6,int_betas),("CALREF",6,cal_betas)):
        for j in range(1,count+1):
            beta,row,_weights=_fit_poola_bootstrap(xD0,yD0,bank,role,j,amodels,adf.iloc[d0_ids].patient_nbr.astype(str).tolist())
            dest.append(beta);a_fit_rows.append({k:v for k,v in row.items() if k!="beta"})
    zIaC=np.column_stack([np.asarray(xcal@b).reshape(-1)-gamma for b in int_betas])[ac]
    zCaC=np.column_stack([np.asarray(xcal@b).reshape(-1)-gamma for b in cal_betas])[ac]
    tcalA=(zCaC<0).mean(axis=1)
    zbarAc=zIaC.mean(axis=1);svarAc=np.var(zIaC,axis=1,ddof=1)
    sigmaA=float(svarAc.mean());vtildeAc=.5*svarAc+.5*sigmaA
    zIaE=np.column_stack([np.asarray(xeval@b).reshape(-1)-gamma for b in int_betas])[ae]
    zbarAe=zIaE.mean(axis=1);svarAe=np.var(zIaE,axis=1,ddof=1);vtildeAe=.5*svarAe+.5*sigmaA
    v0c,v0e,sigma0,ifmeta=_fit_if(xD0,yD0,d0,xcal[ac],xeval[ae])
    prospective={}
    _qcal_margin,margin_map=_score_map(z0c[ac],z0e[ae],tcalA)
    qmargin=base._apply_cal_map(margin_map,z0e[ae])
    _record(aroot/"calibration_Prospective_Margin_D0.json",{"learner":"Prospective_Margin_D0","bank":bank,
        "cal_target_sha256":_sha(tcalA.astype("<f8").tobytes()),"mapping":margin_map})
    prospective["Prospective_Margin_D0"]=(qmargin,"one-dimensional CALREF map")
    q,mp=_calibrate_1d("Prospective_D0_IF_1D",z0c[ac]/np.sqrt(np.maximum(v0c,1e-300)),tcalA,
        z0e[ae]/np.sqrt(np.maximum(v0e,1e-300)),bank,aroot)
    prospective["Prospective_D0_IF_1D"]=(q,"one-dimensional CALREF map")
    xci=np.column_stack((z0c[ac],np.log(np.maximum(v0c,1e-300)),z0c[ac]*np.log(np.maximum(v0c,1e-300)),z0c[ac]/np.sqrt(np.maximum(v0c,1e-300))))
    xei=np.column_stack((z0e[ae],np.log(np.maximum(v0e,1e-300)),z0e[ae]*np.log(np.maximum(v0e,1e-300)),z0e[ae]/np.sqrt(np.maximum(v0e,1e-300))))
    q,fit,folds,chosen=_cv_or_replay("Prospective_D0_IF_Interaction_CV",xci,tcalA,xei,bank,SEED_BASE+bank*100+7,aroot)
    prospective["Prospective_D0_IF_Interaction_CV"]=(q,"CAL-only patient-level 5-fold CV Brier")
    xcm=np.column_stack((z0c[ac],zbarAc,np.log(np.maximum(vtildeAc,1e-300))))
    xem=np.column_stack((z0e[ae],zbarAe,np.log(np.maximum(vtildeAe,1e-300))))
    q,fitm,foldsm,chosenm=_cv_or_replay("Prospective_PoolA_INT6_Tuned3D",xcm,tcalA,xem,bank,SEED_BASE+bank*100+8,aroot)
    prospective["Prospective_PoolA_INT6_Tuned3D"]=(q,"CAL-only patient-level 5-fold CV Brier")
    calmap=base._json_safe({"margin_map":margin_map,"poolA_sigma2":sigmaA,"poolA_IF":ifmeta,
        "v0_cal_mean":float(v0c.mean()),"v0_eval_alert_mean":float(v0e.mean()),
        "cv_if_lambda":chosen,"cv_poolA_int_lambda":chosenm,"cv_if_fold_count":len(folds),"cv_int_fold_count":len(foldsm),
        "poolA_int_model_hashes":[_sha(b.astype("<f8").tobytes()) for b in int_betas],
        "poolA_calref_model_hashes":[_sha(b.astype("<f8").tobytes()) for b in cal_betas],
        "cal_target_sha256":_sha(tcalA.astype("<f8").tobytes()),"cal_alert_count":len(tcalA),
        "poolB_raw_rows_materialized_by_shared_split_loader":True,
        "poolB_used_to_fit_prospective_model_or_calibration":False})
    _record(aroot/"prospective_ready.json",calmap)
    _record(aroot/"selected_lambdas.json",[
        {"bank":bank,"track":"II_Prospective_PoolA_Only","operation":"both future operations","learner":"Prospective_D0_IF_Interaction_CV",
         "selected_lambda":chosen,"cv_mean_brier":min(fit["cv_mean_brier_by_lambda"].values()),"feature_count":fit["feature_count"],
         "coefficients":fit["coefficients"],"intercept":fit["intercept"],"cv_lambda_scores":fit["cv_mean_brier_by_lambda"]},
        {"bank":bank,"track":"II_Prospective_PoolA_Only","operation":"both future operations","learner":"Prospective_PoolA_INT6_Tuned3D",
         "selected_lambda":chosenm,"cv_mean_brier":min(fitm["cv_mean_brier_by_lambda"].values()),"feature_count":fitm["feature_count"],
         "coefficients":fitm["coefficients"],"intercept":fitm["intercept"],"cv_lambda_scores":fitm["cv_mean_brier_by_lambda"]}])
    _save_npz(aroot/"prospective_scores.npz",eval_alert_mask=ae.astype(np.uint8),
        **{f"score_{k}":v[0] for k,v in prospective.items()},v0_cal=v0c,v0_eval_alert=v0e,
        z0_cal_alert=z0c[ac],z0_eval_alert=z0e[ae],tcal_poolA=tcalA)
    return {"bank_dir":bdir,"aidx":aidx,"bidx":bidx,"adf":adf,"pipe":pipe,"xA":xA,"xcal":xcal,"xeval":xeval,
        "yA":yA,"d0_ids":d0_ids,"xD0":xD0,"yD0":yD0,"d0":d0,"d0_patient_ids":adf.iloc[d0_ids].patient_nbr.astype(str).tolist(),"gamma":gamma,"z0c":z0c,"z0e":z0e,
        "ac":ac,"ae":ae,"prospective":prospective,"sigmaA":sigmaA,"v0c":v0c,"v0e":v0e,
        "prospective_fits":a_fit_rows,"bank":bank}


def _latency_summary(samples):
    x=np.asarray(samples,float)
    return {"repetitions":int(len(x)),"mean_wall_ms":float(x[:,0].mean()),"median_wall_ms":float(np.median(x[:,0])),
        "wall_iqr_ms":float(np.quantile(x[:,0],.75)-np.quantile(x[:,0],.25)),
        "mean_cpu_ms":float(x[:,1].mean()),"median_cpu_ms":float(np.median(x[:,1])),
        "cpu_iqr_ms":float(np.quantile(x[:,1],.75)-np.quantile(x[:,1],.25))}


def _time_callable(fn,n_scored,repeats=50,n_input=None):
    if repeats<1 or n_scored is None or n_scored<1 or (n_input is not None and n_input<n_scored):
        raise ValueError("timing requires positive repetitions and 0 < scored rows <= input rows")
    fn()  # warm-up is explicitly excluded from the 50 timed invocations
    out=[]
    for _ in range(repeats):
        c0=time.process_time();w0=time.perf_counter();fn();wall=time.perf_counter()-w0;cpu=time.process_time()-c0
        out.append((1000*wall,1000*cpu))
    row=_latency_summary(out);row["n_scored"]=(int(n_scored) if n_scored is not None else None)
    row["n_input"]=(int(n_input) if n_input is not None else int(n_scored) if n_scored is not None else None)
    row["mean_wall_ms_per_scored_row"]=(row["mean_wall_ms"]/n_scored if n_scored else None)
    row["mean_cpu_ms_per_scored_row"]=(row["mean_cpu_ms"]/n_scored if n_scored else None)
    row["mean_wall_ms_per_input_row"]=(row["mean_wall_ms"]/row["n_input"] if row["n_input"] else None)
    row["mean_cpu_ms_per_input_row"]=(row["mean_cpu_ms"]/row["n_input"] if row["n_input"] else None)
    return row


def _cost_crossing(setup_a,setup_b,cpu_ms_a,cpu_ms_b):
    values=np.asarray([setup_a,setup_b,cpu_ms_a,cpu_ms_b],float)
    if not np.isfinite(values).all() or np.any(values<0):
        raise ValueError("costs must be finite and nonnegative")
    den=cpu_ms_a-cpu_ms_b
    if den==0:
        return None,"COINCIDENT" if setup_a==setup_b else "PARALLEL_NO_CROSSING"
    n=(setup_b-setup_a)*1000/den
    return (float(n),"NONNEGATIVE_CROSSING") if n>=0 else (None,"NO_NONNEGATIVE_CROSSING")


def _operation(bank_state,op,train):
    bank=int(bank_state["bank_dir"].name.replace("bank",""));folder=bank_state["bank_dir"]/op;folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/"operation_results.json"
    if result_path.exists() or result_path.is_symlink():
        result=_readj(result_path)
        folds=[]
        for cvp in sorted(folder.glob("cv_*.json")):
            folds.extend(_readj(cvp).get("folds",[]))
        return (result["track_i_metrics"]+result["track_ii_metrics"],folds,result["selected_lambdas"],
            result["variance_diagnostics"],result["online_latency"],result["setup_costs"],result["cost_rows"],result["cost_crossovers"])
    if not (bank_state["bank_dir"]/"prospective"/"prospective_ready.json").is_file():
        raise RuntimeError("Pool-A-only stage is not frozen before Pool-B access")
    bidx=bank_state["bidx"]
    bdf=train.iloc[bidx].reset_index(drop=True);xB=d1._transform(bdf,bank_state["pipe"]);yB=bdf.y_readmit30.to_numpy(np.int8)
    _record(bank_state["bank_dir"]/"post_poolB_access.json",{"bank":bank,"poolB_patient_sha256":_patient_sha(bdf.patient_nbr.astype(str).tolist()),
        "poolB_transform_schema_sha256":_file_sha(bank_state["bank_dir"]/"feature_metadata.json"),
        "poolB_access_after_prospective_ready":True,"poolB_used_for_prospective_calibration":False})
    xA=bank_state["xA"];yA=bank_state["yA"];xcal=bank_state["xcal"];xeval=bank_state["xeval"]
    gamma=bank_state["gamma"];ac=bank_state["ac"];ae=bank_state["ae"]
    z0c=bank_state["z0c"];z0e=bank_state["z0e"]
    center_cpu0=time.process_time()
    betaR,center_row=_weighted_center(xA,yA,xB,yB,bank,op,folder)
    center_cpu=time.process_time()-center_cpu0
    zRc=np.asarray(xcal@betaR).reshape(-1)-gamma;zRe=np.asarray(xeval@betaR).reshape(-1)-gamma
    k,l=OPS[op]["k"],OPS[op]["l"]
    vRc,vRe,sigmaR,ifmeta=_finite_population_if(xA,yA,xB,yB,betaR,k,l,xcal[ac],xeval[ae])
    if_cpu=float(ifmeta["Sigma_build_cpu_seconds"]+ifmeta["CAL_projection_cpu_seconds"])
    npat=folder/"if_operator.npz"
    _save_npz(npat,sigma_theta=sigmaR,v_cal_alert=vRc,v_eval_alert=vRe)

    modelroot=folder/"models";modelroot.mkdir(parents=True,exist_ok=True)
    betas={};fit_rows=[];draws={}
    for role,j in ROLE_SPECS:
        beta,row,ia,ib=_fit_role(xA,yA,xB,yB,bank,op,role,j,modelroot)
        betas[f"{role}-{j:02d}"]=beta;fit_rows.append({k0:v0 for k0,v0 in row.items() if k0!="beta"})
        draws[f"{role}_{j:02d}_A"]=ia.astype(np.int32);draws[f"{role}_{j:02d}_B"]=ib.astype(np.int32)
    hashes={m:_sha(b.astype("<f8").tobytes()) for m,b in betas.items()}
    unique_outputs=len(set(hashes.values()))
    role_sets=[]
    for role,j in ROLE_SPECS:
        role_sets.append(set(("A",int(v)) for v in draws[f"{role}_{j:02d}_A"]) |
                         set(("B",int(v)) for v in draws[f"{role}_{j:02d}_B"]))
    overlap=[len(role_sets[i]&role_sets[j]) for i in range(len(role_sets)) for j in range(i+1,len(role_sets))]
    _save_npz(folder/"role_draw_indices.npz",**draws)
    _record(folder/"model_inventory.json",{"bank":bank,"operation":op,"models":fit_rows,"role_beta_sha256":hashes,
        "unique_role_beta_hashes":unique_outputs,"role_models":23,"center_beta_sha256":_sha(betaR.astype("<f8").tobytes()),
        "distinct_coefficients_do_not_prove_random_draw_independence":True,
        "cross_model_patient_overlap_expected":True,"sampling":"separate SRSWOR per model and source pool; overlap across draws is possible",
        "cross_model_pairs":len(overlap),"cross_model_pairs_with_shared_patients":int(sum(x>0 for x in overlap)),
        "cross_model_shared_patient_mean":float(np.mean(overlap)),"cross_model_shared_patient_max":int(max(overlap,default=0))})
    int_betas=[betas[f"INT-{j:02d}"] for j in range(1,7)]
    cal_betas=[betas[f"CALREF-{j:02d}"] for j in range(1,7)]
    audit_betas=[betas[f"AUDIT-{j:02d}"] for j in range(1,12)]
    ziC=np.column_stack([np.asarray(xcal@b).reshape(-1)-gamma for b in int_betas])[ac]
    ziE=np.column_stack([np.asarray(xeval@b).reshape(-1)-gamma for b in int_betas])[ae]
    zcC=np.column_stack([np.asarray(xcal@b).reshape(-1)-gamma for b in cal_betas])[ac]
    zaE=np.column_stack([np.asarray(xeval@b).reshape(-1)-gamma for b in audit_betas])[ae]
    audit=(zaE<0).astype(np.int8);tcal=(zcC<0).mean(axis=1)
    if ziC.shape[0]!=len(tcal) or ziE.shape[0]!=audit.shape[0]:raise RuntimeError("CAL/EVAL alert alignment error")
    zbarC=ziC.mean(axis=1);zbarE=ziE.mean(axis=1)
    svarC=np.var(ziC,axis=1,ddof=1);svarE=np.var(ziE,axis=1,ddof=1)
    sigma_g2=float(svarC.mean());vtildeC=.5*svarC+.5*sigma_g2;vtildeE=.5*svarE+.5*sigma_g2
    mC=.5*(z0c[ac]+zbarC);mE=.5*(z0e[ae]+zbarE)
    cal_features={
        "OpCenter_Tuned2D_Mean":(np.column_stack((z0c[ac],zRc[ac])),np.column_stack((z0e[ae],zRe[ae]))),
        "OpCenter_IF_Interaction_CV":(np.column_stack((z0c[ac],zRc[ac],np.log(np.maximum(vRc,1e-300)),zRc[ac]*np.log(np.maximum(vRc,1e-300)),zRc[ac]/np.sqrt(np.maximum(vRc,1e-300)))),
            np.column_stack((z0e[ae],zRe[ae],np.log(np.maximum(vRe,1e-300)),zRe[ae]*np.log(np.maximum(vRe,1e-300)),zRe[ae]/np.sqrt(np.maximum(vRe,1e-300))))),
        "Update_INT6_Tuned2D_Mean":(np.column_stack((z0c[ac],zbarC)),np.column_stack((z0e[ae],zbarE))),
        "Update_INT6_Tuned3D_MV":(np.column_stack((z0c[ac],zbarC,np.log(np.maximum(vtildeC,1e-300)))),np.column_stack((z0e[ae],zbarE,np.log(np.maximum(vtildeE,1e-300))))) }
    scored={};calibrations={};cv_rows=[];selected=[]
    # All one-dimensional mappings use only the relevant six CALREF models.
    raw1={"Margin_D0_1D":(z0c[ac],z0e[ae]),"OpCenter_Mean_1D":(zRc[ac],zRe[ae]),
        "OpCenter_IF_1D":(zRc[ac]/np.sqrt(np.maximum(vRc,1e-300)),zRe[ae]/np.sqrt(np.maximum(vRe,1e-300))),
        "Update_INT6_qMV":(mC/np.sqrt(np.maximum(vtildeC,1e-300)),mE/np.sqrt(np.maximum(vtildeE,1e-300)))}
    for name,(cs,es) in raw1.items():
        q,payload=_calibrate_1d(name,cs,tcal,es,bank,folder);scored[name]=q;calibrations[name]=payload
    cv_seed=SEED_BASE+bank*10000+OPS[op]["code"]
    for name,(xc,xe) in cal_features.items():
        q,fit,folds,lam=_cv_or_replay(name,xc,tcal,xe,bank,cv_seed+len(cv_rows)+1,folder)
        scored[name]=q;cv_rows.extend(folds)
        selected.append({"bank":bank,"operation":op,"learner":name,"selected_lambda":lam,
            "cv_mean_brier":min(fit["cv_mean_brier_by_lambda"].values()),"feature_count":fit["feature_count"],"coefficients":fit["coefficients"],
            "intercept":fit["intercept"],"cv_lambda_scores":fit["cv_mean_brier_by_lambda"]})
    metrics=[]
    for name,q in scored.items():
        metrics.append({"bank":bank,"operation":op,"track":"I_Update_Pool_Aware","method":name,**_metrics(q,audit)})

    # Pool-A-only methods were frozen before xB was transformed. Reuse their
    # stored predictions, and score them only now against this operation's AUDIT.
    for name,(q,_scope) in bank_state["prospective"].items():
        metrics.append({"bank":bank,"operation":op,"track":"II_Prospective_PoolA_Only","method":name,**_metrics(q,audit)})

    # Direct variance closure against the independent 11-model AUDIT outcomes.
    ecalif=_cov(vRe,zaE);ecalif.update({"bank":bank,"operation":op,"variance_source":"OpCenter_IF_vR"})
    e0if=_cov(bank_state["v0e"],zaE);e0if.update({"bank":bank,"operation":op,"variance_source":"Prospective_D0_IF_v0"})

    # U3 setup and cache-free scoring from transformed features (not raw-data
    # end-to-end latency). The only source matrix
    # is the freshly transformed EVAL matrix; z0/v arrays are recomputed inside.
    map_mean=calibrations["OpCenter_Mean_1D"]["mapping"]
    map_if=calibrations["OpCenter_IF_1D"]["mapping"]
    map_int=calibrations["Update_INT6_qMV"]["mapping"]
    nfull=xeval.shape[0];nalert=int(ae.sum());scoring={}
    for cohort in ("all_eval", "d0_alert_only"):
        use_alert=cohort=="d0_alert_only";n=nalert if use_alert else nfull
        def mean_fn():
            if use_alert:
                mask=(np.asarray(xeval@bank_state["d0"]).reshape(-1)-gamma)>=0
                xx=xeval[mask]
            else:xx=xeval
            zz=np.asarray(xx@betaR).reshape(-1)-gamma
            return base._apply_cal_map(map_mean,zz)
        def if_fn():
            if use_alert:
                mask=(np.asarray(xeval@bank_state["d0"]).reshape(-1)-gamma)>=0;xx=xeval[mask]
            else:xx=xeval
            zz=np.asarray(xx@betaR).reshape(-1)-gamma
            xd=xx.toarray();v=np.maximum(np.einsum("ij,jk,ik->i",xd,sigmaR,xd,optimize=True),1e-300)
            return base._apply_cal_map(map_if,zz/np.sqrt(v))
        def int_fn():
            z0full=np.asarray(xeval@bank_state["d0"]).reshape(-1)-gamma
            if use_alert:mask=z0full>=0;xx=xeval[mask];z0=z0full[mask]
            else:xx=xeval;z0=z0full
            zz=np.column_stack([np.asarray(xx@b).reshape(-1)-gamma for b in int_betas])
            zb=zz.mean(axis=1);sv=np.var(zz,axis=1,ddof=1);vv=.5*sv+.5*sigma_g2
            mm=.5*(z0+zb)
            return base._apply_cal_map(map_int,mm/np.sqrt(np.maximum(vv,1e-300)))
        scoring[cohort]={"n_rows":n,"input_rows":nfull,"OpCenter_Mean":_time_callable(mean_fn,n,n_input=nfull),"OpCenter_IF":_time_callable(if_fn,n,n_input=nfull),
            "Update_INT6":_time_callable(int_fn,n,n_input=nfull),"includes_D0_alert_filter":use_alert,"repetitions":50,
            "raw_preprocessing_included":False,"probability_validation_scope":"D0-positive alert rows only",
            "all_eval_D0_pass_included":{"OpCenter_Mean":False,"OpCenter_IF":False,"Update_INT6":True}}
    inventory=_readj(folder/"model_inventory.json")
    cpu_center=float(center_row.get("fit_cpu_seconds",center_cpu))
    map_mean_cpu=float(calibrations["OpCenter_Mean_1D"].get("fit_cpu_seconds",0.0))
    map_if_cpu=float(calibrations["OpCenter_IF_1D"].get("fit_cpu_seconds",0.0))
    map_int_cpu=float(calibrations["Update_INT6_qMV"].get("fit_cpu_seconds",0.0))
    cpu_int=sum(float(inventory["models"][j-1]["fit_cpu_seconds"]) for j in range(1,7))
    setups={"OpCenter_Mean":{"setup_cpu_seconds":cpu_center+map_mean_cpu,"components":"one expected-operator center IRLS fit + 1D CALREF mapping"},
        "OpCenter_IF":{"setup_cpu_seconds":cpu_center+if_cpu+map_if_cpu,"components":"operator center IRLS + finite-population Sigma/CAL projection + 1D CALREF mapping"},
        "Update_INT6":{"setup_cpu_seconds":cpu_int+map_int_cpu,"components":"six independent update INT IRLS fits + 1D CALREF mapping"}}
    for values in setups.values():
        values.update(shared_CALREF_model_fit_cost_included=False,raw_preprocessing_cost_included=False,
                      CV_search_fit_cost_included=False,comparison_scope="specified 1D-calibrated scoring routes only")
    for route in scoring["all_eval"]:
        if route in setups:
            setups[route]["online_all_eval_wall_ms_per_input_row"]=scoring["all_eval"][route]["mean_wall_ms_per_input_row"]
            setups[route]["online_all_eval_cpu_ms_per_input_row"]=scoring["all_eval"][route]["mean_cpu_ms_per_input_row"]
            setups[route]["online_d0_alert_wall_ms_per_input_row"]=scoring["d0_alert_only"][route]["mean_wall_ms_per_input_row"]
            setups[route]["online_d0_alert_cpu_ms_per_input_row"]=scoring["d0_alert_only"][route]["mean_cpu_ms_per_input_row"]
    cost_rows=[];cross_rows=[]
    for cohort in ("all_eval","d0_alert_only"):
        routes=list(setups)
        for rname in routes:
            per= scoring[cohort][rname]["mean_cpu_ms_per_input_row"]
            for nrows in (5000,20000,100000):
                cost_rows.append({"bank":bank,"operation":op,"cohort":cohort,"route":rname,"N":nrows,
                    "setup_cpu_seconds":setups[rname]["setup_cpu_seconds"],"c_online_cpu_ms_per_row":per,
                    "cycle_cpu_seconds_approx":setups[rname]["setup_cpu_seconds"]+per*nrows/1000,
                    "cycle_model":"setup CPU seconds + N * mean online CPU ms/row / 1000"})
        for i,ra in enumerate(routes):
            for rb in routes[i+1:]:
                sa=setups[ra]["setup_cpu_seconds"];sb=setups[rb]["setup_cpu_seconds"]
                ca=scoring[cohort][ra]["mean_cpu_ms_per_input_row"];cb=scoring[cohort][rb]["mean_cpu_ms_per_input_row"]
                nstar,relation=_cost_crossing(sa,sb,ca,cb)
                cross_rows.append({"bank":bank,"operation":op,"cohort":cohort,"route_A":ra,"route_B":rb,
                    "N_star_rows_A_equals_B":nstar,"cost_relation":relation,
                    "interpretation":"crossing of setup + N * CPU milliseconds per incoming input / 1000; either slope ordering is allowed"})
    _save_npz(folder/"prediction_arrays.npz",audit_eval_alert=audit,z0_cal_alert=z0c[ac],z0_eval_alert=z0e[ae],
        zR_cal_alert=zRc[ac],zR_eval_alert=zRe[ae],zINT_cal_alert=ziC,zINT_eval_alert=ziE,
        zCALREF_cal_alert=zcC,zAUDIT_eval_alert=zaE,vR_cal_alert=vRc,vR_eval_alert=vRe,
        vtilde_cal_alert=vtildeC,vtilde_eval_alert=vtildeE,tcal_update=tcal,
        **{f"q_{name}":q for name,q in scored.items()})
    _record(folder/"operation_results.json",{"bank":bank,"operation":op,"status":"COMPLETE",
        "n_alerts_eval":int(audit.shape[0]),"n_alerts_cal":int(len(tcal)),"sigma_g2_int_cal":sigma_g2,
        "center_fit_cpu_seconds":float(center_row["fit_cpu_seconds"]),"if_sigma_cpu_seconds":if_cpu,
        "finite_population_if":ifmeta,"role_model_count":23,"unique_role_coeff_hashes":unique_outputs,
        "track_i_metrics":metrics[:len(scored)],"track_ii_metrics":metrics[len(scored):],
        "variance_diagnostics":[ecalif,e0if],"selected_lambdas":selected,"online_latency":scoring,
        "setup_costs":setups,"cost_rows":cost_rows,"cost_crossovers":cross_rows,
        "calref_target_sha256":_sha(tcal.astype("<f8").tobytes()),"audit_outcome_sha256":_sha(audit.astype("u1").tobytes()),
        "poolB_used_for_track_ii_calibration":False})
    return metrics,cv_rows,selected,[ecalif,e0if],scoring,setups,cost_rows,cross_rows


def _summarize(rows,groups,values):
    out=[]
    keys=sorted({tuple(r[k] for k in groups) for r in rows},key=lambda x:tuple(map(str,x)))
    for key in keys:
        vals=[r for r in rows if tuple(r[k] for k in groups)==key]
        row=dict(zip(groups,key))
        for col in values:
            xs=[float(v[col]) for v in vals if v.get(col) is not None and np.isfinite(float(v[col]))]
            row[col+"_mean"]=float(np.mean(xs)) if xs else None
            row[col+"_min"]=float(np.min(xs)) if xs else None
            row[col+"_max"]=float(np.max(xs)) if xs else None
            row[col+"_median"]=float(np.median(xs)) if xs else None
        row["banks_n"]=len(vals)
        out.append(row)
    return out


def _win_counts(rows):
    metrics=("R_070","AURC100","AUC","Brier","ECE10")
    out=[]
    keys=sorted({(r["track"],r["operation"],r["method"]) for r in rows})
    for track,op,method in keys:
        if method in ("Margin_D0_1D","Prospective_Margin_D0"):continue
        items=[r for r in rows if (r["track"],r["operation"],r["method"])==(track,op,method)]
        row={"track":track,"operation":op,"method":method,"banks":len(items)}
        for baseline,label in (("Margin_D0_1D","margin"),("Prospective_Margin_D0","margin"),("OpCenter_Mean_1D","operator_mean")):
            base_items={r["bank"]:r for r in rows if r["track"]==track and r["operation"]==op and r["method"]==baseline}
            if not base_items:continue
            for metric in metrics:
                cmp=[]
                for item in items:
                    ref=base_items.get(item["bank"])
                    if ref is None or item.get(metric) is None or ref.get(metric) is None:continue
                    cmp.append(float(item[metric])>float(ref[metric]) if metric=="AUC" else float(item[metric])<float(ref[metric]))
                row[f"wins_vs_{label}_{metric}"]=int(sum(cmp));row[f"ncomp_vs_{label}_{metric}"]=len(cmp)
        out.append(row)
    return out


def _markdown_report(report,mean_rows,diag_rows,cost_rows):
    def table(rows,cols):
        lines=["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]
        for row in rows:
            lines.append("| "+" | ".join("" if row.get(c) is None else str(row.get(c)) for c in cols)+" |")
        return "\n".join(lines)
    lines=["# Diabetes Scheme D: Pool Update U1–U3","",
        f"Status: **{report['status']}**","",
        "This is a public UCI Diabetes analysis on the frozen patient-level TRAIN/CAL/EVAL partition. No ICU or confirmation input was read. EVAL outcome labels were not used; AUDIT-only finite model flips are the evaluation target.","",
        "## Protocol and role separation","",
        f"- Each of 6 banks partitions TRAIN into disjoint Pool A/B of 21,029 patients each; D0 uses a 14,000-patient SRSWOR draw from A. Scheme D feature merging/scaling is fit on Pool A only.",
        "- Track II CALREF/bootstrap models and CV are fit from Pool A only and the `prospective_ready.json` checkpoint is persisted before Pool B rows are transformed. Pool B does not enter Track II scores or calibration.",
        "- Each operation uses separate 7k+7k or 14k Pool-B SRSWOR draws for 6 INT, 6 CALREF, and 11 AUDIT. Sampling is without replacement within each model; patient overlap across the 23 models is expected and recorded because 23 × 14k mutually disjoint draws cannot fit within these finite source pools.",
        "- Model roles and draws are separate; distinct coefficient hashes are descriptive and do not prove independence. CALREF targets are used only on CAL; AUDIT labels are used only on the EVAL D0 alert queue.",
        "- Ranges across banks describe training-pool split sensitivity over one shared CAL/EVAL partition; they are not independent-population confidence intervals.","",
        "## U1 mean and range across six banks","",
        table(mean_rows,["track","operation","method","R_070_mean","R_070_min","R_070_max","AURC100_mean","AUC_mean","Brier_mean","ECE10_mean","wins_vs_margin_R_070"]),"",
        "Full per-bank values, all requested metrics, and metric-specific win counts are in `U1_per_bank_metrics.csv`, `U1_mean_range_metrics.csv`, and `U1_win_counts.csv`.","",
        "## U2 variance closure","",
        table(diag_rows,["operation","variance_source","mean_IF_variance_mean","mean_AUDIT_margin_sample_variance_mean","AUDIT_over_IF_mean_ratio_mean","instance_correlation_IF_vs_AUDIT_variance_mean"]),"",
        "The comparison is descriptive with 11 AUDIT models per row. It does not establish asymptotic covariance validity beyond this finite public-data setup.","",
        "## U3 online latency and cycle cost","",
        table(cost_rows,["operation","cohort","route","N","cycle_cpu_seconds_approx_mean"]),"",
        "Each online latency is 50 timed repetitions after one excluded warm-up, starting from a transformed EVAL design matrix; raw preprocessing is excluded. Alert-only scoring recomputes D0 margins and the alert mask inside each timed call. All-row scoring is a computational scenario, not validation of non-alert flip probabilities. Cycle costs are partial CPU estimates; shared CALREF fitting and CV-search costs are not included.","",
        "## Resource and execution audit","",
        f"- Cumulative model fit artifacts: {report['model_fit_artifact_count']} (expected {report['expected_model_fit_artifacts']}); coefficient hashes verified: {report['unique_operation_role_hashes']}/{report['expected_operation_role_hashes']}.",
        f"- Cumulative fit CPU seconds: {report['fit_cpu_seconds']:.3f}; task wall seconds: {report['elapsed_wall_seconds']:.3f}; peak RSS: {report['peak_rss_bytes']} bytes.",
        f"- Source Diabetes CSV SHA-256: `{report['diabetes_csv_sha256']}`.",
        "- U3 latency runs score feature matrices only; no prediction cache was read. `q` outputs use the saved CALREF map for each route.","",
        "## Files","",
        "See `U1_per_bank_metrics.csv`, `U1_mean_range_metrics.csv`, `U1_win_counts.csv`, `U1_selected_lambdas.csv`, `U1_existing_bootstrap_context.csv`, `U2_variance_diagnostics.csv`, `U3_online_cost_by_bank.csv`, `U3_cost_crossovers.csv`, and `U3_cost_mean_range.csv`. Per-bank model fits, input draws, variance matrices, and checkpoints sit under `bank0`–`bank5`. The historical same-pool Scheme D Bootstrap table is context only because its D0 training size differs.","",
        "No causal claim is made. The experiment estimates finite AUDIT-model flip behavior for the stated sampling and update operators.",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true",help="resume only from matching create-only checkpoints")
    args=parser.parse_args()
    if OUT.is_symlink():raise RuntimeError(f"output root is a symlink: {OUT}")
    existed=OUT.exists()
    OUT.mkdir(parents=True,exist_ok=True)
    final=OUT/"U1_U3_FINAL_REPORT.json"
    if final.exists() or final.is_symlink():
        if final.is_symlink():raise RuntimeError("final report path is a symlink")
        reviewed=OUT/"review/review_summary.json"
        print(reviewed.read_text() if reviewed.is_file() else final.read_text());return
    if existed and any(OUT.iterdir()) and not args.resume:
        raise RuntimeError("output directory has incomplete checkpoints; pass --resume to validate and continue")
    from threadpoolctl import threadpool_limits
    thread_limit=threadpool_limits(limits=1)
    _threads_one()
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[name]="1"
    protocol={"authorization":"current user U1-U3 task message; public Diabetes only",
        "status":"EXECUTION_FIXED","seed_base":SEED_BASE,"banks":6,"pool_sizes":{"A":POOL_N,"B":POOL_N},
        "pool_split":"independent deterministic TRAIN permutation per bank; first 21,029 to A, remainder to B",
        "D0":{"sample_n":FIT_N,"source":"Pool A","sampling":"SRS without replacement"},
        "operations":OPS,"role_models_per_operation":{"INT":6,"CALREF":6,"AUDIT":11},
        "operation_name_note":"Half refers to 7000 samples from each source pool, not retention of half the D0 training patients; each A subset is sampled independently of the D0 subset",
        "role_model_sampling":"separate model-role SRSWOR draws within shared fixed pools; cross-model overlap and duplicate outputs recorded; not a requirement for disjoint patient sets",
        "scheme":"Scheme_D_Merged_Rare50_Unscaled","feature_fit_scope":"Pool A only; held fixed for A/B/CAL/EVAL",
        "ridge_lambda":RIDGE,"solver":"sum binary log loss + 0.5*||beta_slopes||^2; exact damped Newton/IRLS; intercept unpenalized",
        "operator_center":"k/N_A times sum Pool A loss + l/N_B times sum Pool B loss; total fit weight 14,000",
        "finite_population_IF":"J=k(1-k/N_A)S_A+l(1-l/N_B)S_B; H weighted expected Hessian plus ridge slopes",
        "CV":{"grid":list(LAMBDA_GRID),"folds":5,"unit":"patient within CAL alerts","selection":"mean validation Brier; fold-local feature standardization; intercept unpenalized"},
        "track_II":"Pool-A D0, bootstrap INT6 and CALREF6 only; ready checkpoint written before transforming Pool B",
        "track_I":"operation-specific CALREF6 on CAL; operation AUDIT11 on EVAL D0 alert queue",
        "threads":1,"gpu":False,"online_timing":{"timed_repetitions":50,"warmups":1,"prediction_cache_read":False},
        "medical_data_scope":"public UCI Diabetes patient-level split; no ICU/confirmation; shared loader materializes EVAL y but it is not used in fitting, selection, or flip evaluation",
        "expected_model_fit_artifacts":6+6*12+12*(1+23),"output_create_only":True}
    _record(OUT/"protocol.json",protocol)
    splits,_audit=_splits();train,cal,eva=(splits[k] for k in ("TRAIN","CAL","EVAL"))
    data_path=pub.DATA_DIR/"diabetic_data.csv"
    csv_sha=_file_sha(data_path)
    input_summary={"source_file":"public UCI diabetic_data.csv","source_path":str(data_path),"source_sha256":csv_sha,
        "partition_counts":{"TRAIN":len(train),"CAL":len(cal),"EVAL":len(eva)},
        "patient_ids_unique_by_partition":{k:not splits[k].patient_nbr.astype(str).duplicated().any() for k in ("TRAIN","CAL","EVAL")},
        "eval_outcome_labels_materialized_by_shared_loader":True,
        "eval_outcome_labels_used_for_fitting_selection_or_flip_evaluation":False}
    _record(OUT/"input_manifest.json",input_summary)
    start_wall=time.monotonic();start_cpu=time.process_time();bank_metrics=[];cv_rows=[];lambdas=[];diagnostics=[];latencies=[];costs=[];crossovers=[];thresholds=[]
    for bank in range(6):
        state=_bank_poola_stage(bank,train,cal,eva)
        thresholds.append(_readj(state["bank_dir"]/"D0_threshold.json"))
        lambdas.extend(_readj(state["bank_dir"]/"prospective"/"selected_lambdas.json"))
        for op in OPS:
            m,cv,lam,diag,lat,setup,cost,cross=_operation(state,op,train)
            bank_metrics.extend(m);cv_rows.extend([{**r,"track":"I_Update_Pool_Aware","operation":op} for r in cv]);lambdas.extend(lam);diagnostics.extend(diag)
            costs.extend(cost);crossovers.extend(cross)
            for cohort,methods in lat.items():
                for route,values in methods.items():
                    if route not in ("OpCenter_Mean","OpCenter_IF","Update_INT6"):continue
                    latencies.append({"bank":bank,"operation":op,"cohort":cohort,"route":route,**values,
                        "includes_D0_alert_filter":bool(methods["includes_D0_alert_filter"])})
        print(f"U1 bank {bank} complete",flush=True)
    for bank in range(6):
        pdir=OUT/f"bank{bank}"/"prospective"
        for cpath in sorted(pdir.glob("cv_*.json")):
            payload=_readj(cpath)
            cv_rows.extend([{**r,"track":"II_Prospective_PoolA_Only","operation":"both future operations"} for r in payload.get("folds",[])])
    metric_keys=("R_070","AURC100","AUC","Brier","ECE10","R_100","n_alerts")
    aggregate=_summarize(bank_metrics,["track","operation","method"],metric_keys)
    winrows=_win_counts(bank_metrics)
    wmap={(r["track"],r["operation"],r["method"]):r for r in winrows}
    for row in aggregate:
        row.update({k:v for k,v in wmap.get((row["track"],row["operation"],row["method"]),{}).items()
                    if k.startswith("wins_") or k.startswith("ncomp_")})
    diagmeans=_summarize(diagnostics,["operation","variance_source"],("mean_IF_variance","mean_AUDIT_margin_sample_variance","AUDIT_over_IF_mean_ratio","instance_correlation_IF_vs_AUDIT_variance"))
    costmeans=_summarize(costs,["operation","cohort","route","N"],("setup_cpu_seconds","c_online_cpu_ms_per_row","cycle_cpu_seconds_approx"))
    crossovermeans=_summarize(crossovers,["operation","cohort","route_A","route_B"],("N_star_rows_A_equals_B",))
    fit_records=[];hash_checks=[];overlap_records=[]
    for bank in range(6):
        bp=OUT/f"bank{bank}"
        for rp in sorted(bp.glob("*/models/*/model.json")):
            row=_readj(rp);fit_records.append({k:v for k,v in row.items() if k!="beta"})
        for op in OPS:
            inv=_readj(bp/op/"model_inventory.json");hash_checks.append({"bank":bank,"operation":op,"unique_role_beta_hashes":inv["unique_role_beta_hashes"],"role_models":inv["role_models"]})
            overlap_records.append({"bank":bank,"operation":op,**{k:inv[k] for k in ("cross_model_pairs","cross_model_pairs_with_shared_patients","cross_model_shared_patient_mean","cross_model_shared_patient_max")}})
    peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if sys.platform.startswith("linux") else 1))
    report={"status":"U1_U2_U3_EXECUTION_COMPLETE","protocol_sha256":_file_sha(OUT/"protocol.json"),"input_manifest_sha256":_file_sha(OUT/"input_manifest.json"),
        "diabetes_csv_sha256":csv_sha,"partition_counts":input_summary["partition_counts"],"thresholds":thresholds,
        "model_fit_artifact_count":len(fit_records),"expected_model_fit_artifacts":protocol["expected_model_fit_artifacts"],
        "fit_cpu_seconds":float(sum(float(r.get("fit_cpu_seconds",0)) for r in fit_records)),
        "fit_wall_seconds_sum":float(sum(float(r.get("fit_wall_seconds",0)) for r in fit_records)),
        "elapsed_wall_seconds":float(time.monotonic()-start_wall),"process_cpu_seconds":float(time.process_time()-start_cpu),
        "peak_rss_bytes":peak,"unique_operation_role_hashes":int(sum(r["unique_role_beta_hashes"] for r in hash_checks)),
        "expected_operation_role_hashes":int(23*6*2),"coefficient_hash_checks":hash_checks,"cross_model_overlap":overlap_records,
        "new_prediction_model_fits":len(fit_records),"mapping_fits":"one 1D CALREF mapping per bank/method/operation, plus CAL-only CV learners; all saved by create-only checkpoints",
        "prospective_poolB_access_order":"all 6 banks have prospective_ready.json before corresponding post_poolB_access.json",
        "eval_outcome_labels_materialized_by_shared_loader":True,
        "eval_outcome_labels_used_for_fitting_selection_or_flip_evaluation":False,
        "medical_source":"UCI Diabetes public benchmark only","gpu":False,"threads":1}
    _writecsv(OUT/"U1_per_bank_metrics.csv",bank_metrics)
    _writecsv(OUT/"U1_mean_range_metrics.csv",aggregate)
    _writecsv(OUT/"U1_win_counts.csv",winrows)
    _writecsv(OUT/"U1_selected_lambdas.csv",lambdas)
    _writecsv(OUT/"U1_cv_fold_scores.csv",cv_rows)
    _writecsv(OUT/"U2_variance_diagnostics.csv",diagnostics)
    _writecsv(OUT/"U2_variance_mean_range.csv",diagmeans)
    _writecsv(OUT/"U3_online_cost_by_bank.csv",latencies)
    _writecsv(OUT/"U3_cost_crossovers.csv",crossovers)
    _writecsv(OUT/"U3_cost_mean_range.csv",costmeans)
    _writecsv(OUT/"U3_cost_crossover_mean_range.csv",crossovermeans)
    _writecsv(OUT/"U1_pool_overlap_audit.csv",overlap_records)
    _writecsv(OUT/"U1_model_fit_inventory.csv",fit_records)
    prior_path=d1.OUT/"D1_per_bank_metrics.csv"
    prior_context=[]
    if prior_path.is_file() and not prior_path.is_symlink():
        prior=pd.read_csv(prior_path)
        prior=prior[prior.scheme==d1.SCHEMES[3]]
        for row in prior.to_dict("records"):
            if row.get("method") in ("Margin","FusionMean","qMV","ConditionalIF"):
                prior_context.append({"source":"prior D1 Scheme D same-pool experiment","bank":int(row["bank"]),
                    "method":row["method"],"R_070":float(row["R_070"]),"AURC100":float(row["AURC100"]),
                    "AUC_flip":float(row["AUC_flip"]),"Brier":float(row["Brier"]),"ECE10":float(row["ECE10"]),
                    "threshold_tau":float(row["threshold_tau"]),"D0_train_n":33646,
                    "comparability":"context only; previous same-pool D0/training size differs from current 14k Pool-A D0 and is not pooled into U1"})
    _writecsv(OUT/"U1_existing_bootstrap_context.csv",prior_context)
    _record(OUT/"U1_U3_FINAL_REPORT.json",report)
    md=_markdown_report(report,aggregate,diagmeans,costmeans)
    if (OUT/"U1_U3_FINAL_REPORT.md").exists():
        if (OUT/"U1_U3_FINAL_REPORT.md").read_text()!=md:raise RuntimeError("create-only markdown differs")
    else:base._write_new(OUT/"U1_U3_FINAL_REPORT.md",md.encode())
    artifact_files=sorted(p for p in OUT.rglob("*") if p.is_file() and not p.is_symlink() and p.name!="U1_U3_artifact_manifest.json")
    manifest={"artifact_count":len(artifact_files),"files":[{"path":str(p.relative_to(OUT)),"sha256":_file_sha(p),"bytes":p.stat().st_size} for p in artifact_files]}
    _record(OUT/"U1_U3_artifact_manifest.json",manifest)
    print(md)


if __name__=="__main__":
    main()

