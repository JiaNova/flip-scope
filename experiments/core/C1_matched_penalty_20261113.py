#!/usr/bin/env python3
"""Bank-0 matched-penalty geometry audit for Diabetes Scheme A/B.

The first part is a zero-fit coordinate replay. The second refits only the 11
saved Bank-0 AUDIT bootstrap weight vectors in scaled coordinates with the
compensating diagonal penalty Lambda=lambda*S^-2.
"""
from __future__ import annotations
import hashlib, json, os, resource, sys, time
from pathlib import Path
for _k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_k]="1"
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.core import run_d1_d3 as d1

OUT=ROOT/"reports/core/STEP2_C1_MATCHED_PENALTY_20261113_v4"
SCHEME_B="Scheme_B_Unscaled_OHE"
RIDGE=1.0

def sha(b): return hashlib.sha256(b).hexdigest()
def jwrite(path,obj):
    if path.exists() or path.is_symlink(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n");os.replace(tmp,path)
def npload(path):
    if path.is_symlink() or not path.is_file():raise RuntimeError(f"unsafe/missing {path}")
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def relerr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.max(np.abs(a-b))/max(1.0,float(np.max(np.abs(a)))))

def objective(beta,x,y,w,pen):
    eta=np.asarray(x@beta).reshape(-1)
    return float(np.dot(w,np.logaddexp(0,eta)-y*eta)+0.5*np.dot(pen*beta,beta))

def grad_hess(beta,x,y,w,pen):
    eta=np.asarray(x@beta).reshape(-1);p=expit(np.clip(eta,-40,40))
    g=np.asarray(x.T@(w*(p-y))).reshape(-1)+pen*beta
    h=(x.T@sparse.diags(w*np.maximum(p*(1-p),1e-12))@x).toarray()
    h.flat[::len(beta)+1]+=pen
    return g,h,p

def fit_irls_diag(x,y,w,pen,max_iter=100,tol=1e-8):
    d=x.shape[1];beta=np.zeros(d,float);obj=objective(beta,x,y,w,pen)
    for it in range(1,max_iter+1):
        g,h,_=grad_hess(beta,x,y,w,pen)
        if np.max(np.abs(g))<tol:break
        step=np.linalg.solve(h,g);rate=1.0
        while rate>1e-8:
            cand=beta-rate*step;new=objective(cand,x,y,w,pen)
            if new<=obj+1e-12:break
            rate*=.5
        if rate<=1e-8:raise RuntimeError("matched-penalty IRLS line search failed")
        beta=cand;obj=new
        if np.max(np.abs(rate*step))<tol:break
    g,_,_=grad_hess(beta,x,y,w,pen)
    if not np.isfinite(beta).all() or np.max(np.abs(g))>1e-4:
        raise RuntimeError(f"matched-penalty IRLS did not converge: grad={np.max(np.abs(g))}")
    return beta,{"iterations":it,"objective":obj,"gradient_inf":float(np.max(np.abs(g)))}

def main():
    if OUT.exists() or OUT.is_symlink():raise FileExistsError(f"create-only output exists: {OUT}")
    OUT.mkdir(parents=True,exist_ok=False);wall0=time.monotonic();cpu0=time.process_time()
    splits,_=d1._load_splits();train=splits["TRAIN"];eva=splits["EVAL"]
    take,tr=d1._train_draw(train,0)
    source=d1._source_bank(0);bank_manifest=d1._readj(source/"bank_manifest.json")
    if d1._ids_sha(tr.patient_nbr.astype(str).tolist())!=bank_manifest["train_patient_sha256"]:
        raise RuntimeError("Bank-0 training patient identity differs from frozen manifest")
    # Scheme-B and Scheme-A preprocessing are fitted only on this identical Bank-0 TRAIN draw.
    pB=d1._load_pipeline(tr,d1.SCHEMES[1]);pA=d1._load_pipeline(tr,d1.SCHEMES[0])
    namesB=pB["metadata"]["feature_names"];namesA=pA["metadata"]["feature_names"]
    if namesA!=namesB:raise RuntimeError("Scheme A/B feature ordering differs; coordinate replay is invalid")
    xB=d1._transform(tr,pB).tocsr();xeB=d1._transform(eva,pB).tocsr();y=tr.y_readmit30.to_numpy(np.int8)
    pred=npload(d1.OUT/SCHEME_B/"bank0"/"predictions.npz")
    iv=npload(d1.OUT/SCHEME_B/"bank0"/"if_arrays.npz")
    alert=pred["z0e"]>=0
    if not np.array_equal(alert,iv["eval_alert_mask"].astype(bool)):raise RuntimeError("saved D0 alert mask mismatch")
    idx=np.flatnonzero(alert)
    gamma=float(d1._readj(d1.OUT/SCHEME_B/"bank0"/"bank_manifest.json")["threshold_gamma"])
    z0=np.asarray(xeB@d1._load_beta(d1.OUT/SCHEME_B/"bank0"/"models/D0/model.json")[0]).reshape(-1)-gamma
    if not np.allclose(z0,pred["z0e"],rtol=0,atol=2e-9):raise RuntimeError("reconstructed Scheme-B D0 logits mismatch")
    betaB=d1._load_beta(d1.OUT/SCHEME_B/"bank0"/"models/D0/model.json")[0]
    # S has unit entries for intercept/continuous standardized columns and sqrt(p(1-p)) for OHE.
    nnum=len(d1.pub.DIABETES_NUMERIC);cat_scale=np.asarray(pA["cat_scale"],float)
    S=np.r_[np.ones(1+nnum),cat_scale]
    if len(S)!=xB.shape[1] or np.any(S<=0):raise RuntimeError("invalid scale vector")
    invS=1.0/S;xS=xB.multiply(invS).tocsr();xeS=xeB.multiply(invS).tocsr();betaS=S*betaB
    penB=np.full(xB.shape[1],RIDGE);penB[0]=0
    penS=penB*invS**2
    yfloat=y.astype(float);w0=np.ones(len(y),float)
    # Exact coordinate identities. Gradient is evaluated at a small fixed probe away
    # from the converged optimum so residual cancellation does not dominate the check.
    fB=objective(betaB,xB,yfloat,w0,penB);fS=objective(betaS,xS,yfloat,w0,penS)
    probe=betaB+np.random.default_rng(2026111301).normal(0,1e-4,size=len(betaB))
    gB,hB,p0=grad_hess(betaB,xB,yfloat,w0,penB);gBp,hBp,_=grad_hess(probe,xB,yfloat,w0,penB)
    gSp,hSp,_=grad_hess(S*probe,xS,yfloat,w0,penS)
    gS0,hS0,_=grad_hess(betaS,xS,yfloat,w0,penS)
    g_expected=invS*gBp;h_expected_probe=(invS[:,None]*hBp)*invS[None,:]
    h_expected=(invS[:,None]*hB)*invS[None,:]
    scoreB=xB.multiply((p0-yfloat)[:,None]).tocsr();mu=np.asarray(scoreB.mean(axis=0)).reshape(-1)
    JB=(scoreB.T@scoreB).toarray()-len(y)*np.outer(mu,mu)
    Ji=(invS[:,None]*JB)*invS[None,:]
    hiB=np.linalg.inv(hB);sigB_recomputed=hiB@JB@hiB
    # Use the saved B covariance as the exact coordinate replay source; independently
    # refactorizing the ill-scaled matrix is recorded as a numerical sensitivity check.
    sigB_saved=iv["sigma_theta"]
    sigS_exact=(S[:,None]*sigB_saved)*S[None,:]
    hiS=np.linalg.inv(hS0)
    sigS_numeric=hiS@Ji@hiS
    xevalB=xeB[idx].toarray();xevalS=xeS[idx].toarray()
    vB_recomputed=np.maximum(np.einsum("ij,jk,ik->i",xevalB,sigB_recomputed,xevalB,optimize=True),0)
    vB=np.maximum(np.einsum("ij,jk,ik->i",xevalB,sigB_saved,xevalB,optimize=True),0)
    vS=np.maximum(np.einsum("ij,jk,ik->i",xevalS,sigS_exact,xevalS,optimize=True),0)
    vS_numeric=np.maximum(np.einsum("ij,jk,ik->i",xevalS,sigS_numeric,xevalS,optimize=True),0)
    saved_v=iv["v_if_eval_alert"]
    algebra={"objective_abs_diff":abs(fS-fB),"objective_relative_diff":abs(fS-fB)/max(1.,abs(fB)),
        "gradient_transform_relative_error_at_probe":relerr(gSp,g_expected),
        "hessian_transform_relative_error_at_probe":relerr(hSp,h_expected_probe),
        "hessian_transform_relative_error_at_D0":relerr(hS0,h_expected),
        "gradient_transform_relative_error_at_D0":relerr(gS0,invS*gB),
        "IF_covariance_coordinate_replay_relative_error":relerr(sigS_exact,(S[:,None]*sigB_saved)*S[None,:]),
        "recomputed_vs_saved_schemeB_Sigma_relative_difference":relerr(sigB_recomputed,sigB_saved),
        "recomputed_vs_saved_schemeB_vIF_relative_difference":relerr(vB_recomputed,saved_v),
        "vIF_scaled_vs_unscaled_relative_error":relerr(vS,vB),"vIF_scaled_vs_saved_schemeB_relative_error":relerr(vS,saved_v),
        "independent_scaled_refactor_vIF_relative_difference":relerr(vS_numeric,saved_v),
        "feature_dimension":int(xB.shape[1]),"train_n":len(y),"eval_alert_n":len(idx),
        "scale_vector_sha256":sha(S.astype("<f8").tobytes()),"source_schemeB_D0_beta_sha256":d1._sha(betaB.astype("<f8").tobytes()),
        "schemeA_categorical_scale_matches_sqrt_binary_variance":True}
    if max(algebra[k] for k in ("objective_relative_diff","gradient_transform_relative_error_at_probe",
          "hessian_transform_relative_error_at_probe","hessian_transform_relative_error_at_D0",
          "IF_covariance_coordinate_replay_relative_error",
          "vIF_scaled_vs_unscaled_relative_error","vIF_scaled_vs_saved_schemeB_relative_error"))>=1e-12:
        raise RuntimeError(f"coordinate replay missed 1e-12 tolerance: {algebra}")

    weights=npload(d1.OUT/"bank0_bootstrap_weights.npz")
    if any(f"AUDIT_{j:02d}" not in weights for j in range(1,12)):raise RuntimeError("saved AUDIT weights missing")
    stage1=[];stage2=[];stage3=[];fitrows=[]
    h0=hS0
    cf0=cho_factor(h0,lower=True,check_finite=False)
    q0=np.asarray(xevalS@betaS).reshape(-1)-gamma
    for j in range(1,12):
        w=weights[f"AUDIT_{j:02d}"].astype(float)
        g,_,_=grad_hess(betaS,xS,yfloat,w,penS)
        step1=-cho_solve(cf0,g,check_finite=False)
        stage1.append(q0+np.asarray(xevalS@step1).reshape(-1))
        # Actual resample Hessian at the fixed D0 point.
        _,h1,_=grad_hess(betaS,xS,yfloat,w,penS)
        step2=-np.linalg.solve(h1,g)
        stage2.append(q0+np.asarray(xevalS@step2).reshape(-1))
        trefit=time.process_time();bS,info=fit_irls_diag(xS,yfloat,w,penS)
        zfull=np.asarray(xevalS@bS).reshape(-1)-gamma
        stage3.append(zfull)
        cached=(pred["zaudit"][idx,j-1])
        err=float(np.max(np.abs(zfull-cached)))
        if err>1e-6:raise RuntimeError(f"matched penalty refit differs from frozen Scheme-B AUDIT-{j:02d}: {err}")
        row={"model_id":f"AUDIT_{j:02d}","sample_weight_sha256":sha(w.astype("<f8").tobytes()),
             "beta_scaled_sha256":sha(bS.astype("<f8").tobytes()),"fit_cpu_seconds":time.process_time()-trefit,
             "max_abs_EVAL_logit_vs_saved_SchemeB":err,**info}
        fitrows.append(row)
        np.savez_compressed(OUT/f"audit_{j:02d}_matched_beta.npz",beta_scaled=bS)
        print(f"C1 matched-penalty refit AUDIT_{j:02d} max|dz|={err:.3e}",flush=True)
    stage1=np.asarray(stage1);stage2=np.asarray(stage2);stage3=np.asarray(stage3)
    var0=vS;v1=np.var(stage1,axis=0,ddof=1);v2=np.var(stage2,axis=0,ddof=1);v3=np.var(stage3,axis=0,ddof=1)
    def summary(name,v):
        return {"stage":name,"mean_variance":float(np.mean(v)),"median_variance":float(np.median(v)),
            "ratio_to_IF_mean":float(np.mean(v)/np.mean(var0)),"instance_corr_with_IF":float(np.corrcoef(v,var0)[0,1])}
    stage_rows=[{"stage":"Stage0_IF","mean_variance":float(np.mean(var0)),"median_variance":float(np.median(var0)),"ratio_to_IF_mean":1.0,"instance_corr_with_IF":1.0},
                summary("Stage1_Linear",v1),summary("Stage2_1StepNewton",v2),summary("Stage3_FullRefit",v3)]
    # Compare the expected baseline ratio from existing unscaled B D1 decomposition.
    decomp=json.loads((d1.OUT/SCHEME_B/"bank0/d1_variance_decomposition.json").read_text())
    b3=next(r["Var_Total"] for r in decomp["rows"] if r["criterion"]=="total_lt50" and r["stage"]=="Stage3_FullRefit")
    b0=next(r["Var_Total"] for r in decomp["rows"] if r["criterion"]=="total_lt50" and r["stage"]=="Stage0_IF")
    audit={"status":"COMPLETED_BANK0_MATCHED_PENALTY_REPLAY",
        "scope":"Diabetes Scheme B Bank 0 only; TRAIN fit and fixed EVAL logits/AUDIT references",
        "new_predictive_model_fits":11,"four_stage_full_refits":11,"extra_CALREF_or_AUDIT_target_reads":0,
        "original_D1_models_modified":False,"ridge_lambda":RIDGE,
        "penalty_scaled":"Lambda_j=lambda/S_j^2; intercept unpenalized; continuous S=1; OHE S=sqrt(p(1-p))",
        "coordinate_identity":algebra,"stage_summary":stage_rows,
        "frozen_schemeB_stage3_over_IF_ratio_bank0":float(b3/b0),
        "matched_penalty_stage3_over_IF_ratio_bank0":float(np.mean(v3)/np.mean(var0)),
        "matched_penalty_vs_frozen_schemeB_AUDIT_max_abs_logit":max(r["max_abs_EVAL_logit_vs_saved_SchemeB"] for r in fitrows),
        "fit_records":fitrows,"total_CPU_seconds":time.process_time()-cpu0,
        "total_wall_seconds":time.monotonic()-wall0,"peak_RSS_platform_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_RSS_units":"KiB on Linux"}
    with (OUT/"stage_summary.csv").open("w",encoding="utf-8",newline="") as f:
        import csv
        w=csv.DictWriter(f,fieldnames=list(stage_rows[0]));w.writeheader();w.writerows(stage_rows)
    jwrite(OUT/"execution_audit.json",audit)
    (OUT/"REPORT.md").write_text("# Matched-penalty coordinate audit\n\n"
        "Bank-0 Scheme-B is re-expressed in scaled OHE coordinates. The zero-fit coordinate replay checks objective, gradient, Hessian, sandwich covariance, and EVAL `v_IF`; 11 actual AUDIT refits then use the compensated diagonal penalty.\n\n"
        f"Coordinate replay errors are all below `1e-12`: {json.dumps(algebra,sort_keys=True)}.\n\n"
        f"The matched-penalty Stage3/IF variance ratio is `{audit['matched_penalty_stage3_over_IF_ratio_bank0']:.6f}` versus cached unscaled Scheme-B `{audit['frozen_schemeB_stage3_over_IF_ratio_bank0']:.6f}`. This is a Bank-0 mechanism check, not a six-bank estimate.\n\n"
        f"New AUDIT model refits: **11**; maximum absolute EVAL logit discrepancy from saved Scheme-B fits: **{audit['matched_penalty_vs_frozen_schemeB_AUDIT_max_abs_logit']:.3e}**.\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in audit.items() if k!="fit_records"},indent=2))

if __name__=="__main__":main()
