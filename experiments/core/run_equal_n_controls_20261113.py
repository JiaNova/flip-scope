#!/usr/bin/env python3
"""Add Pool-A SRSWOR/bootstrap controls at the update sample size.

Existing D0 models and alert queues are held fixed. Existing Half/Full update
scores are imported from the validated Step-1 cache; only missing matched-size
Pool-A reference models are fit. EVAL outcomes are never loaded.
"""
from __future__ import annotations
import csv,hashlib,json,os,resource,sys,time
from pathlib import Path
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[v]="1"
if hasattr(os,"sched_getaffinity"):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import numpy as np
from scipy import sparse
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.core import run_w1_w4 as base
from experiments.core import run_u1_u3 as u1
from experiments.core import run_c1_c4_v28 as c1
from experiments.core import run_d1_d3 as d1
from experiments.core import run_r4_credit_external as r4
from experiments.public_benchmarks import run_public_experiments as pub

OUT=ROOT/"reports/core/STEP2_C2_EQUAL_N_20261113_v3"
STEP1=ROOT/"reports/core/STEP1_FAIR_BASELINES_20261113"
U1_DIR=ROOT/"reports/core/S1_S4_20261111/U1_POOL_UPDATE_20261112"
R4_DIR=r4.RUN_DIR
U1_ROLE=[("INT",i) for i in range(1,7)]+[("CALREF",i) for i in range(1,7)]+[("AUDIT",i) for i in range(1,12)]
METHODS=("Margin","OpCenter_Mean","Conditional_IF","Center_EmpVar","Fused_Center_Only","Variance_Only_qV","qMV","Tuned_INT6_3D","Tuned_INT4_Ratio")

def sha(b):return hashlib.sha256(b).hexdigest()
def seed(*parts):return int.from_bytes(hashlib.sha256(("C2_EQUAL_N_20261113|"+"|".join(map(str,parts))).encode()).digest()[:8],"big")%(2**32-1)
def readj(p):
    if p.is_symlink() or not p.is_file():raise RuntimeError(f"unsafe/missing {p}")
    return json.loads(p.read_text())
def loadnpz(p):
    if p.is_symlink() or not p.is_file():raise RuntimeError(f"unsafe/missing {p}")
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def writej(p,obj):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n");os.replace(tmp,p)
def writecsv(p,rows):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
    tmp=p.with_name(p.name+".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,p)
def writenpz(p,**a):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp.npz");np.savez_compressed(tmp,**a);os.replace(tmp,p)
def fit_model(x,y,w,path,model_id,bank,operation,role,replicate,draw_digest):
    t0=time.process_time();beta,info=base._fit_irls(x.tocsr(),np.asarray(y,np.int8),np.asarray(w,float),ridge=1.)
    meta={"model_id":model_id,"bank":bank,"operation":operation,"role":role,"replicate":replicate,
        "train_n":int(np.asarray(y).size),"sample_weight_sum":float(np.sum(w)),"draw_digest":draw_digest,
        "fit_cpu_seconds":time.process_time()-t0,"beta_sha256":sha(beta.astype("<f8").tobytes()),"solver":info,"beta":beta.tolist()}
    writej(path,meta);return beta,{k:v for k,v in meta.items() if k!="beta"}
def margins(x,beta,gamma):return np.asarray(x@beta).reshape(-1)-gamma
def fit_bootstrap_if(x,y,beta,nboot,wcenter):
    """Sandwich variance for nboot multinomial draws from the fixed D0 multiset."""
    t0=time.process_time();n,d=x.shape;pen=np.full(d,1.);pen[0]=0.
    wcenter=np.asarray(wcenter,float)
    if wcenter.ndim==0:wcenter=float(wcenter)
    elif wcenter.shape!=(n,) or not np.allclose(wcenter,wcenter[0],rtol=0,atol=0):raise ValueError("bootstrap center weights must be a constant scalar/vector")
    eta=np.asarray(x@beta).reshape(-1);p=expit(np.clip(eta,-40,40));u=x.multiply((p-y)[:,None]).tocsr()
    mu=np.asarray(u.mean(axis=0)).reshape(-1);jraw=(u.T@u).toarray()-n*np.outer(mu,mu)
    cov=jraw/max(n-1,1);J=float(nboot)*cov
    h=(x.T@sparse.diags(wcenter*np.maximum(p*(1-p),1e-12))@x).toarray();h.flat[::d+1]+=pen
    hi=np.linalg.inv(h);sigma=hi@J@hi;sigma=(sigma+sigma.T)*.5
    center_weight=float(wcenter if np.ndim(wcenter)==0 else wcenter[0])
    return sigma,{"variance_cpu_seconds":time.process_time()-t0,"n_d0_rows":n,"bootstrap_n":nboot,"center_weight":center_weight}
def pred_variance(x,sigma):
    xa=x.toarray();return np.maximum(np.einsum("ij,jk,ik->i",xa,sigma,xa,optimize=True),0.)
def fit_heads(bank,op,z0c,z0e,zRc,zRe,zIc,zIe,zCc,zAe,vIc,vIe,folder,cvseed):
    tcal=(zCc<0).mean(axis=1);audit=(zAe<0).astype(np.int8)
    zbC=zIc.mean(axis=1);zbE=zIe.mean(axis=1);s2C=np.var(zIc,axis=1,ddof=1);s2E=np.var(zIe,axis=1,ddof=1)
    sig2=float(s2C.mean());vtC=.5*s2C+.5*sig2;vtE=.5*s2E+.5*sig2;mtC=.5*(z0c+zbC);mtE=.5*(z0e+zbE)
    rawC={"Margin":z0c,"OpCenter_Mean":zRc,"OpCenter_IF":zRc/np.sqrt(np.maximum(vIc,1e-12)),
        "Center_EmpVar":zRc/np.sqrt(vtC),"Fused_Center_Only":mtC,"Variance_Only_qV":z0c/np.sqrt(vtC),"qMV":mtC/np.sqrt(vtC)}
    rawE={"Margin":z0e,"OpCenter_Mean":zRe,"OpCenter_IF":zRe/np.sqrt(np.maximum(vIe,1e-12)),
        "Center_EmpVar":zRe/np.sqrt(vtE),"Fused_Center_Only":mtE,"Variance_Only_qV":z0e/np.sqrt(vtE),"qMV":mtE/np.sqrt(vtE)}
    q={};maps={}
    for name,s in rawC.items():
        mp=c1._fit_score_map(s,tcal);maps[name]=mp;q[name]=c1._apply_score_map(mp,rawE[name])
    x3c=np.column_stack((z0c,zbC,np.log(np.maximum(vtC,1e-300))));x3e=np.column_stack((z0e,zbE,np.log(np.maximum(vtE,1e-300))))
    f3,p3,fold3=u1._cv_brier("Tuned_INT6_3D",x3c,tcal,x3e,bank,cvseed);q["Tuned_INT6_3D"]=p3
    x4c=np.column_stack((z0c,zbC,np.log(np.maximum(vtC,1e-300)),mtC/np.sqrt(vtC)))
    x4e=np.column_stack((z0e,zbE,np.log(np.maximum(vtE,1e-300)),mtE/np.sqrt(vtE)))
    f4,p4,fold4=u1._cv_brier("Tuned_INT4_Ratio",x4c,tcal,x4e,bank,cvseed+1);q["Tuned_INT4_Ratio"]=p4
    dataset_name=folder.parents[1].name
    met=[{"dataset":dataset_name,"bank":bank,"operation":op,"method":name,**u1._metrics(p,audit)} for name,p in q.items()]
    calpath=folder/"calibration_and_cv.json"
    calobj={"one_dim_maps":maps,"Tuned_INT6_3D":f3,"Tuned_INT4_Ratio":f4,
        "CALREF_target_sha256":sha(tcal.astype("<f8").tobytes()),"lambda_grid":list(u1.LAMBDA_GRID),"EVAL_used_for_selection":False,
        "cv_fit_count":len(fold3)+len(fold4)}
    if calpath.exists():
        if readj(calpath)!=json.loads(json.dumps(calobj,sort_keys=True)):raise RuntimeError(f"existing CAL result differs: {calpath}")
    else:writej(calpath,calobj)
    predpath=folder/"prediction_arrays.npz"
    arr={"z0_cal":z0c,"z0_eval":z0e,"zR_cal":zRc,"zR_eval":zRe,"zINT_cal":zIc,"zINT_eval":zIe,
        "zCALREF_cal":zCc,"zAUDIT_eval":zAe,"audit_eval":audit,"vIF_cal":vIc,"vIF_eval":vIe,"vtilde_cal":vtC,"vtilde_eval":vtE,
        "scored_method_names":np.asarray(sorted(q),dtype="U64"),"scored":np.column_stack([q[k] for k in sorted(q)])}
    if predpath.exists():
        old=loadnpz(predpath)
        if old["scored_method_names"].tolist()!=arr["scored_method_names"].tolist() or not np.allclose(old["scored"],arr["scored"],rtol=0,atol=1e-12):
            raise RuntimeError(f"existing predictions differ: {predpath}")
    else:writenpz(predpath,**arr)
    return met,[*fold3,*fold4],sig2

def diabetes_run(out,method_rows,fit_rows,cv_rows):
    selected,_=pub.load_selected_diabetes();train=selected[selected.split=="TRAIN"].reset_index(drop=True);cal=selected[selected.split=="CAL"].reset_index(drop=True);eva=selected[selected.split=="EVAL"].reset_index(drop=True)
    for bank in range(6):
        bdir=U1_DIR/f"bank{bank}";pool=loadnpz(bdir/"pool_split.npz");aidx=pool["poolA_global_indices"].astype(int);bidx=pool["poolB_global_indices"].astype(int)
        A=train.iloc[aidx].reset_index(drop=True);B=train.iloc[bidx].reset_index(drop=True);pipe=d1._fit_schema(A,d1.SCHEMES[3])
        # A is the frozen Pool-A feature-fitting scope.  Check its actual feature
        # schema against the saved prospective model artifact before reusing D0.
        saved_meta=readj(bdir/"feature_metadata.json")
        saved_names=saved_meta.get("feature_names")
        if saved_names is not None and list(pipe["metadata"]["feature_names"])!=list(saved_names):
            raise RuntimeError(f"Diabetes Scheme-D Pool-A feature order changed for bank {bank}")
        xA=d1._transform(A,pipe);xcal=d1._transform(cal,pipe);xe=d1._transform(eva,pipe);yA=A.y_readmit30.to_numpy(np.int8)
        d0meta=readj(bdir/"poolA_only/models/D0/model.json");d0=np.asarray(d0meta["beta"],float)
        d0draw=loadnpz(bdir/"poolA_only/models/D0/sampling.npz")["poolA_relative"].astype(int);xD0=xA[d0draw];yD0=yA[d0draw]
        th=readj(bdir/"D0_threshold.json");gamma=float(th["gamma0"]);z0cAll=margins(xcal,d0,gamma);z0eAll=margins(xe,d0,gamma);ac=z0cAll>=0;ae=z0eAll>=0
        prospective=loadnpz(bdir/"prospective/prospective_scores.npz")
        if not np.array_equal(ae,prospective["eval_alert_mask"].astype(bool)):
            raise RuntimeError("Diabetes frozen D0 EVAL alert queue mismatch")
        if int(ac.sum())!=int(th["cal_alerts"]) or int(ae.sum())!=int(th["eval_alerts"]):
            raise RuntimeError("Diabetes frozen D0 CAL/EVAL alert count mismatch")
        if not np.allclose(z0cAll[ac],prospective["z0_cal_alert"],rtol=0,atol=1e-10):
            raise RuntimeError("Diabetes frozen D0 CAL alert margins mismatch")
        if not np.allclose(z0eAll[ae],prospective["z0_eval_alert"],rtol=0,atol=1e-10):
            raise RuntimeError("Diabetes frozen D0 EVAL alert margins mismatch")
        z0c=z0cAll[ac];z0e=z0eAll[ae];xc=xcal[ac];xeA=xe[ae]
        proot=bdir/"prospective/models"
        intB=[np.asarray(readj(proot/f"INT-{i:02d}/model.json")["beta"],float) for i in range(1,7)]
        calB=[np.asarray(readj(proot/f"CALREF-{i:02d}/model.json")["beta"],float) for i in range(1,7)]
        # Existing Pool-A bootstrap INT/CALREF are reused; only the missing AUDIT column set is fitted.
        ziC=np.column_stack([margins(xc,b,gamma) for b in intB]);zcC=np.column_stack([margins(xc,b,gamma) for b in calB])
        for op in ("PoolA_Bootstrap","PoolA_SRSWOR"):
            folder=out/f"Diabetes_Scheme_D/bank{bank}/{op}";folder.mkdir(parents=True)
            if op=="PoolA_Bootstrap":
                zIc=ziC;zCc=zcC;zIe=np.column_stack([margins(xeA,b,gamma) for b in intB]);center=d0
                auditB=[];idx_arrays={"D0_PoolA_draw":d0draw.astype(np.int32)}
                for j in range(1,12):
                    sd=seed("Diabetes",bank,op,"AUDIT",j);rng=np.random.default_rng(sd);draw=rng.integers(0,len(d0draw),size=14000);ids=d0draw[draw]
                    weights=np.ones(14000);beta,rec=fit_model(xA[ids],yA[ids],weights,folder/f"AUDIT_{j:02d}.json",f"Diabetes/bank{bank}/{op}/AUDIT-{j:02d}",bank,op,"AUDIT",j,sha(ids.astype("<i8").tobytes()))
                    fit_rows.append(rec);idx_arrays[f"AUDIT_{j:02d}_PoolA_idx"]=ids.astype(np.int32);auditB.append(margins(xeA,beta,gamma))
                zCc=zcC;zIe=np.column_stack([margins(xeA,b,gamma) for b in intB]);zAe=np.column_stack(auditB)
                zRc=z0c;zRe=z0e
                pv=loadnpz(bdir/"prospective/prospective_scores.npz");vIc=pv["v0_cal"];vIe=pv["v0_eval_alert"]
                writenpz(folder/"sampling_indices.npz",**idx_arrays)
            else:
                n=14000;sd=seed("Diabetes",bank,op,"SRSWOR");ia=np.sort(np.random.default_rng(sd).choice(len(A),n,replace=False))
                # Expected finite-population training objective is the operator center.
                wc=np.full(len(A),n/len(A));center,rec=fit_model(xA,yA,wc,folder/"operator_center.json",f"Diabetes/bank{bank}/{op}/CENTER",bank,op,"OperatorCenter",0,sha(ia.astype("<i8").tobytes()))
                fit_rows.append(rec);zRc=margins(xc,center,gamma);zRe=margins(xeA,center,gamma)
                ziCcols=[];ziEcols=[];zcCcols=[];zaEcols=[];idx_arrays={"SRSWOR_seed":np.asarray([sd],np.int64)}
                for role,j in U1_ROLE:
                    rs=seed("Diabetes",bank,op,role,j);take=np.sort(np.random.default_rng(rs).choice(len(A),n,replace=False))
                    beta,rec=fit_model(xA[take],yA[take],np.ones(n),folder/f"{role}_{j:02d}.json",f"Diabetes/bank{bank}/{op}/{role}-{j:02d}",bank,op,role,j,sha(take.astype("<i8").tobytes()))
                    fit_rows.append(rec);idx_arrays[f"{role}_{j:02d}_PoolA_idx"]=take.astype(np.int32)
                    zc=margins(xc,beta,gamma);ze=margins(xeA,beta,gamma)
                    if role=="INT":ziCcols.append(zc);ziEcols.append(ze)
                    elif role=="CALREF":zcCcols.append(zc)
                    else:zaEcols.append(ze)
                zIc=np.column_stack(ziCcols);zIe=np.column_stack(ziEcols);zCc=np.column_stack(zcCcols);zAe=np.column_stack(zaEcols)
                # One-source SRSWOR Sandwich: represent the absent B pool as a zero-weight dummy row.
                xdummy=sparse.csr_matrix((1,xA.shape[1]));vIc,vIe,_,ifmeta=u1._finite_population_if(xA,yA,xdummy,np.zeros(1,np.int8),center,n,0,xc,xeA)
                idx_arrays["IF_cpu_seconds"]=np.asarray([ifmeta["total_cpu_seconds"]]);writenpz(folder/"sampling_indices.npz",**idx_arrays)
            met,folds,sig2=fit_heads(bank,op,z0c,z0e,zRc,zRe,zIc,zIe,zCc,zAe,vIc,vIe,folder,seed("Diabetes",bank,op,"CV"))
            method_rows.extend({"dataset":"Diabetes_Scheme_D","training_n":14000,"bank":bank,"operation":op,**r} for r in met)
            cv_rows.extend({"dataset":"Diabetes_Scheme_D","bank":bank,"operation":op,**r} for r in folds)
            writej(folder/"operation_manifest.json",{"dataset":"Diabetes_Scheme_D","bank":bank,"operation":op,"fixed_D0_reused":True,
                "D0_n":14000,"reference_n":14000,"CAL_threshold_and_alert_queue_reused":True,"cal_alert_n":int(ac.sum()),"eval_alert_n":int(ae.sum()),
                "new_predictive_model_fits":11 if op=="PoolA_Bootstrap" else 24,"existing_INT_CALREF_reused":op=="PoolA_Bootstrap",
                "fit_sample_policy":"Pool-A multinomial bootstrap from D0 multiset" if op=="PoolA_Bootstrap" else "Pool-A SRSWOR",
                "EVAL_outcome_labels_used_for_fit_or_selection":False,"sigma2_cal":sig2})

def credit_run(out,method_rows,fit_rows,cv_rows):
    split,y,audit=r4.parse_without_eval_labels()
    for bank in range(6):
        a_ids,b_ids=r4.bank_pool_ids(split["TRAIN"],bank);tr=split["TRAIN"].set_index("ID",drop=False)
        A=tr.loc[sorted(a_ids)].reset_index(drop=True);B=tr.loc[sorted(b_ids)].reset_index(drop=True);prep=r4.fit_preprocessor(A)
        xA=r4.transform(A,prep);xB=r4.transform(B,prep);xcal=r4.transform(split["CAL"],prep);xe=r4.transform(split["EVAL"],prep)
        ymap={int(i):int(v) for i,v in zip(split["TRAIN"].ID,y["TRAIN"])};yA=np.asarray([ymap[int(i)] for i in A.ID],np.int8);yB=np.asarray([ymap[int(i)] for i in B.ID],np.int8)
        olddir=R4_DIR/f"bank{bank}";d0meta=readj(olddir/"D0.json");d0=np.asarray(d0meta["beta"],float);gamma=float(readj(olddir/"bank_manifest.json")["gamma0"])
        if list(prep["feature_names"])!=list(readj(olddir/"preprocessor.json")["feature_names"]):
            raise RuntimeError(f"Credit Pool-A feature order changed for bank {bank}")
        z0cAll=margins(xcal,d0,gamma);z0eAll=margins(xe,d0,gamma);ac=z0cAll>=0;ae=z0eAll>=0;z0c=z0cAll[ac];z0e=z0eAll[ae];xc=sparse.csr_matrix(xcal[ac]);xeA=sparse.csr_matrix(xe[ae])
        old_scores=loadnpz(olddir/"Op_Half_Replace/prediction_arrays.npz")
        if not np.allclose(z0c,old_scores["z0_cal_alert"],rtol=0,atol=1e-10) or not np.allclose(z0e,old_scores["z0_eval_alert"],rtol=0,atol=1e-10):
            raise RuntimeError(f"Credit frozen D0 margins/alert queue changed for bank {bank}")
        draw=loadnpz(olddir/"draws.npz")["D0_poolA_draw"].astype(int);xD0=sparse.csr_matrix(xA[draw]);yD0=np.asarray([yA[i] for i in draw],np.int8)
        for op in ("PoolA_Bootstrap","PoolA_SRSWOR"):
            folder=out/f"UCI_Credit/bank{bank}/{op}";folder.mkdir(parents=True)
            n=6000
            if op=="PoolA_Bootstrap":
                wc=np.full(len(draw),n/len(draw));center,rec=fit_model(xD0,yD0,wc,folder/"operator_center.json",f"Credit/bank{bank}/{op}/CENTER",bank,op,"OperatorCenter",0,sha(draw.astype("<i8").tobytes()))
                fit_rows.append(rec);ziCcols=[];ziEcols=[];zcCcols=[];zaEcols=[];idx_arrays={"D0_PoolA_draw":draw.astype(np.int32)}
                for role,j in U1_ROLE:
                    rs=seed("Credit",bank,op,role,j);rng=np.random.default_rng(rs);take=rng.integers(0,len(draw),size=n);poolidx=draw[take]
                    xx=sparse.csr_matrix(xA[poolidx]);yy=yA[poolidx]
                    beta,rec=fit_model(xx,yy,np.ones(n),folder/f"{role}_{j:02d}.json",f"Credit/bank{bank}/{op}/{role}-{j:02d}",bank,op,role,j,sha(poolidx.astype("<i8").tobytes()))
                    fit_rows.append(rec);idx_arrays[f"{role}_{j:02d}_PoolA_idx"]=poolidx.astype(np.int32)
                    zc=margins(xc,beta,gamma);ze=margins(xeA,beta,gamma)
                    if role=="INT":ziCcols.append(zc);ziEcols.append(ze)
                    elif role=="CALREF":zcCcols.append(zc)
                    else:zaEcols.append(ze)
                zIc=np.column_stack(ziCcols);zIe=np.column_stack(ziEcols);zCc=np.column_stack(zcCcols);zAe=np.column_stack(zaEcols)
                Sigma,ifmeta=fit_bootstrap_if(xD0,yD0,center,n,np.full(len(draw),n/len(draw)))
                vIc=pred_variance(xc,Sigma);vIe=pred_variance(xeA,Sigma);idx_arrays["IF_cpu_seconds"]=np.asarray([ifmeta["variance_cpu_seconds"]]);writenpz(folder/"sampling_indices.npz",**idx_arrays)
            else:
                sd=seed("Credit",bank,op,"SRSWOR");ia=np.sort(np.random.default_rng(sd).choice(len(A),n,replace=False));wA=np.full(len(A),n/len(A))
                center,rec=fit_model(sparse.csr_matrix(xA),yA,wA,folder/"operator_center.json",f"Credit/bank{bank}/{op}/CENTER",bank,op,"OperatorCenter",0,sha(ia.astype("<i8").tobytes()))
                fit_rows.append(rec);ziCcols=[];ziEcols=[];zcCcols=[];zaEcols=[];idx_arrays={"SRSWOR_seed":np.asarray([sd],np.int64)}
                for role,j in U1_ROLE:
                    rs=seed("Credit",bank,op,role,j);take=np.sort(np.random.default_rng(rs).choice(len(A),n,replace=False))
                    beta,rec=fit_model(sparse.csr_matrix(xA[take]),yA[take],np.ones(n),folder/f"{role}_{j:02d}.json",f"Credit/bank{bank}/{op}/{role}-{j:02d}",bank,op,role,j,sha(take.astype("<i8").tobytes()))
                    fit_rows.append(rec);idx_arrays[f"{role}_{j:02d}_PoolA_idx"]=take.astype(np.int32)
                    zc=margins(xc,beta,gamma);ze=margins(xeA,beta,gamma)
                    if role=="INT":ziCcols.append(zc);ziEcols.append(ze)
                    elif role=="CALREF":zcCcols.append(zc)
                    else:zaEcols.append(ze)
                zIc=np.column_stack(ziCcols);zIe=np.column_stack(ziEcols);zCc=np.column_stack(zcCcols);zAe=np.column_stack(zaEcols)
                # Single-source finite-population IF.
                dummy=sparse.csr_matrix((1,xA.shape[1]));vIc,vIe,_,ifmeta=u1._finite_population_if(sparse.csr_matrix(xA),yA,dummy,np.zeros(1,np.int8),center,n,0,xc,xeA)
                idx_arrays["IF_cpu_seconds"]=np.asarray([ifmeta["total_cpu_seconds"]]);writenpz(folder/"sampling_indices.npz",**idx_arrays)
            zRc=margins(xc,center,gamma);zRe=margins(xeA,center,gamma)
            met,folds,sig2=fit_heads(bank,op,z0c,z0e,zRc,zRe,zIc,zIe,zCc,zAe,vIc,vIe,folder,seed("Credit",bank,op,"CV"))
            method_rows.extend({"dataset":"UCI_Credit","training_n":6000,"bank":bank,"operation":op,**r} for r in met)
            cv_rows.extend({"dataset":"UCI_Credit","bank":bank,"operation":op,**r} for r in folds)
            writej(folder/"operation_manifest.json",{"dataset":"UCI_Credit","bank":bank,"operation":op,"fixed_D0_reused":True,
                "D0_training_n":12000,"reference_n":6000,"CAL_threshold_and_alert_queue_reused":True,"cal_alert_n":int(ac.sum()),"eval_alert_n":int(ae.sum()),
                "new_predictive_model_fits":24,"sigma2_cal":sig2,
                "operator_center_weight_sum":float(6000),"fit_sample_policy":"D0 multiset bootstrap n=6000" if op=="PoolA_Bootstrap" else "Pool-A SRSWOR n=6000",
                "EVAL_outcome_labels_used_for_fit_or_selection":False})

def main():
    if OUT.exists() or OUT.is_symlink():raise FileExistsError(f"create-only output exists: {OUT}")
    OUT.mkdir(parents=True);w0=time.monotonic();c0=time.process_time();method_rows=[];fit_rows=[];cv_rows=[]
    diabetes_run(OUT,method_rows,fit_rows,cv_rows);credit_run(OUT,method_rows,fit_rows,cv_rows)
    # Add the already fitted matched-size Half/Full rows: same saved D0, thresholds, queue and per-model n.
    cached=list(csv.DictReader((STEP1/"point_metrics_by_bank.csv").open()))
    for r in cached:
        if r["operation"] not in ("Op_Half_Replace","Op_Full_Replace") or r["dataset"] not in ("Diabetes_Scheme_D","UCI_Credit"):continue
        method_rows.append({"dataset":r["dataset"],"training_n":14000 if r["dataset"]=="Diabetes_Scheme_D" else 6000,
            "bank":int(r["bank"]),"operation":r["operation"],"method":r["method"],**{k:float(r[k]) for k in ("R_070","AURC100","AUC","Brier","ECE10")},
            "n_alerts":int(r["n_alerts"]),"audit_B":int(r["audit_B"]),"new_model_fits":0,"cached_same_D0":True})
    summaries=[]
    for ds in ("Diabetes_Scheme_D","UCI_Credit"):
        for op in ("PoolA_SRSWOR","PoolA_Bootstrap","Op_Half_Replace","Op_Full_Replace"):
            for m in sorted({r["method"] for r in method_rows if r.get("dataset")==ds and r.get("operation")==op}):
                rr=[r for r in method_rows if r.get("dataset")==ds and r.get("operation")==op and r.get("method")==m]
                row={"dataset":ds,"training_n":14000 if ds=="Diabetes_Scheme_D" else 6000,"operation":op,"method":m,"banks_n":len(rr)}
                for metric in ("R_070","AURC100","AUC","Brier","ECE10"):
                    vals=np.asarray([x[metric] for x in rr],float);row[metric+"_mean"]=float(vals.mean());row[metric+"_min"]=float(vals.min());row[metric+"_max"]=float(vals.max())
                summaries.append(row)
    writecsv(OUT/"same_d0_same_n_per_bank.csv",method_rows);writecsv(OUT/"same_d0_same_n_summary.csv",summaries)
    writecsv(OUT/"fit_records.csv",fit_rows);writecsv(OUT/"cv_fold_scores.csv",cv_rows)
    audit={"status":"COMPLETED","D0_models_refit":0,"new_model_fits":len(fit_rows),"datasets":["Diabetes_Scheme_D","UCI_Credit"],
        "same_D0_threshold_and_EVAL_alert_queue_verified":True,"fixed_reference_n":{"Diabetes":14000,"Credit":6000},
        "new_fits_by_dataset":{"Diabetes":sum(x["model_id"].startswith("Diabetes/") for x in fit_rows),"Credit":sum(x["model_id"].startswith("Credit/") for x in fit_rows)},
        "CALREF_maps_and_CV_fits":len(cv_rows),"EVAL_outcome_labels_used_for_fit_or_selection":False,
        "Diabetes_EVAL_outcome_columns_materialized_by_existing_audit_loader":True,
        "Credit_EVAL_outcome_values_materialized":False,
        "cpu_threads":1,"gpu":False,
        "cpu_seconds":time.process_time()-c0,"wall_seconds":time.monotonic()-w0,
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "comparability_note":"All four operations are evaluated on the same saved D0 and alert queue within each bank; every refit uses the same n. Existing Half/Full result rows come from the validated Step-1 cache."}
    writej(OUT/"execution_audit.json",audit);print(json.dumps(audit,indent=2))

if __name__=="__main__":main()
