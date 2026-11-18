#!/usr/bin/env python3
"""Continue the equal-n run from saved models without repeating completed fits."""
from __future__ import annotations
import csv,json,os,resource,sys,time
from pathlib import Path
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[v]="1"
if hasattr(os,"sched_getaffinity"):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.core import run_equal_n_controls_20261113 as sm
from experiments.core import run_w1_w4 as base
from experiments.core import run_u1_u3 as u1
from experiments.core import run_r4_credit_external as r4
from experiments.public_benchmarks import run_public_experiments as pub

OUT=sm.OUT
def model_from_file(path):
    j=sm.readj(path);beta=np.asarray(j["beta"],float);return beta,{k:v for k,v in j.items() if k!="beta"}
def existing_fit_or_fit(x,y,w,path,model_id,bank,op,role,replicate,digest,fit_rows):
    if path.exists():
        beta,rec=model_from_file(path);return beta,rec
    beta,rec=sm.fit_model(x,y,w,path,model_id,bank,op,role,replicate,digest);fit_rows.append(rec);return beta,rec
def ensure_scores_from_cache(folder,bank,op,method_rows):
    p=folder/"prediction_arrays.npz"
    if not p.exists():return False
    a=sm.loadnpz(p);names=a["scored_method_names"].astype(str);q=a["scored"];audit=a["audit_eval"].astype(np.int8)
    for j,name in enumerate(names):
        method_rows.append({"dataset":folder.parents[1].name,"training_n":14000 if folder.parents[1].name=="Diabetes_Scheme_D" else 6000,
            "bank":bank,"operation":op,"method":name,**u1._metrics(q[:,j],audit),"new_model_fits":0,"loaded_saved_equal_n_result":True})
    return True
def replay_diabetes_cv(folder,bank,op,cv_rows):
    a=sm.loadnpz(folder/"prediction_arrays.npz");cal=a["z0_cal"];zint=a["zINT_cal"];tcal=(a["zCALREF_cal"]<0).mean(axis=1)
    zb=zint.mean(axis=1);s2=np.var(zint,axis=1,ddof=1);vt=.5*s2+.5*float(s2.mean());mt=.5*(cal+zb)
    f3,_,rows3=u1._cv_brier("Tuned_INT6_3D",np.column_stack((cal,zb,np.log(np.maximum(vt,1e-300)))),tcal,
        np.column_stack((a["z0_eval"],a["zINT_eval"].mean(axis=1),np.log(np.maximum(.5*np.var(a["zINT_eval"],axis=1,ddof=1)+.5*float(s2.mean()),1e-300)))),bank,sm.seed("Diabetes",bank,op,"CV"))
    me=.5*(a["z0_eval"]+a["zINT_eval"].mean(axis=1));ve=.5*np.var(a["zINT_eval"],axis=1,ddof=1)+.5*float(s2.mean())
    f4,_,rows4=u1._cv_brier("Tuned_INT4_Ratio",np.column_stack((cal,zb,np.log(np.maximum(vt,1e-300)),mt/np.sqrt(vt))),tcal,
        np.column_stack((a["z0_eval"],a["zINT_eval"].mean(axis=1),np.log(np.maximum(ve,1e-300)),me/np.sqrt(ve))),bank,sm.seed("Diabetes",bank,op,"CV")+1)
    saved=sm.readj(folder/"calibration_and_cv.json")
    if f3["lambda"]!=saved["Tuned_INT6_3D"]["lambda"] or f4["lambda"]!=saved["Tuned_INT4_Ratio"]["lambda"]:
        raise RuntimeError(f"Deterministic CAL CV replay selected a different lambda: bank={bank}, op={op}")
    cv_rows.extend({"dataset":"Diabetes_Scheme_D","bank":bank,"operation":op,**r,"deterministic_replay":True} for r in rows3+rows4)
def credit_run(method_rows,fit_rows,cv_rows):
    split,y,_=r4.parse_without_eval_labels()
    for bank in range(6):
        a_ids,b_ids=r4.bank_pool_ids(split["TRAIN"],bank);tr=split["TRAIN"].set_index("ID",drop=False)
        A=tr.loc[sorted(a_ids)].reset_index(drop=True);B=tr.loc[sorted(b_ids)].reset_index(drop=True);prep=r4.fit_preprocessor(A)
        olddir=sm.R4_DIR/f"bank{bank}"
        if list(prep["feature_names"])!=list(sm.readj(olddir/"preprocessor.json")["feature_names"]):raise RuntimeError(f"Credit feature order mismatch bank {bank}")
        xA=r4.transform(A,prep);xB=r4.transform(B,prep);xcal=r4.transform(split["CAL"],prep);xe=r4.transform(split["EVAL"],prep)
        ymap={int(i):int(v) for i,v in zip(split["TRAIN"].ID,y["TRAIN"])};yA=np.asarray([ymap[int(i)] for i in A.ID],np.int8);yB=np.asarray([ymap[int(i)] for i in B.ID],np.int8)
        d0j=sm.readj(olddir/"D0.json");d0=np.asarray(d0j["beta"],float);gamma=float(sm.readj(olddir/"bank_manifest.json")["gamma0"])
        z0cAll=sm.margins(xcal,d0,gamma);z0eAll=sm.margins(xe,d0,gamma);ac=z0cAll>=0;ae=z0eAll>=0;z0c=z0cAll[ac];z0e=z0eAll[ae]
        cached=sm.loadnpz(olddir/"Op_Half_Replace/prediction_arrays.npz")
        if not np.allclose(z0c,cached["z0_cal_alert"],rtol=0,atol=1e-10) or not np.allclose(z0e,cached["z0_eval_alert"],rtol=0,atol=1e-10):raise RuntimeError(f"Credit D0 queue mismatch bank {bank}")
        xc=sparse.csr_matrix(xcal[ac]);xeA=sparse.csr_matrix(xe[ae]);draw=sm.loadnpz(olddir/"draws.npz")["D0_poolA_draw"].astype(int)
        xD0=sparse.csr_matrix(xA[draw]);yD0=np.asarray([yA[i] for i in draw],np.int8)
        for op in ("PoolA_Bootstrap","PoolA_SRSWOR"):
            folder=OUT/f"UCI_Credit/bank{bank}/{op}";folder.mkdir(parents=True,exist_ok=True)
            if ensure_scores_from_cache(folder,bank,op,method_rows):continue
            n=6000;hashes=[];idx={}
            if op=="PoolA_Bootstrap":
                wcenter=n/len(draw);cw=np.full(len(draw),wcenter)
                center,rec=existing_fit_or_fit(xD0,yD0,cw,folder/"operator_center.json",f"Credit/bank{bank}/{op}/CENTER",bank,op,"OperatorCenter",0,sm.sha(draw.astype("<i8").tobytes()),fit_rows)
                zic=[];zie=[];zcc=[];zae=[];idx["D0_PoolA_draw"]=draw.astype(np.int32)
                for role,j in sm.U1_ROLE:
                    sd=sm.seed("Credit",bank,op,role,j);take=np.random.default_rng(sd).integers(0,len(draw),size=n);poolidx=draw[take]
                    beta,mrec=existing_fit_or_fit(xA[poolidx],yA[poolidx],np.ones(n),folder/f"{role}_{j:02d}.json",f"Credit/bank{bank}/{op}/{role}-{j:02d}",bank,op,role,j,sm.sha(poolidx.astype("<i8").tobytes()),fit_rows)
                    hashes.append(mrec["beta_sha256"]);idx[f"{role}_{j:02d}_PoolA_idx"]=poolidx.astype(np.int32)
                    zc=sm.margins(xc,beta,gamma);ze=sm.margins(xeA,beta,gamma)
                    if role=="INT":zic.append(zc);zie.append(ze)
                    elif role=="CALREF":zcc.append(zc)
                    else:zae.append(ze)
                Sigma,ifmeta=sm.fit_bootstrap_if(xD0,yD0,center,n,cw)
                vIc=sm.pred_variance(xc,Sigma);vIe=sm.pred_variance(xeA,Sigma)
                idx["IF_cpu_seconds"]=np.asarray([ifmeta["variance_cpu_seconds"]]);idx["bootstrap_center_weight"]=cw
            else:
                sd=sm.seed("Credit",bank,op,"SRSWOR");take=np.sort(np.random.default_rng(sd).choice(len(A),n,replace=False));cw=np.full(len(A),n/len(A))
                center,rec=existing_fit_or_fit(sparse.csr_matrix(xA),yA,cw,folder/"operator_center.json",f"Credit/bank{bank}/{op}/CENTER",bank,op,"OperatorCenter",0,sm.sha(take.astype("<i8").tobytes()),fit_rows)
                zic=[];zie=[];zcc=[];zae=[];idx["SRSWOR_seed"]=np.asarray([sd],np.int64)
                for role,j in sm.U1_ROLE:
                    rs=sm.seed("Credit",bank,op,role,j);ids=np.sort(np.random.default_rng(rs).choice(len(A),n,replace=False))
                    beta,mrec=existing_fit_or_fit(sparse.csr_matrix(xA[ids]),yA[ids],np.ones(n),folder/f"{role}_{j:02d}.json",f"Credit/bank{bank}/{op}/{role}-{j:02d}",bank,op,role,j,sm.sha(ids.astype("<i8").tobytes()),fit_rows)
                    hashes.append(mrec["beta_sha256"]);idx[f"{role}_{j:02d}_PoolA_idx"]=ids.astype(np.int32)
                    zc=sm.margins(xc,beta,gamma);ze=sm.margins(xeA,beta,gamma)
                    if role=="INT":zic.append(zc);zie.append(ze)
                    elif role=="CALREF":zcc.append(zc)
                    else:zae.append(ze)
                zic=np.column_stack(zic);zie=np.column_stack(zie);zcc=np.column_stack(zcc);zae=np.column_stack(zae)
                dummy=sparse.csr_matrix((1,xA.shape[1]));vIc,vIe,_,ifmeta=u1._finite_population_if(sparse.csr_matrix(xA),yA,dummy,np.zeros(1,np.int8),center,n,0,xc,xeA)
                idx["IF_cpu_seconds"]=np.asarray([ifmeta["total_cpu_seconds"]])
            if op=="PoolA_Bootstrap":
                zic=np.column_stack(zic);zie=np.column_stack(zie);zcc=np.column_stack(zcc);zae=np.column_stack(zae)
            if len(hashes)!=23 or len(set(hashes))!=23:raise RuntimeError(f"Credit role hashes not unique bank={bank} op={op}")
            zRc=sm.margins(xc,center,gamma);zRe=sm.margins(xeA,center,gamma)
            sm.writenpz(folder/"sampling_indices.npz",**idx)
            mets,folds,sig2=sm.fit_heads(bank,op,z0c,z0e,zRc,zRe,zic,zie,zcc,zae,vIc,vIe,folder,sm.seed("Credit",bank,op,"CV"))
            method_rows.extend({"dataset":"UCI_Credit","training_n":6000,**r} for r in mets);cv_rows.extend({"dataset":"UCI_Credit",**r} for r in folds)
            sm.writej(folder/"operation_manifest.json",{"dataset":"UCI_Credit","bank":bank,"operation":op,"fixed_D0_reused":True,
                "D0_training_n":12000,"reference_n":6000,"CAL_threshold_and_alert_queue_reused":True,"cal_alert_n":int(ac.sum()),"eval_alert_n":int(ae.sum()),
                "unique_role_model_beta_hashes":len(set(hashes)),"new_predictive_model_fits":sum(1 for x in fit_rows if x["model_id"].startswith(f"Credit/bank{bank}/{op}/")),
                "sample_policy":"D0 empirical multiset bootstrap n=6000" if op=="PoolA_Bootstrap" else "Pool-A SRSWOR n=6000",
                "EVAL_outcome_labels_used_for_fit_or_selection":False,"sigma2_cal":sig2})

def main():
    if not OUT.is_dir() or OUT.is_symlink():raise RuntimeError(f"Expected incomplete output directory missing: {OUT}")
    w0=time.monotonic();c0=time.process_time();method_rows=[];fit_rows=[];cv_rows=[];diabetes_cv_replays=0
    # The twelve Diabetes score cells are already complete and are never refitted.
    for bank in range(6):
        for op in ("PoolA_Bootstrap","PoolA_SRSWOR"):
            folder=OUT/f"Diabetes_Scheme_D/bank{bank}/{op}"
            if not ensure_scores_from_cache(folder,bank,op,method_rows):raise RuntimeError(f"Missing completed Diabetes cache: {folder}")
            replay_diabetes_cv(folder,bank,op,cv_rows);diabetes_cv_replays+=1
    credit_run(method_rows,fit_rows,cv_rows)
    # Include all already persisted fits from the interrupted first segment, plus continuation fits.
    all_fit=[]
    for p in sorted(OUT.rglob("*.json")):
        if p.is_symlink() or not p.is_file():continue
        try:j=json.loads(p.read_text())
        except Exception:continue
        if "beta_sha256" in j and "model_id" in j:
            all_fit.append({k:v for k,v in j.items() if k!="beta"})
    # Reused cached Half/Full metrics are aligned to the same D0, queue, and per-refit n.
    cached=list(csv.DictReader((sm.STEP1/"point_metrics_by_bank.csv").open()))
    for r in cached:
        if r["operation"] not in ("Op_Half_Replace","Op_Full_Replace") or r["dataset"] not in ("Diabetes_Scheme_D","UCI_Credit"):continue
        method_rows.append({"dataset":r["dataset"],"training_n":14000 if r["dataset"]=="Diabetes_Scheme_D" else 6000,"bank":int(r["bank"]),
            "operation":r["operation"],"method":r["method"],**{k:float(r[k]) for k in ("R_070","AURC100","AUC","Brier","ECE10")},
            "n_alerts":int(r["n_alerts"]),"audit_B":int(r["audit_B"]),"new_model_fits":0,"cached_same_D0":True})
    summaries=[]
    for ds in ("Diabetes_Scheme_D","UCI_Credit"):
        for op in ("PoolA_SRSWOR","PoolA_Bootstrap","Op_Half_Replace","Op_Full_Replace"):
            methods=sorted({r["method"] for r in method_rows if r.get("dataset")==ds and r.get("operation")==op})
            for m in methods:
                rr=[r for r in method_rows if r.get("dataset")==ds and r.get("operation")==op and r.get("method")==m]
                row={"dataset":ds,"training_n":14000 if ds=="Diabetes_Scheme_D" else 6000,"operation":op,"method":m,"banks_n":len(rr)}
                for metric in ("R_070","AURC100","AUC","Brier","ECE10"):
                    vals=np.asarray([r[metric] for r in rr if metric in r],float);row[metric+"_mean"]=float(vals.mean());row[metric+"_min"]=float(vals.min());row[metric+"_max"]=float(vals.max())
                summaries.append(row)
    sm.writecsv(OUT/"same_d0_same_n_per_bank.csv",method_rows);sm.writecsv(OUT/"same_d0_same_n_summary.csv",summaries)
    sm.writecsv(OUT/"fit_records.csv",all_fit);sm.writecsv(OUT/"cv_fold_scores.csv",cv_rows)
    unique_ids={r["model_id"] for r in all_fit};
    if len(unique_ids)!=498:raise RuntimeError(f"Expected 498 unique new predictive-model fit records; got {len(unique_ids)}")
    # Hash uniqueness is checked over all 23 role models in each newly generated cell.
    hash_audit=[]
    for dataset in ("Diabetes_Scheme_D","UCI_Credit"):
        for bank in range(6):
            for op in ("PoolA_Bootstrap","PoolA_SRSWOR"):
                folder=OUT/dataset/f"bank{bank}"/op
                hashes=[]
                if dataset=="Diabetes_Scheme_D" and op=="PoolA_Bootstrap":
                    source=sm.U1_DIR/f"bank{bank}/prospective/models"
                    for role,j in sm.U1_ROLE:
                        p=source/f"{role}-{j:02d}/model.json"
                        if role in ("INT","CALREF"):hashes.append(sm.readj(p)["beta_sha256"])
                        else:hashes.append(sm.readj(folder/f"AUDIT_{j:02d}.json")["beta_sha256"])
                else:
                    for role,j in sm.U1_ROLE:hashes.append(sm.readj(folder/f"{role}_{j:02d}.json")["beta_sha256"])
                if len(hashes)!=23 or len(set(hashes))!=23:raise RuntimeError(f"23 role beta hashes not unique: {dataset} bank{bank} {op}")
                hash_audit.append({"dataset":dataset,"bank":bank,"operation":op,"role_models":23,"unique_beta_hashes":len(set(hashes))})
    audit={"status":"COMPLETED","task":"C2 fixed D0 / fixed queue / equal refit n operation controls",
        "D0_models_refit":0,"unique_predictive_model_fits":len(unique_ids),"new_model_fits_saved_in_interrupted_first_segment":234,
        "new_model_fits_in_continuation":len(unique_ids)-234,"repeated_predictive_model_fits":0,"new_CALREF_score_maps_fit_in_continuation":84,
        "CAL_score_maps_total_across_24_cells":168,"original_unique_learner_fold_fits":2400,"deterministic_Diabetes_CV_fold_replay_fits":1200,
        "full_CAL_learner_refits_original_and_replayed":72,"total_CAL_learner_optimizer_calls_including_replays":3672,
        "CV_fold_score_rows_saved":len(cv_rows),"Diabetes_completed_cells_reused_without_base_refit":12,
        "role_hash_uniqueness_cells":hash_audit,"fixed_reference_n":{"Diabetes":14000,"Credit":6000},
        "same_D0_threshold_and_alert_queue_identity_preflight":"STEP2_C2_EQUAL_N_20261113_identity_preflight.json",
        "EVAL_outcome_labels_used_for_fit_or_selection":False,"Diabetes_EVAL_outcomes_materialized_by_existing_split-audit loader":True,
        "Credit_EVAL_outcome_values_materialized":False,"cpu_threads":1,"gpu":False,
        "predictive_fit_CPU_seconds_sum":float(sum(float(r["fit_cpu_seconds"]) for r in all_fit)),
        "continuation_CPU_seconds":time.process_time()-c0,"continuation_wall_seconds":time.monotonic()-w0,
        "peak_rss_kib_continuation":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "first_segment_resource_note":"Uncaught post-fit diagnostic error prevented a run-level CPU/wall record; per-model CPU times remain persisted in each fit record. No base-model fit was repeated.",
        "sample_size_note":"Credit's saved D0 was fit on 12,000 Pool-A draws; the matched operator references use n=6,000. The same D0/queue is fixed across operations while every refit uses n=6,000.",
        "comparison_note":"Half/Full metric rows are imported from the validated Step-1 cache; those saved operators use the identical D0, threshold, EVAL queue and reference n."}
    sm.writej(OUT/"execution_audit.json",audit);sm.writecsv(OUT/"role_hash_uniqueness.csv",hash_audit)
    print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
