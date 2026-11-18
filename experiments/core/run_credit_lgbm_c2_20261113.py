#!/usr/bin/env python3
"""Credit LightGBM cross-operator replay under the frozen R4 split/draw plan."""
from __future__ import annotations
import csv,hashlib,json,os,resource,sys,time
from pathlib import Path
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
if hasattr(os,"sched_getaffinity"):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import lightgbm as lgb
import numpy as np
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.core import run_r4_credit_external as r4
from experiments.core import run_u1_u3 as u1
from experiments.core import run_c1_c4_v28 as c1

OUT=ROOT/"reports/core/STEP2_C2_CREDIT_LGBM_20261113"
OPS=("Bootstrap","Op_Half_Replace","Op_Full_Replace")
ROLE_SPECS=[("INT",i) for i in range(1,7)]+[("CALREF",i) for i in range(1,7)]+[("AUDIT",i) for i in range(1,12)]

def sha(b):return hashlib.sha256(b).hexdigest()
def rjson(p):
    if p.is_symlink() or not p.is_file():raise RuntimeError(f"unsafe/missing {p}")
    return json.loads(p.read_text())
def wjson(p,obj):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n");os.replace(tmp,p)
def wcsv(p,rows):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for x in rows for k in x))
    tmp=p.with_name(p.name+".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,p)
def wbytes(p,data):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp");tmp.write_bytes(data);os.replace(tmp,p)
def wnpz(p,**arrays):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp.npz")
    if tmp.exists() or tmp.is_symlink():raise FileExistsError(tmp)
    np.savez_compressed(tmp,**arrays);os.replace(tmp,p)
def params(seed):
    p={"objective":"binary","metric":"binary_logloss","learning_rate":.05,"num_leaves":15,"max_depth":5,
       "min_data_in_leaf":40,"lambda_l2":1.,"feature_fraction":1.,"bagging_fraction":1.,"bagging_freq":0,
       "deterministic":True,"force_col_wise":True,"num_threads":1,"device_type":"cpu","verbosity":-1,
       "seed":int(seed),"data_random_seed":int(seed),"feature_fraction_seed":int(seed),"bagging_seed":int(seed),"drop_seed":int(seed)}
    return p
def fit_lgb(x,y,w,seed,model_path,model_meta):
    startc=time.process_time();startw=time.monotonic()
    ds=lgb.Dataset(x,label=y,weight=w,free_raw_data=True)
    booster=lgb.train(params(seed),ds,num_boost_round=100)
    raw=booster.model_to_string().encode()
    meta={**model_meta,"seed":int(seed),"fit_cpu_seconds":time.process_time()-startc,
          "fit_wall_seconds":time.monotonic()-startw,"lightgbm_version":lgb.__version__,"num_boost_round":100,
          "thread_count":1,"device":"cpu","model_sha256":sha(raw)}
    wbytes(model_path,raw)
    wjson(model_path.with_suffix(".json"),meta)
    return booster,meta
def zscore(model,x,gamma):
    p=np.clip(np.asarray(model.predict(x,num_threads=1),float),1e-12,1-1e-12)
    return logit(p)-gamma
def data_hash_ids(ids):return sha("\n".join(map(str,ids)).encode())

def main():
    if OUT.exists() or OUT.is_symlink():raise FileExistsError(f"create-only output exists: {OUT}")
    OUT.mkdir(parents=True,exist_ok=False);startw=time.monotonic();startc=time.process_time()
    split,y,audit=r4.parse_without_eval_labels()
    all_rows=[];variance_rows=[];fit_rows=[];cv_rows=[];prediction_count=0
    source_sha=r4.file_sha(r4.DATA_PATH)
    for bank in range(6):
        bdir=OUT/f"bank{bank}";bdir.mkdir()
        a_ids,b_ids=r4.bank_pool_ids(split["TRAIN"],bank)
        train=split["TRAIN"].set_index("ID",drop=False)
        A=train.loc[sorted(a_ids)].reset_index(drop=True);B=train.loc[sorted(b_ids)].reset_index(drop=True)
        prep=r4.fit_preprocessor(A)
        xA=r4.transform(A,prep);xB=r4.transform(B,prep);xc_all=r4.transform(split["CAL"],prep);xe_all=r4.transform(split["EVAL"],prep)
        ym={int(i):int(v) for i,v in zip(split["TRAIN"].ID,y["TRAIN"])}
        yA=np.asarray([ym[int(i)] for i in A.ID],np.int8);yB=np.asarray([ym[int(i)] for i in B.ID],np.int8)
        ycal=np.asarray([int(v) for v in y["CAL"]],np.int8)
        with np.load(r4.RUN_DIR/f"bank{bank}/draws.npz",allow_pickle=False) as dz:
            saved_d0=dz["D0_poolA_draw"].astype(np.int64)
            if not np.array_equal(dz["poolA_ids"],A.ID.to_numpy()) or not np.array_equal(dz["poolB_ids"],B.ID.to_numpy()):
                raise RuntimeError(f"Pool ID order mismatch bank={bank}")
        d0seed=r4.seed(bank,"D0","sample")
        expected=np.random.default_rng(d0seed).choice(np.arange(len(A)),size=12000,replace=True)
        if not np.array_equal(saved_d0,expected):raise RuntimeError(f"D0 exact draw mismatch bank={bank}")
        d0path=bdir/"D0.txt"
        beta0,meta0=fit_lgb(xA[saved_d0],yA[saved_d0],np.ones(12000),d0seed,d0path,
            {"model_id":f"Credit_LGBM/bank{bank}/D0","bank":bank,"role":"D0","train_n":12000,
             "sampling":"same Pool-A bootstrap D0 draw as R4 logistic","training_id_sha256":data_hash_ids(A.ID.iloc[saved_d0])})
        fit_rows.append(meta0)
        z0c_all=zscore(beta0,xc_all,0.);pcal=np.clip(expit(z0c_all),1e-12,1-1e-12)
        tau=float(np.quantile(pcal,.80));gamma=float(logit(np.clip(tau,1e-12,1-1e-12)))
        z0c_all-=gamma;z0e_all=zscore(beta0,xe_all,gamma)
        ac=z0c_all>=0;ae=z0e_all>=0;idx=np.flatnonzero(ae)
        z0c=z0c_all[ac];z0e=z0e_all[ae];xcal=xc_all[ac];xe=xe_all[ae]
        wjson(bdir/"preprocessor.json",{"source":"R4 frozen Credit preprocessor, fit on Pool A only","feature_names":prep["feature_names"],
            "numeric_means":prep["means"],"numeric_scales":prep["scales"],"categorical_maps":prep["cat_maps"],
            "split_id_hashes":audit["split_id_sha256"],"eval_label_values_materialized":False})
        wjson(bdir/"bank_manifest.json",{"bank":bank,"poolA_n":len(A),"poolB_n":len(B),"feature_dimension":len(prep["feature_names"]),
            "tau0":tau,"gamma0":gamma,"cal_alert_n":int(ac.sum()),"eval_alert_n":int(ae.sum()),
            "D0_model_sha256":meta0["model_sha256"],"exact_R4_pool_partition_reused":True,"eval_labels_used":False})
        for op in OPS:
            odir=bdir/op;odir.mkdir();zRcal=z0c.copy();zReval=z0e.copy()
            zIc=[];zIe=[];zCc=[];zAe=[];model_meta=[];sample_arrays={"D0_poolA_draw":saved_d0}
            for role,j in ROLE_SPECS:
                drawseed=r4.seed(bank,op,role,j)
                oldmeta=rjson(r4.RUN_DIR/f"bank{bank}/{op}/models/{role}-{j:02d}.json")
                if op=="Bootstrap":
                    rng=np.random.default_rng(drawseed);pos=rng.integers(0,len(saved_d0),size=12000)
                    ia=saved_d0[pos];ib=np.empty(0,np.int64)
                    xx=xA[ia];yy=yA[ia];ww=np.ones(len(ia),float)
                    mult=np.bincount(ia,minlength=len(A)).astype(np.int64)
                    if sha(mult.astype("<f8").tobytes())!=oldmeta["poolA_multiplicity_sha256"]:
                        raise RuntimeError(f"bootstrap multiplicity mismatch {bank}/{role}-{j}")
                else:
                    rng=np.random.default_rng(drawseed)
                    if op=="Op_Half_Replace":
                        ia=np.sort(rng.choice(len(A),3000,replace=False));ib=np.sort(rng.choice(len(B),3000,replace=False))
                    else:
                        ia=np.empty(0,np.int64);ib=np.sort(rng.choice(len(B),6000,replace=False))
                    hashA=data_hash_ids(A.ID.iloc[ia]) if len(ia) else sha(b"")
                    hashB=data_hash_ids(B.ID.iloc[ib]) if len(ib) else sha(b"")
                    if hashA!=oldmeta["PoolA_ids_sha256"] or hashB!=oldmeta["PoolB_ids_sha256"]:
                        raise RuntimeError(f"update exact draw mismatch {bank}/{op}/{role}-{j}")
                    xx=__import__('scipy').sparse.vstack((xA[ia],xB[ib]),format="csr")
                    yy=np.r_[yA[ia],yB[ib]];ww=np.ones(len(yy),float)
                mid=f"{role}-{j:02d}"
                smeta={"model_id":f"Credit_LGBM/bank{bank}/{op}/{mid}","bank":bank,"operation":op,"role":role,
                    "replicate":j,"draw_seed":drawseed,"train_n":len(yy),"sampling_source":"exact R4 logistic draw indices/multiplicity",
                    "PoolA_index_sha256":sha(np.asarray(ia,dtype="<i8").tobytes()),"PoolB_index_sha256":sha(np.asarray(ib,dtype="<i8").tobytes()),
                    "original_R4_logistic_sample_weight_sha256":oldmeta["sample_weight_sha256"]}
                model,pathmeta=fit_lgb(xx,yy,ww,drawseed,odir/f"{mid}.txt",smeta);model_meta.append(pathmeta);fit_rows.append(pathmeta)
                zc=zscore(model,xcal,gamma);ze=zscore(model,xe,gamma);prediction_count+=2
                sample_arrays[f"{mid}_PoolA_idx"]=ia.astype(np.int32)
                sample_arrays[f"{mid}_PoolB_idx"]=ib.astype(np.int32)
                if role=="INT":zIc.append((zc,ze))
                elif role=="CALREF":zCc.append(zc)
                else:zAe.append(ze)
            zIC=np.column_stack([p[0] for p in zIc]);zIE=np.column_stack([p[1] for p in zIc])
            zCC=np.column_stack(zCc);zAE=np.column_stack(zAe);audit_mat=(zAE<0).astype(np.int8);tcal=(zCC<0).mean(axis=1)
            zbarc=zIC.mean(axis=1);zbare=zIE.mean(axis=1);s2c=np.var(zIC,axis=1,ddof=1);s2e=np.var(zIE,axis=1,ddof=1)
            sigma2=float(s2c.mean());vtc=.5*s2c+.5*sigma2;vte=.5*s2e+.5*sigma2;mc=.5*(z0c+zbarc);me=.5*(z0e+zbare)
            rawcal={"Margin":z0c,"Fused_Center_Only":mc,"Variance_Only_qV":z0c/np.sqrt(vtc),"qMV":mc/np.sqrt(vtc)}
            rawev={"Margin":z0e,"Fused_Center_Only":me,"Variance_Only_qV":z0e/np.sqrt(vte),"qMV":me/np.sqrt(vte)}
            q={};maps={}
            for name,score in rawcal.items():
                mapping=c1._fit_score_map(score,tcal);maps[name]=mapping;q[name]=c1._apply_score_map(mapping,rawev[name])
            x3c=np.column_stack((z0c,zbarc,np.log(np.maximum(vtc,1e-300))));x3e=np.column_stack((z0e,zbare,np.log(np.maximum(vte,1e-300))))
            fit3,q3,fold3=u1._cv_brier("Tuned_INT6_3D",x3c,tcal,x3e,bank,r4.seed(bank,op,"CV",3));q["Tuned_INT6_3D"]=q3
            x4c=np.column_stack((z0c,zbarc,np.log(np.maximum(vtc,1e-300)),mc/np.sqrt(vtc)))
            x4e=np.column_stack((z0e,zbare,np.log(np.maximum(vte,1e-300)),me/np.sqrt(vte)))
            fit4,q4,fold4=u1._cv_brier("Tuned_INT4_Ratio",x4c,tcal,x4e,bank,r4.seed(bank,op,"CV",4));q["Tuned_INT4_Ratio"]=q4
            for f in (fold3,fold4):
                cv_rows.extend({"bank":bank,"operation":op,**r} for r in f)
            wjson(odir/"calibration_fits.json",{"maps":maps,"Tuned_INT6_3D":fit3,"Tuned_INT4_Ratio":fit4,
                "CALREF_target_sha256":sha(tcal.astype("<f8").tobytes()),"selected_lambda_grid":list(u1.LAMBDA_GRID),
                "CV_patient_level":True,"EVAL_target_used":False})
            wcsv(odir/"sampling_hashes.csv",[{"model":r["model_id"],"sampling_seed":r["draw_seed"],
                "PoolA_index_sha256":r["PoolA_index_sha256"],"PoolB_index_sha256":r["PoolB_index_sha256"],
                "matching_logistic_draw_verified":True} for r in model_meta])
            wnpz(odir/"sampling_indices.npz",**sample_arrays)
            np.savez_compressed(odir/"prediction_arrays.npz",alert_idx=idx.astype(np.int32),
                z0_cal_alert=z0c,z0_eval_alert=z0e,zINT_cal_alert=zIC,zINT_eval_alert=zIE,
                zCALREF_cal_alert=zCC,zAUDIT_eval_alert=zAE,audit_eval_alert=audit_mat,
                vtilde_cal_alert=vtc,vtilde_eval_alert=vte,sigma_g2=np.asarray(sigma2),
                q_scored=np.column_stack([q[k] for k in sorted(q)]),q_method_names=np.asarray(sorted(q),dtype="U64"))
            for name,prob in q.items():
                met=u1._metrics(prob,audit_mat)
                all_rows.append({"bank":bank,"operation":op,"method":name,**met,"sigma_g2":sigma2,
                    "eval_alert_n":len(audit_mat),"cal_alert_n":len(tcal),"eval_alert_rate":len(audit_mat)/6000,
                    "selected_lambda":fit4["lambda"] if name=="Tuned_INT4_Ratio" else fit3["lambda"] if name=="Tuned_INT6_3D" else None})
            vif=None
            variance_rows.append({"bank":bank,"operation":op,"n_eval_alert":len(audit_mat),"mean_audit_flip_rate":float(audit_mat.mean()),
                "mean_INT_sample_variance":float(s2e.mean()),"sigma_g2_cal":sigma2,
                "AUDIT_var_vs_INT_var_ratio_mean":float(np.mean(np.var(zAE,axis=1,ddof=1))/max(np.mean(s2e),1e-300)),
                "D0_threshold_tau":tau,"D0_alert_rate":len(audit_mat)/6000})
            manifest={"bank":bank,"operation":op,"prediction_model_fit_count_including_shared_D0":23,
                "operator_center_fit_count":0,"sample_size":12000 if op=="Bootstrap" else 6000,
                "models":model_meta,"D0_model_sha256":meta0["model_sha256"],"tau0":tau,"gamma0":gamma,
                "eval_labels_used":False,"role_model_hashes_unique":len({r["model_sha256"] for r in model_meta}),
                "CV_learner_count":2,"CV_search_fit_count":100,"CV_final_fit_count":2}
            wjson(odir/"operation_manifest.json",manifest)
        wcsv(bdir/"bank_thresholds.csv",[{"bank":bank,"tau0":tau,"cal_alert_n":int(ac.sum()),
            "eval_alert_n":int(ae.sum()),"eval_alert_rate":len(idx)/6000}])
        print(f"Credit LightGBM bank {bank} done; eval alerts={len(idx)}",flush=True)
    # Six-bank descriptive summaries and paired win counts against Margin.
    summ=[]
    for op in OPS:
        for method in sorted({r["method"] for r in all_rows if r["operation"]==op}):
            rr=[r for r in all_rows if r["operation"]==op and r["method"]==method]
            row={"operation":op,"method":method,"banks_n":len(rr)}
            for metric in ("R_070","AURC100","AUC","Brier","ECE10"):
                v=np.asarray([x[metric] for x in rr],float);row[metric+"_mean"]=float(v.mean());row[metric+"_min"]=float(v.min());row[metric+"_max"]=float(v.max())
            if method!="Margin":
                for metric in ("R_070","AURC100","Brier","ECE10"):
                    dif=[]
                    for b in range(6):
                        m=next(x for x in all_rows if x["bank"]==b and x["operation"]==op and x["method"]==method)
                        ref=next(x for x in all_rows if x["bank"]==b and x["operation"]==op and x["method"]=="Margin")
                        dif.append(float(m[metric])-float(ref[metric]))
                    row[f"delta_{metric}_vs_Margin_mean"]=float(np.mean(dif));row[f"wins_vs_Margin_{metric}_out_of_6"]=int(sum(x<0 for x in dif))
            summ.append(row)
    wcsv(OUT/"per_bank_metrics.csv",all_rows);wcsv(OUT/"mean_range_metrics.csv",summ);wcsv(OUT/"variance_closure.csv",variance_rows);wcsv(OUT/"cv_fold_scores.csv",cv_rows)
    audit_out={"status":"COMPLETED","dataset":"UCI Credit Default; no eval labels read","source_sha256":source_sha,
        "partition_and_draw_source":"exact existing R4 pool IDs, D0 bootstrap rows, and per-role sample indices verified against logistic model manifests",
        "base_model_fits":len(fit_rows),"shared_D0_fits":6,"per_operation_role_fits":23,"operator_center_fits":0,
        "D0_fit_count":6,"INT_fits":108,"CALREF_fits":108,"AUDIT_fits":198,
        "CALREF_maps":6*3*4,"Tuned_INT3D_and_INT4_CV_search_fits":len(cv_rows),
        "CV_final_fits":6*3*2,"prediction_batches":prediction_count,"eval_label_values_materialized":False,
        "lightgbm_version":lgb.__version__,"boosting_rounds":100,"threads":1,"gpu_used":False,
        "cpu_seconds":time.process_time()-startc,"wall_seconds":time.monotonic()-startw,
        "peak_rss_platform_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"peak_rss_units":"KiB on Linux"}
    wjson(OUT/"execution_audit.json",audit_out)
    print(json.dumps(audit_out,indent=2))

if __name__=="__main__":main()
