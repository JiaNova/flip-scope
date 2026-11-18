#!/usr/bin/env python3
"""Nested Credit Logistic AUDIT sensitivity B=11/25/50; predictions/maps frozen."""
from __future__ import annotations
import csv,hashlib,json,os,resource,sys,time
from pathlib import Path
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
if hasattr(os,"sched_getaffinity"):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import numpy as np
from scipy import sparse
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.core import run_r4_credit_external as r4
from experiments.core import run_w1_w4 as base
from experiments.core import run_u1_u3 as u1
from experiments.core import run_c1_c4_v28 as c1

OUT=ROOT/"reports/core/STEP2_C3_CREDIT_AUDIT_B50_20261113"
OPS=("Bootstrap","Op_Half_Replace","Op_Full_Replace")
COMPARES={"IF_minus_Center":("OpCenter_IF_1D","OpCenter_Mean_1D"),
          "IF_minus_Tuned3D":("OpCenter_IF_1D","Update_INT6_Tuned3D_MV"),
          "IF_minus_Tuned4":("OpCenter_IF_1D","Tuned_INT4_Ratio")}
B_LEVELS=(11,25,50)
def sha(b):return hashlib.sha256(b).hexdigest()
def wjson(p,obj):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n");os.replace(tmp,p)
def write_or_verify_json(p,obj):
    """Resume-only: accept an existing artifact iff its parsed content is identical."""
    if p.is_symlink():raise RuntimeError(f"refuse symlink resume artifact: {p}")
    if p.exists():
        if not p.is_file() or rjson(p)!=obj:raise RuntimeError(f"resume artifact differs: {p}")
        return False
    wjson(p,obj);return True
def wcsv(p,rows):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.with_name(p.name+".tmp").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(p.with_name(p.name+".tmp"),p)
def wbytes(p,b):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp");tmp.write_bytes(b);os.replace(tmp,p)
def wnpz(p,**a):
    if p.exists() or p.is_symlink():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp.npz");np.savez_compressed(tmp,**a);os.replace(tmp,p)
def read_npz(p):
    if p.is_symlink() or not p.is_file():raise RuntimeError(f"unsafe/missing {p}")
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rjson(p):
    if p.is_symlink() or not p.is_file():raise RuntimeError(f"unsafe/missing {p}")
    return json.loads(p.read_text())
def data_hash_ids(ids):return sha("\n".join(map(str,ids)).encode())
def score_tuned4(arr,fit):
    z0c,z0e=arr["z0_cal_alert"],arr["z0_eval_alert"];ziC,ziE=arr["zINT_cal_alert"],arr["zINT_eval_alert"]
    vc,ve=arr["vtilde_cal_alert"],arr["vtilde_eval_alert"]
    bc,be=ziC.mean(axis=1),ziE.mean(axis=1);mc=.5*(z0c+bc);me=.5*(z0e+be)
    xc=np.column_stack((z0c,bc,np.log(np.maximum(vc,1e-300)),mc/np.sqrt(vc)))
    xe=np.column_stack((z0e,be,np.log(np.maximum(ve,1e-300)),me/np.sqrt(ve)))
    return u1._predict_brier(fit,xe)
def paired_metric(q1,q0,audit,metric):
    t=audit.mean(axis=1);one=np.ones(len(t))
    if metric=="Brier":return float(np.mean((q1[:,None]-audit)**2)-np.mean((q0[:,None]-audit)**2))
    m1=c1._WeightedRisk(q1).metrics(t,one);m0=c1._WeightedRisk(q0).metrics(t,one)
    return float(m1[metric]-m0[metric])

def main():
    if OUT.is_symlink():raise RuntimeError(f"refuse symlink output: {OUT}")
    if not OUT.exists():OUT.mkdir(parents=True,exist_ok=False)
    elif not OUT.is_dir():raise RuntimeError(f"unsafe output path: {OUT}")
    wall0=time.monotonic();cpu0=time.process_time();new_fit_count=0;resumed_fit_count=0
    split,y,audit=r4.parse_without_eval_labels();fit_records=[];pred_rows=[];delta_rows=[];bank_events=[]
    for bank in range(6):
        bankdir=OUT/f"bank{bank}"
        if bankdir.is_symlink():raise RuntimeError(f"refuse symlink bank dir: {bankdir}")
        bankdir.mkdir(exist_ok=True)
        a_ids,b_ids=r4.bank_pool_ids(split["TRAIN"],bank)
        train=split["TRAIN"].set_index("ID",drop=False);A=train.loc[sorted(a_ids)].reset_index(drop=True);B=train.loc[sorted(b_ids)].reset_index(drop=True)
        ymap={int(i):int(v) for i,v in zip(split["TRAIN"].ID,y["TRAIN"])}
        yA=np.asarray([ymap[int(i)] for i in A.ID],np.int8);yB=np.asarray([ymap[int(i)] for i in B.ID],np.int8)
        prep=r4.fit_preprocessor(A);xA=r4.transform(A,prep);xB=r4.transform(B,prep)
        xcal=r4.transform(split["CAL"],prep);xeva=r4.transform(split["EVAL"],prep)
        savedprep=rjson(r4.RUN_DIR/f"bank{bank}/preprocessor.json")
        if prep["feature_names"]!=savedprep["feature_names"] or prep["means"]!=savedprep["numeric_means"] or prep["scales"]!=savedprep["numeric_scales"]:
            raise RuntimeError(f"reconstructed Pool-A preprocessor differs from frozen R4 bank={bank}")
        split_bank=rjson(r4.RUN_DIR/f"bank{bank}/bank_manifest.json");gamma=float(split_bank["gamma0"])
        d0draw=read_npz(r4.RUN_DIR/f"bank{bank}/draws.npz")["D0_poolA_draw"].astype(np.int64)
        label={};fitinfo={}
        for op in OPS:
            oldarr=read_npz(r4.RUN_DIR/f"bank{bank}/{op}/prediction_arrays.npz")
            # Frozen D0 alert queue and CAL mapping predictions.
            z0c_all=np.asarray(xcal@np.asarray(rjson(r4.RUN_DIR/f"bank{bank}/D0.json")["beta"],float)).reshape(-1)-gamma
            z0e_all=np.asarray(xeva@np.asarray(rjson(r4.RUN_DIR/f"bank{bank}/D0.json")["beta"],float)).reshape(-1)-gamma
            ac=z0c_all>=0;ae=z0e_all>=0;idx=np.flatnonzero(ae)
            if len(idx)!=len(oldarr["audit_eval_alert"]):raise RuntimeError(f"frozen D0 alert count mismatch {bank}/{op}")
            if not np.allclose(z0c_all[ac],oldarr["z0_cal_alert"],rtol=0,atol=2e-9) or not np.allclose(z0e_all[ae],oldarr["z0_eval_alert"],rtol=0,atol=2e-9):
                raise RuntimeError(f"frozen D0 score/row order mismatch {bank}/{op}")
            zAUDIT=[oldarr["zAUDIT_eval_alert"][:,j].copy() for j in range(11)]
            new_models=[];draw_arrays={"D0_poolA_draw":d0draw.astype(np.int32)}
            for j in range(12,51):
                role="AUDIT";seed=r4.seed(bank,op,role,j);rng=np.random.default_rng(seed)
                if op=="Bootstrap":
                    take=rng.integers(0,len(d0draw),size=12000);ia=d0draw[take];ib=np.empty(0,np.int64)
                    weights=np.bincount(ia,minlength=len(A)).astype(float);xx=xA;yy=yA;ww=weights
                    draw_hash=sha(weights.astype("<f8").tobytes())
                elif op=="Op_Half_Replace":
                    ia=np.sort(rng.choice(len(A),3000,replace=False));ib=np.sort(rng.choice(len(B),3000,replace=False))
                    xx=sparse.vstack((xA[ia],xB[ib]),format="csr");yy=np.r_[yA[ia],yB[ib]];ww=np.ones(len(yy))
                    draw_hash=sha((data_hash_ids(A.ID.iloc[ia])+data_hash_ids(B.ID.iloc[ib])).encode())
                else:
                    ia=np.empty(0,np.int64);ib=np.sort(rng.choice(len(B),6000,replace=False))
                    xx=xB[ib];yy=yB[ib];ww=np.ones(len(yy));draw_hash=data_hash_ids(B.ID.iloc[ib])
                model_id=f"bank{bank}/{op}/AUDIT-{j:02d}"
                model_path=bankdir/f"{op}_AUDIT_{j:02d}.json"
                if model_path.is_symlink():raise RuntimeError(f"refuse symlink model: {model_path}")
                if model_path.exists():
                    saved=rjson(model_path);beta=np.asarray(saved["beta"],float)
                    if (saved.get("model_id")!=model_id or saved.get("seed")!=int(seed)
                        or saved.get("draw_digest")!=draw_hash or saved.get("train_n")!=len(yy)
                        or saved.get("beta_sha256")!=sha(beta.astype("<f8").tobytes())):
                        raise RuntimeError(f"saved AUDIT model does not match reconstructed draw: {model_path}")
                    solver=saved.get("solver",{});fitcpu=float(saved.get("fit_cpu_seconds",0.0));resumed_fit_count+=1
                    fit_status="reused_after_interruption"
                else:
                    # Fit the original R4 penalized-logistic objective; no new calibration is fitted.
                    t0=time.process_time();beta,solver=base._fit_irls(xx.tocsr(),yy.astype(float),ww.astype(float),ridge=1.0)
                    fitcpu=time.process_time()-t0;new_fit_count+=1;fit_status="new_fit"
                    payload={"model_id":model_id,"bank":bank,"operation":op,"role":"AUDIT","replicate":j,
                        "seed":int(seed),"train_n":int(len(yy)),"fit_cpu_seconds":fitcpu,"draw_digest":draw_hash,
                        "beta_sha256":sha(beta.astype("<f8").tobytes()),"solver":solver,"beta":beta.tolist()}
                    wjson(model_path,payload)
                fit_records.append({"model_id":model_id,"bank":bank,"operation":op,"role":"AUDIT","replicate":j,
                    "seed":int(seed),"train_n":int(len(yy)),"fit_cpu_seconds":fitcpu,"draw_digest":draw_hash,
                    "beta_sha256":sha(beta.astype("<f8").tobytes()),"fit_status":fit_status})
                z=np.asarray(xeva@beta).reshape(-1)[idx]-gamma;zAUDIT.append(z)
                new_models.append(model_id);draw_arrays[f"AUDIT_{j:02d}_Aidx"]=ia.astype(np.int32);draw_arrays[f"AUDIT_{j:02d}_Bidx"]=ib.astype(np.int32)
            zAUDITmat=np.column_stack(zAUDIT);wnpz(bankdir/f"{op}_AUDIT_sampling_indices.npz",**draw_arrays)
            frozen=qexisting={}
            names=[str(v) for v in oldarr["scored_method_names"].tolist()];scored={n:oldarr["scored"][:,i] for i,n in enumerate(names)}
            fit4=rjson(ROOT/"reports/core/STEP1_FAIR_BASELINES_20261113/fits/UCI_Credit"/op/f"bank{bank}.json")["Tuned_INT4_Ratio_fit"]
            q4=score_tuned4(oldarr,fit4)
            frozen={"OpCenter_IF_1D":scored["OpCenter_IF_1D"],"OpCenter_Mean_1D":scored["OpCenter_Mean_1D"],
                "Update_INT6_Tuned3D_MV":scored["Update_INT6_Tuned3D_MV"],"Tuned_INT4_Ratio":q4}
            bank_events.append({"bank":bank,"operation":op,"B11_event_count":int(oldarr["audit_eval_alert"].sum()),
                "B50_event_count":int(zAUDITmat.shape[1] and np.sum(zAUDITmat<0)),"B50_event_trials":int(zAUDITmat.size)})
            for Bn in B_LEVELS:
                am=(zAUDITmat[:,:Bn]<0).astype(np.int8)
                for method,qv in frozen.items():
                    met=u1._metrics(qv,am)
                    pred_rows.append({"bank":bank,"operation":op,"audit_B":Bn,"method":method,**met,
                        "observed_flip_events":int(am.sum()),"flip_trials":int(am.size),"eval_alert_n":len(am)})
                for pair,(lhs,rhs) in COMPARES.items():
                    for metric in ("R_070","AURC100","Brier"):
                        delta=paired_metric(frozen[lhs],frozen[rhs],am,metric)
                        delta_rows.append({"bank":bank,"operation":op,"audit_B":Bn,"pair":pair,
                            "left":lhs,"right":rhs,"metric":metric,"delta":delta,
                            "observed_flip_events":int(am.sum()),"flip_trials":int(am.size)})
            wnpz(bankdir/f"{op}_audit_targets_B11_25_50.npz",zAUDIT_eval_alert=zAUDITmat,
                audit_B11=(zAUDITmat[:,:11]<0).astype(np.int8),audit_B25=(zAUDITmat[:,:25]<0).astype(np.int8),audit_B50=(zAUDITmat<0).astype(np.int8),
                frozen_q_names=np.asarray(sorted(frozen),dtype="U64"),frozen_q=np.column_stack([frozen[k] for k in sorted(frozen)]))
            write_or_verify_json(bankdir/f"{op}_audit_extension_manifest.json",{"bank":bank,"operation":op,"old_AUDIT_models_reused":11,
                "new_AUDIT_model_fits":39,"nested_B_values":[11,25,50],"same_D0_INT_CALREF_and_CAL_maps_frozen":True,
                "new_AUDIT_seed_rule":"R4.seed(bank, operation, 'AUDIT', j), j=12..50","EVAL_labels_used":False,
                "new_model_ids":new_models})
        print(f"Credit nested AUDIT bank {bank} finished; added models=117",flush=True)
    summaries=[]
    for (op,Bn,pair,metric) in sorted({(r["operation"],r["audit_B"],r["pair"],r["metric"]) for r in delta_rows}):
        rr=[r for r in delta_rows if (r["operation"],r["audit_B"],r["pair"],r["metric"])==(op,Bn,pair,metric)]
        vals=np.asarray([r["delta"] for r in rr],float)
        summaries.append({"operation":op,"audit_B":Bn,"pair":pair,"metric":metric,"mean_delta_across_banks":float(vals.mean()),
            "min":float(vals.min()),"max":float(vals.max()),"wins_negative_out_of_6":int(np.sum(vals<0)),"banks_n":len(vals)})
    wcsv(OUT/"audit_sensitivity_per_bank.csv",pred_rows);wcsv(OUT/"audit_paired_deltas_by_bank.csv",delta_rows)
    wcsv(OUT/"audit_sensitivity_summary.csv",summaries);wcsv(OUT/"observed_event_counts.csv",bank_events);wcsv(OUT/"new_fit_records.csv",fit_records)
    payload={"status":"COMPLETED_NESTED_AUDIT_SENSITIVITY","dataset":"UCI Credit R4 logistic, saved scores/CAL maps frozen",
        "new_AUDIT_logistic_fits_this_resume":new_fit_count,"resumed_AUDIT_logistic_fits":resumed_fit_count,
        "unique_AUDIT_logistic_fits_for_this_task":new_fit_count+resumed_fit_count,
        "B_levels":[11,25,50],"bank_operation_cells":18,
        "new_fits_per_bank_operation":39,"total_AUDIT_model_counts_per_cell":{"B11":11,"B25":25,"B50":50},
        "no_D0_INT_CALREF_or_map_refits":True,"eval_labels_used":False,
        "cpu_seconds":time.process_time()-cpu0,"wall_seconds":time.monotonic()-wall0,
        "peak_rss_platform_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"peak_rss_units":"KiB on Linux"}
    wjson(OUT/"execution_audit.json",payload);print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
