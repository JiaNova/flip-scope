"""Public-data follow-up: independent calibration references and variable P2 draws.

Original results remain intact. All evaluations here are exploratory reuse of
existing public EVAL cohorts. No approval files or frozen-hash checks are used.
"""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.public_benchmarks import run_public_experiments as old

OUT = ROOT / "reports/public_benchmarks/followup"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old._json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def loss(theta, x, t):
    p = expit(theta[0]+theta[1]*x)
    residual = 2*(p-t)*p*(1-p)
    return float(np.mean((p-t)**2)+1e-6*theta[1]**2), np.array([
        residual.mean(), (residual*x).mean()+2e-6*theta[1]])


def calibrate(xc, tc, xe):
    center, scale = float(np.mean(xc)), max(float(np.std(xc)), 1e-8)
    x = (xc-center)/scale
    result = minimize(loss, [float(logit(np.clip(np.mean(tc), 1e-6, 1-1e-6))), -1.],
                      args=(x, tc), jac=True, method="L-BFGS-B", bounds=[(-30, 30), (-30, 0)],
                      options={"maxiter":500, "ftol":1e-13, "gtol":1e-9})
    if not result.success: raise RuntimeError(result.message)
    q = expit(result.x[0]+result.x[1]*(xe-center)/scale)
    return q, {"center":center, "scale":scale, "coefficients":result.x.tolist(),
               "CAL_MSE_to_reference_frequency":float(result.fun), "success":True}


def evaluate(bank, operation, features, raw, tc, audit):
    calibrated, maps = {}, {}
    for method, (xc, xe) in features.items():
        calibrated[method], maps[method] = calibrate(xc, tc, xe)
    scores = {**calibrated, **raw}
    baseline = calibrated["margin_calibrated"]
    target = audit.mean(axis=1)
    rows = [{"bank_id":bank, "operation":operation, "method":method,
             **old._risk_metrics(q, target, audit, baseline, baseline)} for method, q in scores.items()]
    return rows, maps, scores


def role_scores(zc, ze, ic, ie, sigma2, if_pair=None, surrogate=None):
    mc, me = ic.mean(axis=1), ie.mean(axis=1)
    vc = .5*np.var(ic, axis=1, ddof=1)+.5*sigma2
    ve = .5*np.var(ie, axis=1, ddof=1)+.5*sigma2
    uc, ue = (.5*zc+.5*mc)/np.sqrt(vc), (.5*ze+.5*me)/np.sqrt(ve)
    features = {"margin_calibrated":(zc,ze), "int_mean_calibrated":(mc,me),
                "int_frequency_calibrated":(-(ic<0).sum(axis=1),-(ie<0).sum(axis=1)),
                "qV_calibrated":(zc/np.sqrt(vc),ze/np.sqrt(ve)), "qMV_calibrated":(uc,ue)}
    raw = {"qMV_raw_probability":ndtr(-ue), "qV_raw_probability":ndtr(-ze/np.sqrt(ve)),
           "q_global_1":ndtr(-ze/np.sqrt(sigma2)), "q_sqrt2":ndtr(-ze/np.sqrt(2*sigma2))}
    if if_pair is not None:
        features["IF_conditional_calibrated"] = if_pair
        raw["IF_conditional_raw_probability"] = ndtr(-if_pair[1])
    if surrogate is not None: features["surrogate_calibrated"] = surrogate
    return features, raw


def run_p3():
    folder = OUT/"p3_independent_calibration"; folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); cpu = time.process_time(); fits = 0; rows = []
    df = pd.read_csv(old.DATA_DIR/"diabetic_data.csv", dtype=str, keep_default_na=False, low_memory=False)
    selected, data_audit = old._parse_diabetes_rows(df)
    cal = selected[selected.split=="CAL"].reset_index(drop=True)
    evaluation = selected[selected.split=="EVAL"].reset_index(drop=True)
    train = selected[selected.split=="TRAIN"]
    assert set(cal.patient_nbr).isdisjoint(evaluation.patient_nbr)
    summaries = []
    for bank in range(3):
        work = folder/f"bank{bank}"; work.mkdir(exist_ok=True)
        prep = old.read_json(old._p3_bank_dir(bank)/"preprocessor/preprocessor.json")
        bt = train[train.bank_id==bank]
        train_x = old.transform_diabetes_lgb(bt, prep); labels = bt.y_readmit30.to_numpy(dtype=float)
        cal_x = old.transform_diabetes_lgb(cal, prep)
        with np.load(old._p3_bank_dir(bank)/"prediction_cache/arrays.npz", allow_pickle=False) as a:
            ncal = int(a["cal_count"][0]); assert ncal==len(cal)
            p0 = a["p_D0"].copy()
            ints = np.column_stack([a[f"p_INT-{j:02d}"] for j in range(1,7)])
            audits = np.column_stack([a[f"p_AUDIT-{j:02d}"] for j in range(1,12)])
        gamma = float(logit(np.quantile(p0[:ncal], .8)))
        z0 = old.safe_logit(p0)-gamma; zi = old.safe_logit(ints)-gamma
        c, e = z0[:ncal]>=0, z0[ncal:]>=0
        # Check cached patient/row pairing against the published per-patient output.
        published = pd.read_csv(old.OUT_ROOT/f"p3_three_bank_results/bank{bank}_eval_alert_scores.csv")
        keys = evaluation.loc[e,"patient_nbr"].map(lambda p:old.sha_text("patient:"+str(p))).tolist()
        assert keys==published.patient_key_sha256.tolist()
        yaudit = (old.safe_logit(audits[ncal:][e])-gamma<0).astype(np.int8)
        assert np.max(np.abs(yaudit.mean(axis=1)-published.t_audit.to_numpy()))<1e-12
        refs = []
        for j in range(6):
            seed = 2026111700+10000*bank+j
            path = work/f"CALREF{j}.txt"
            if path.exists(): booster = lgb.Booster(model_file=str(path))
            else:
                rng = np.random.default_rng(seed)
                weights = np.bincount(rng.integers(len(bt), size=len(bt)), minlength=len(bt))
                params = {"objective":"binary", "learning_rate":.05, "num_leaves":15,
                    "max_depth":5, "min_data_in_leaf":40, "lambda_l2":1., "num_threads":1,
                    "verbosity":-1, "deterministic":True, "force_col_wise":True,
                    "seed":seed, "feature_fraction":1., "bagging_fraction":1., "bagging_freq":0}
                booster = lgb.train(params, lgb.Dataset(train_x, label=labels, weight=weights,
                    categorical_feature=list(old.DIABETES_CATEGORICAL)), num_boost_round=100)
                booster.save_model(str(path)); fits += 1
            refs.append((old.safe_logit(booster.predict(cal_x, num_threads=1))-gamma)[c])
        tcal = (np.column_stack(refs)<0).mean(axis=1)
        ic, ie = zi[:ncal][c], zi[ncal:][e]
        sigma2 = float(np.mean(np.var(ic,axis=1,ddof=1)))
        transform, delta, eta = old._p3_proxy_fit(cal,bt,prep,z0[:ncal],p0[:ncal],zi[:ncal],sigma2)
        all_x = pd.concat([cal,evaluation],ignore_index=True)
        px = old.transform_sparse_proxy(all_x,prep,transform,z0,p0)
        proxy_u = (z0+delta.predict(px))/np.sqrt(np.clip(np.exp(eta.predict(px)),.1*sigma2,10*sigma2))
        features, raw = role_scores(z0[:ncal][c],z0[ncal:][e],ic,ie,sigma2,
                                   surrogate=(proxy_u[:ncal][c],proxy_u[ncal:][e]))
        features["shrunk_mean_only_calibrated"]=(.5*z0[:ncal][c]+.5*ic.mean(axis=1),
                                                .5*z0[ncal:][e]+.5*ie.mean(axis=1))
        result, maps, scores = evaluate(bank,"bootstrap_independent_CALREF",features,raw,tcal,yaudit)
        # Monotonic recalibration preserves the teacher's ranking and risk curve.
        raw_metric = next(r for r in result if r["method"]=="qMV_raw_probability")
        calibrated_metric = next(r for r in result if r["method"]=="qMV_calibrated")
        assert abs(raw_metric["R_070"]-calibrated_metric["R_070"])<1e-12
        assert abs(raw_metric["AUC_vs_AUDIT_flip"]-calibrated_metric["AUC_vs_AUDIT_flip"])<1e-12
        rows.extend(result); write_metrics(folder/"per_bank_metrics.csv",rows)
        dump(work/"calibrators.json",maps)
        np.savez_compressed(work/"scores.npz",tcal=tcal,audit=yaudit,**scores)
        summaries.append({"bank":bank,"CAL_alerts":int(c.sum()),"EVAL_alerts":int(e.sum()),
                          "calibration_reference_models":6,"sigma2":sigma2})
        print(f"P3 bank {bank}: completed independent-reference calibration",flush=True)
    report={"status":"P3_FAIR_CALIBRATION_COMPLETED","new_reference_model_fits_this_run":fits,
        "new_Ridge_fits":6,"new_calibrator_fits":21,"banks":summaries,
        "calibration_targets":"six new CALREF models; AUDIT not used for any fitted target",
        "feature_teacher":"original six INT models", "evaluation":"original eleven AUDIT; existing public EVAL reused, exploratory",
        "wall_seconds":time.monotonic()-started,"cpu_seconds":time.process_time()-cpu}
    dump(folder/"report.json",report)
    print(json.dumps(report),flush=True)


def conditional_if(xa, ya, xb, yb, beta, x, alpha, bootstrap=False):
    pa, pb = expit(xa@beta), expit(xb@beta)
    ua, ub = (pa-ya)[:,None]*xa, (pb-yb)[:,None]*xb
    penalty = np.ones(len(beta)); penalty[0]=0
    h = xa.T@((pa*(1-pa))[:,None]*xa)+np.diag(penalty)
    inv = np.linalg.inv(h)
    ca, cb = ua-ua.mean(axis=0), ub-ub.mean(axis=0)
    n, nb = len(xa),len(xb)
    if bootstrap:
        shift = np.zeros(len(beta)); j=ca.T@ca
    else:
        k = int(round(alpha*n)); l=n-k
        shift = -inv@((alpha-1)*ua.sum(axis=0)+(l/nb)*ub.sum(axis=0))
        # Exact covariance of sums under independent SRS without replacement.
        j = k*(1-k/n)*(ca.T@ca)/(n-1)+l*(1-l/nb)*(cb.T@cb)/(nb-1)
    m = x@beta+x@shift
    v = np.einsum("ij,jk,ik->i",x@inv,j,x@inv,optimize=True)
    assert np.isfinite(v).all() and v.min()>=-1e-12
    return m/np.sqrt(np.maximum(v,1e-12)),v


def anchor_if(xa, ya, xb, yb, beta, x, alpha, bootstrap=False):
    """Linearize the random operator at its expected-objective minimizer."""
    n,nb=len(xa),len(xb)
    pa,pb=expit(xa@beta),expit(xb@beta)
    ua,ub=(pa-ya)[:,None]*xa,(pb-yb)[:,None]*xb
    ca,cb=ua-ua.mean(axis=0),ub-ub.mean(axis=0)
    penalty=np.ones(len(beta));penalty[0]=0
    if bootstrap:
        h=xa.T@((pa*(1-pa))[:,None]*xa)+np.diag(penalty)
        j=ca.T@ca
    else:
        k=int(round(alpha*n));l=n-k
        h=alpha*(xa.T@((pa*(1-pa))[:,None]*xa))+(l/nb)*(xb.T@((pb*(1-pb))[:,None]*xb))+np.diag(penalty)
        j=k*(1-k/n)*(ca.T@ca)/(n-1)+l*(1-l/nb)*(cb.T@cb)/(nb-1)
    inv=np.linalg.inv(h)
    variance=np.einsum("ij,jk,ik->i",x@inv,j,x@inv,optimize=True)
    assert np.isfinite(variance).all() and variance.min()>=-1e-12
    return (x@beta)/np.sqrt(np.maximum(variance,1e-12)),variance


def run_p2():
    folder=OUT/"p2_variable_reference_sets";folder.mkdir(parents=True,exist_ok=True)
    prior=old.read_json(folder/"report.json") if (folder/"report.json").exists() else {}
    reference_fits_before=prior.get("reference_model_fits_total",prior.get("new_Logistic_fits_this_run",0))
    map_fits_before=prior.get("calibrator_fits_total",prior.get("new_calibrator_fits",0))
    started=time.monotonic();cpu=time.process_time();fits=0;anchor_fits=0;map_fits=0;rows=[];validity=[]
    _,_,parts=old._adult_groups_for_p2();pools=old._p2_bank_pools(parts["train"])
    occupied={g["group_id"] for a,b in pools for g in a+b}
    reserve=sorted((g for g in parts["train"] if g["group_id"] not in occupied),key=lambda g:g["group_id"])
    assert len(reserve)>=6000
    for bank,(a,b_old) in enumerate(pools):
        work=folder/f"bank{bank}";work.mkdir(exist_ok=True)
        b=b_old+reserve[bank*2000:(bank+1)*2000]
        design=old.read_json(old.RUN_ROOT/f"adult_p2/bank-{bank}/pool_manifest/fit_design.json")
        fit=old._adult_lr_transform_fit(a,numeric_feature_indices=design["numeric_feature_indices"])
        xa,xb=old._adult_lr_features(a,fit),old._adult_lr_features(b,fit)
        ya,yb=old._group_soft_labels(a),old._group_soft_labels(b)
        xc,xe=old._adult_lr_features(parts["calibration"],fit),old._adult_lr_features(parts["evaluation"],fit)
        xall=np.vstack([xc,xe]);ncal=len(xc)
        beta0=np.asarray(old.read_json(old._p2_fit_path(bank,"shared","D0")/"model.json")["beta"])
        z0=xall@beta0;c,e=z0[:ncal]>=0,z0[ncal:]>=0
        with np.load(old.RUN_ROOT/f"adult_p2/bank-{bank}/logit_prediction_cache/arrays.npz",allow_pickle=False) as prior:
            prior_z0=prior["z_shared_D0"]
        cache_difference=float(np.max(np.abs(z0-prior_z0)))
        print(f"P2 bank {bank}: corrected feature-order D0 cache max difference={cache_difference:.6g}",flush=True)
        dump(work/"pools.json",{"A":[g["group_id"] for g in a],"expanded_B":[g["group_id"] for g in b],
             "operation_change":"A and D0 retained; B expanded from 2000 to 4000; per-reference SRS draws vary",
             "legacy_numeric_feature_indices":design["numeric_feature_indices"],
             "old_D0_logit_cache_max_difference_after_order_restore":cache_difference,
             "target":"conditional retraining from fixed empirical pools, not independent superpopulation"})
        for op,alpha in [("alpha_0",0.),("alpha_0_5",.5),("bootstrap",None)]:
            path=work/f"{op}.npz"
            if path.exists():
                with np.load(path,allow_pickle=False) as saved:
                    betas=saved["betas"].copy();z=saved["logits"].copy()
            else:
                bs=[]
                for role,number in [("INT",6),("CALREF",6),("AUDIT",11)]:
                    for j in range(number):
                        if op=="bootstrap" and role!="CALREF":
                            beta=np.asarray(old.read_json(old._p2_fit_path(bank,op,f"{role}-{j+1:02d}")/"model.json")["beta"])
                        else:
                            seed=2026111800+bank*10000+{"alpha_0":0,"alpha_0_5":1000,"bootstrap":2000}[op]+{"INT":0,"CALREF":100,"AUDIT":200}[role]+j
                            rng=np.random.default_rng(seed)
                            if op=="bootstrap":
                                weights=np.bincount(rng.integers(len(a),size=len(a)),minlength=len(a))
                                beta,_=old._fit_regularized_logistic(xa,ya,weights)
                            else:
                                k=int(alpha*len(a));l=len(a)-k
                                ia=rng.choice(len(a),k,replace=False);ib=rng.choice(len(b),l,replace=False)
                                assert len(set(ia))==k and len(set(ib))==l
                                beta,_=old._fit_regularized_logistic(np.vstack([xa[ia],xb[ib]]),np.r_[ya[ia],yb[ib]])
                            fits+=1
                        bs.append(beta)
                betas=np.row_stack(bs);z=xall@betas.T
                np.savez_compressed(path,betas=betas,logits=z)
            unique=len(np.unique(np.round(betas,decimals=12),axis=0))
            assert unique==23,(bank,op,unique)
            ic,ie=z[:ncal,:6][c],z[ncal:,:6][e]
            tc=(z[:ncal,6:12][c]<0).mean(axis=1)
            audit=(z[ncal:,12:][e]<0).astype(np.int8)
            sigma2=float(np.var(ic,axis=1,ddof=1).mean());assert sigma2>1e-12
            uif,vif=conditional_if(xa,ya,xb,yb,beta0,xall,alpha or 0.,op=="bootstrap")
            features,raw=role_scores(z0[:ncal][c],z0[ncal:][e],ic,ie,sigma2,
                                    if_pair=(uif[:ncal][c],uif[ncal:][e]))
            if op=="bootstrap":
                beta_anchor=beta0
            else:
                anchor_path=work/f"{op}_anchor.json"
                if anchor_path.exists():beta_anchor=np.asarray(old.read_json(anchor_path)["beta"])
                else:
                    weights=np.r_[np.full(len(a),alpha),np.full(len(b),(1-alpha)*len(a)/len(b))]
                    beta_anchor,fit_info=old._fit_regularized_logistic(np.vstack([xa,xb]),np.r_[ya,yb],weights)
                    dump(anchor_path,{"beta":beta_anchor,"fit_info":fit_info,
                        "weights_A":alpha,"weights_B":(1-alpha)*len(a)/len(b),
                        "role":"expected random-training objective center; not an AUDIT fit"})
                    anchor_fits+=1
            uanchor,vanchor=anchor_if(xa,ya,xb,yb,beta_anchor,xall,alpha or 0.,op=="bootstrap")
            anchor_mean=xall@beta_anchor
            features["anchor_mean_calibrated"]=(anchor_mean[:ncal][c],anchor_mean[ncal:][e])
            features["IF_anchor_calibrated"]=(uanchor[:ncal][c],uanchor[ncal:][e])
            raw["IF_anchor_raw_probability"]=ndtr(-uanchor[ncal:][e])
            result,maps,scores=evaluate(bank,op,features,raw,tc,audit)
            map_fits+=len(maps)
            rows.extend(result);write_metrics(folder/"per_bank_metrics.csv",rows)
            validity.append({"bank":bank,"operation":op,"unique_coefficient_vectors":unique,
                "INT_unique":len(np.unique(np.round(betas[:6],12),axis=0)),
                "CALREF_unique":len(np.unique(np.round(betas[6:12],12),axis=0)),
                "AUDIT_unique":len(np.unique(np.round(betas[12:],12),axis=0)),
                "CAL_INT_variance_mean":sigma2,"IF_variance_EVAL_mean":float(vif[ncal:][e].mean()),
                "AUDIT_variance_EVAL_mean":float(np.var(z[ncal:,12:][e],axis=1,ddof=1).mean()),
                "anchor_IF_variance_EVAL_mean":float(vanchor[ncal:][e].mean()),
                "IF_model":"conditional finite-pool mean shift and sampling-without-replacement FPC; bootstrap uses conditional multinomial covariance"})
            dump(work/f"{op}_calibrators.json",maps)
            np.savez_compressed(work/f"{op}_scores.npz",audit=audit,**scores)
            print(f"P2 bank {bank} {op}: INT/CALREF/AUDIT unique 6/6/11",flush=True)
    report={"status":"P2_VARIABLE_FINITE_POOL_REFERENCE_REPAIR_COMPLETED","new_Logistic_fits_this_run":fits,
        "reference_model_fits_total":reference_fits_before+fits,
        "new_expected_objective_anchor_fits":anchor_fits,"anchor_model_fits_total":6,
        "D0_models_reused":3,"bootstrap_INT_AUDIT_models_reused":51,"new_calibrator_fits":map_fits,
        "calibrator_fits_total":map_fits_before+map_fits,
        "validity":validity,"wall_seconds":time.monotonic()-started,"cpu_seconds":time.process_time()-cpu,
        "limits":["changed sampling design reported as a new experiment; old degenerate results retained",
                  "fixed finite training pools and already-used public EVAL; not superpopulation or clinical confirmation",
                  "IF is a local linear approximation, not a guarantee or empirical verification of C/V0=1, VR/V0=2"]}
    dump(folder/"report.json",report);print(json.dumps(report),flush=True)


def selfcheck():
    population=np.array([[0.,1.],[1.,-1.],[2.,2.],[4.,0.]])
    sums=np.array([population[list(idx)].sum(axis=0) for idx in itertools.combinations(range(4),2)])
    exact=np.cov(sums,rowvar=False,ddof=0)
    formula=2*(1-2/4)*np.cov(population,rowvar=False,ddof=1)
    assert np.max(np.abs(exact-formula))<1e-12
    x=np.array([-1.,.3,1.2]);t=np.array([.8,.4,.1]);theta=np.array([.2,-.8])
    _,grad=loss(theta,x,t)
    numeric=np.array([(loss(theta+1e-6*np.eye(2)[j],x,t)[0]-loss(theta-1e-6*np.eye(2)[j],x,t)[0])/2e-6 for j in range(2)])
    assert np.max(np.abs(grad-numeric))<1e-8
    print("Finite-population covariance enumeration and calibration gradient checks passed.")


def run_utility():
    """Describe true public readmission outcomes at the same alert budget."""
    patients=pd.read_csv(old.OUT_ROOT/"diabetes_data_audit/selected_patient_split_manifest.csv")
    outcomes=patients[patients.split=="EVAL"].set_index("patient_key_sha256").y_readmit30.to_dict()
    total=sum(outcomes.values());rows=[]
    for bank in range(3):
        published=pd.read_csv(old.OUT_ROOT/f"p3_three_bank_results/bank{bank}_eval_alert_scores.csv")
        y=np.array([outcomes[key] for key in published.patient_key_sha256],float)
        with np.load(OUT/f"p3_independent_calibration/bank{bank}/scores.npz",allow_pickle=False) as data:
            for method in ["margin_calibrated","int_mean_calibrated","qMV_calibrated","surrogate_calibrated"]:
                q=data[method]
                for coverage in [.7,.9]:
                    values,counts=np.unique(q,return_counts=True);mass=coverage*len(q)
                    boundary=values[np.searchsorted(np.cumsum(counts),mass,side="left")]
                    weights=(q<boundary).astype(float);ties=q==boundary
                    weights[ties]=(mass-weights.sum())/ties.sum()
                    assert abs(weights.sum()-mass)<1e-8
                    tp=float(weights@y)
                    rows.append({"bank":bank,"method":method,"coverage":coverage,
                        "retained_alert_weight":mass,"observed_readmission_precision":tp/mass,
                        "sensitivity_among_all_EVAL_readmissions":tp/total,
                        "D0_true_positive_retention":tp/y.sum(),"evaluation_true_readmission_count":total,
                        "interpretation":"descriptive public discharge benchmark; fractional ties; no clinical claim"})
    write_metrics(OUT/"p3_observed_outcome_utility.csv",rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["check","p3","p2","utility"])
    args=parser.parse_args()
    if hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    {"check":selfcheck,"p3":run_p3,"p2":run_p2,"utility":run_utility}[args.stage]()
