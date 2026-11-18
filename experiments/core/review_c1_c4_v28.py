#!/usr/bin/env python3
"""Numerical C1-C4 review. No model or CAL fit; historical package retained."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import expit, ndtr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core import run_c1_c4_v28 as work
from experiments.core.review_r1_r4 import metrics as independent_metrics

OUT = work.OUT / "review"
KEYS = ["dataset", "operation", "pair", "metric"]


def save(name, records):
    pd.DataFrame(records).to_csv(OUT / name, index=False)


def weighted_metrics(q, target, weights):
    """Independent grouped fractional-tie risk over a weighted empirical cohort."""
    q = np.asarray(q); t = np.asarray(target); w = np.asarray(weights)
    levels, inverse = np.unique(q, return_inverse=True)
    mass = np.bincount(inverse, weights=w, minlength=len(levels))
    flips = np.bincount(inverse, weights=w*t, minlength=len(levels))
    keep = mass > 0
    mass, flips = mass[keep], flips[keep]
    cum = np.r_[0., np.cumsum(mass)]; ycum = np.r_[0., np.cumsum(flips)]
    cuts = np.arange(1,101)/100 * w.sum()
    index = np.searchsorted(cum[1:], cuts, side="left")
    fraction = (cuts-cum[index])/mass[index]
    risks = (ycum[index]+fraction*flips[index])/cuts
    bins = np.minimum(np.floor(q*10).astype(int),9)
    ece = sum(abs(np.sum(w[bins==b]*(q[bins==b]-t[bins==b]))) for b in range(10))/w.sum()
    return {"R_070": risks[69], "AURC100": risks.mean(), "ECE10": ece}


def paired_from_target(case, target, weights):
    result = {}
    qs = case["q"]
    met = {name: weighted_metrics(qs[name],target,weights)
           for name in {item for pair in work.PAIR_SPECS.values() for item in pair}}
    for pair,(left,right) in work.PAIR_SPECS.items():
        result[pair] = {name: met[left][name]-met[right][name] for name in met[left]}
        result[pair]["Brier"] = np.sum(weights*(qs[left]**2-qs[right]**2-2*(qs[left]-qs[right])*target))/weights.sum()
    return result


def review_c1():
    frame = pd.read_csv(work.OUT / "C1_score_matched_metrics.csv")
    qdiff = mdiff = std_diff = gradient_max = loss_diff = 0.
    for bank in range(6):
        p = work._read_npz(work.U1 / f"bank{bank}/prospective/prospective_scores.npz")
        for op in ("Op_Half_Replace", "Op_Full_Replace"):
            a = work._read_npz(work.U1 / f"bank{bank}/{op}/prediction_arrays.npz")
            saved = work._read_npz(work.OUT / "c1_recalibrated" / f"bank{bank}_{op}.npz")
            for _,row in frame[(frame.bank==bank)&(frame.operation==op)].iterrows():
                numerator, divisor = row.score.split("_div_")
                old = row.calibration_target == "Old_Target_Calibrated"
                cal = a[f"{numerator}_cal_alert"].copy(); eva = a[f"{numerator}_eval_alert"].copy()
                if divisor != "1":
                    vcal = p["v0_cal"] if divisor == "sqrt_v0" else a["vR_cal_alert"]
                    veval = p["v0_eval_alert"] if divisor == "sqrt_v0" else a["vR_eval_alert"]
                    cal /= np.sqrt(np.maximum(vcal,1e-300)); eva /= np.sqrt(np.maximum(veval,1e-300))
                std_diff = max(std_diff, abs(cal.mean()-row.score_CAL_mean), abs(cal.std()-row.score_CAL_sd))
                target = p["tcal_poolA"] if old else a["tcal_update"]
                standardized = (cal-row.score_CAL_mean)/row.score_CAL_sd
                pcal = expit(np.clip(row.map_intercept+row.map_slope*standardized,-40,40))
                residual = pcal-target
                gradient = 2*residual*pcal*(1-pcal)
                gradient_max = max(gradient_max,abs(gradient.mean()),abs(np.mean(gradient*standardized)))
                loss_diff = max(loss_diff,abs(np.mean(residual**2)-row.map_brier_loss_CAL))
                q = expit(np.clip(row.map_intercept+row.map_slope*(eva-row.score_CAL_mean)/row.score_CAL_sd,-40,40))
                qdiff = max(qdiff,float(np.max(np.abs(q-saved[f"{row.score}__{row.calibration_target}"]))))
                m = independent_metrics(q,a["audit_eval_alert"])
                for key,column in (("R_070","R_070"),("AURC100","AURC100"),("Brier","Brier_finite_AUDIT"),("ECE10","ECE10"),("AUC_flip","AUC")):
                    mdiff = max(mdiff,abs(m[key]-float(row[column])))
    assert qdiff < 1e-12 and mdiff < 1e-12 and std_diff < 1e-12 and loss_diff < 1e-12 and gradient_max < 1e-5
    return {"maps_replayed":len(frame),"max_probability_difference":qdiff,
            "max_metric_difference":mdiff,"max_CAL_standardization_difference":std_diff,
            "max_CAL_gradient_inf_recomputed":float(gradient_max),"max_CAL_loss_difference":float(loss_diff),
            "slope_min":float(frame.map_slope.min()),"slope_max":float(frame.map_slope.max())}


def review_bootstrap(datasets):
    reps = pd.read_csv(work.OUT/"C2_row_bootstrap_replicates.csv")
    summaries = pd.read_csv(work.OUT/"C2_row_bootstrap_summary.csv")
    digests = pd.read_csv(work.OUT/"C2_patient_weight_digests.csv")
    summary_diff = sample_diff = 0.
    samples_checked = 0; weight_checks = 0
    for key, group in reps.groupby(KEYS,sort=False):
        assert len(group)==1000 and set(group.replicate)==set(range(1,1001))
        row = summaries.set_index(KEYS).loc[key]
        values = group.mean_delta_across_6_banks.to_numpy()
        for col,value in (("bootstrap_mean",values.mean()),("row_bootstrap_SE",values.std(ddof=1)),
                          ("percentile_2_5",np.quantile(values,.025)),("percentile_97_5",np.quantile(values,.975))):
            summary_diff = max(summary_diff,abs(value-row[col]))
    for dataset_index,(dataset,data) in enumerate(datasets.items()):
        rng = np.random.default_rng(np.random.SeedSequence([work.BOOTSTRAP_SEED,dataset_index]))
        n = len(data["ids"])
        for rep in range(1,1001):
            counts = rng.multinomial(n,np.full(n,1/n)).astype(np.uint32)
            digest = hashlib.sha256(counts.astype("<u4").tobytes()).hexdigest()
            assert digest == digests[(digests.dataset==dataset)&(digests.replicate==rep)].iloc[0].patient_weight_sha256
            weight_checks += 1
            if rep not in (1,2,17,97,1000): continue
            samples_checked += 1
            for op,cases in data["operations"].items():
                values = {pair:{metric:[] for metric in work.METRICS} for pair in work.PAIR_SPECS}
                for case in cases.values():
                    result = paired_from_target(case,case["audit"].mean(axis=1),counts[case["alert_idx"]])
                    for pair in values:
                        for metric in work.METRICS: values[pair][metric].append(result[pair][metric])
                selected = reps[(reps.dataset==dataset)&(reps.operation==op)&(reps.replicate==rep)]
                for _,row in selected.iterrows():
                    value = np.mean(values[row.pair][row.metric])
                    sample_diff = max(sample_diff,abs(value-row.mean_delta_across_6_banks))
    assert summary_diff < 1e-12 and sample_diff < 1e-12
    return {"replicate_rows_summarized":len(reps),"summary_rows":len(summaries),
            "weight_vectors_reproduced":weight_checks,"full_weight_replicates_independently_checked":samples_checked,
            "max_summary_difference":summary_diff,"max_checked_replicate_difference":sample_diff,
            "scope":"all summary quantiles and all seed weights; numerical recutting checked on 5 replicates per dataset, not every replicate"}


def review_jackknife(datasets):
    summaries = []; deletions = []; comparisons = []
    old = pd.read_csv(work.OUT/"C2_audit_loao_summary.csv").set_index(KEYS)
    all_linear_diff = 0.
    for dataset,data in datasets.items():
        for op,cases in data["operations"].items():
            rows, drops = work._stratified_column_jackknife(cases,dataset,op)
            summaries.extend(rows); deletions.extend(drops)
            # Independent check for linear-in-reference targets: the variance
            # of a mean of B independent columns in each independent bank.
            bank_column = {pair:{metric:[] for metric in work.METRICS} for pair in work.PAIR_SPECS}
            for case in cases.values():
                results = [paired_from_target(case,case["audit"][:,j],np.ones(len(case["audit"]))) for j in range(11)]
                for pair in bank_column:
                    for metric in work.METRICS:
                        bank_column[pair][metric].append([r[pair][metric] for r in results])
            for row in rows:
                pair, metric = row["pair"],row["metric"]
                deleted = np.array([[r["mean_delta_across_6_banks"] for r in drops if r["pair"]==pair and r["metric"]==metric and r["omitted_bank"]==bank]
                                    for bank in sorted(cases)])
                permuted = np.vstack([np.roll(deleted[i],i+1) for i in range(6)])
                assert abs(work._stratified_jackknife_se(deleted)-work._stratified_jackknife_se(permuted))<1e-12
                previous = old.loc[(dataset,op,pair,metric)]
                # Old simultaneous deletions can be recovered from individual
                # grand means: theta + sum_k(theta_{k,-j}-theta).
                point = float(pd.read_csv(work.OUT/"C2_point_paired_summary.csv").set_index(KEYS).loc[(dataset,op,pair,metric)].paired_mean)
                old_sync = point+np.sum(deleted-point,axis=0)
                old_perm = point+np.sum(permuted-point,axis=0)
                old_se = np.sqrt(10/11*np.sum((old_sync-old_sync.mean())**2))
                perm_se = np.sqrt(10/11*np.sum((old_perm-old_perm.mean())**2))
                assert abs(old_se-previous.jackknife_SE)<1e-12
                if metric != "ECE10":
                    columns = np.asarray(bank_column[pair][metric])
                    independent_se = float(np.sqrt(np.sum(columns.var(axis=1,ddof=1))/11)/6)
                    all_linear_diff=max(all_linear_diff,abs(independent_se-row["jackknife_SE"]))
                comparisons.append({**{key:row[key] for key in KEYS},"old_synchronized_SE":old_se,
                                    "old_SE_after_bankwise_column_relabeling":perm_se,
                                    "correct_stratified_SE":row["jackknife_SE"]})
    assert all_linear_diff < 1e-12
    save("C2_stratified_audit_summary.csv",summaries); save("C2_stratified_audit_deletions.csv",deletions)
    save("C2_SE_permutation_check.csv",comparisons)
    paired = pd.read_csv(work.OUT/"Table1_paired_effects.csv")
    new = pd.DataFrame(summaries).set_index(KEYS)
    for index,row in paired.iterrows():
        corrected = new.loc[tuple(row[key] for key in KEYS)]
        for column,source in (("audit_LOAO_min","LOAO_min"),("audit_LOAO_max","LOAO_max"),("audit_LOAO_SE","jackknife_SE")):
            paired.loc[index,column] = corrected[source]
    paired["audit_scope"]="one column in one bank deleted; independent bank strata; conditional MC diagnostic, ECE SE approximate"
    paired.to_csv(OUT/"Table1_paired_effects_reviewed.csv",index=False)
    return {"summary_rows":len(summaries),"stratified_deletions":len(deletions),
            "max_linear_endpoint_SE_vs_direct_column_variance":all_linear_diff,
            "max_legacy_SE_change_under_column_relabeling":float(max(abs(r["old_synchronized_SE"]-r["old_SE_after_bankwise_column_relabeling"]) for r in comparisons))}


def posterior_counterexample():
    # Explicit two-point prior; conditional moments do not determine its CDF.
    z,a=1.,2.
    posterior_plus=float(expit(2*a*z))
    mean=a*(2*posterior_plus-1)
    variance=1+4*a*a*posterior_plus*(1-posterior_plus)
    exact=posterior_plus*ndtr(-a)+(1-posterior_plus)*ndtr(a)
    approx=ndtr(-mean/np.sqrt(variance))
    return {"prior":"equal mass at mu=-2,+2; independent N(0,1) observation/retrain errors",
            "z":z,"predictive_mean":mean,"predictive_variance":variance,
            "exact_flip_probability":float(exact),"Gaussian_moment_approximation":float(approx)}


def render_reviewed_manuscript():
    text=(work.OUT/"MANUSCRIPT_CORE_DRAFT.md").read_text()
    intro=text.index("## 1. Introduction and scope")
    text=("# Manuscript core draft: scientifically reviewed C1–C4\n\n"
          "> Working research draft; the historical directory name does not authorize final table lock or submission. "
          "Unchanged source tables remain in the [original package](../README.md); scientific corrections are in [REVIEW.md](REVIEW.md). "
          "No predictive models or CAL maps were fitted in this review.\n\n"+text[intro:])
    text=text.replace("the EVAL outcome labels were not opened in this task.",
                      "this reanalysis did not use EVAL outcome values for model fitting, calibration or selection. Historical outcome utility is carried forward from derived summaries.")
    text=text.replace("Fixed latent mean and independent noise; the classical `sqrt(2) sigma` construction does not encode a concrete pool update",
                      "Gaussian observation identities require an explicit noise/latent model; the sqrt(2) plug-in used in our early experiments is not mandated by Tweedie's identity")
    text=text.replace("We compare matched-pool bootstrap and pool replacement and use the three-covariance conditional moments; the results bound when independent-noise shrinkage transports poorly",
                      "We examine specified bootstrap and pool-update designs; scalar Gaussian moments are model-dependent context, not an exact theorem for all finite-pool retraining risks")
    start=text.index("Under a conditional Gaussian approximation")
    end=text.index("## 4. Main Empirical Results",start)
    methods='''The following scalar observation model is illustrative, not an identity for conditioning on an entire fitted D0 and all finite-pool information. Let `Z0=mu+eps0` and `ZR=mu+epsR`, where `(eps0,epsR)` is zero-mean jointly Gaussian, independent of `mu`, with **noise** variances `V0>0, VR>=0`, covariance `C`, and `V0*VR>=C^2`. These are not generally the marginal variances of Z0 and ZR after mixing over mu. For `l(z)=log m(z)`, the Gaussian observation identities give exact scalar conditional moments under this model:

`MR(z)=z+(V0-C)l'(z)` and `WR(z)=V0+VR-2C+(V0-C)^2 l''(z)`.

The model assumes a common latent center. It does not automatically cover population drift between training pools. If noise variances vary across instances, the density and derivatives must condition on the corresponding variance/noise stratum. Moment knowledge alone does not determine a crossing probability: `Phi(-s0*MR/sqrt(WR))` is a **moment-matched Gaussian approximation**, exact only when the conditional predictive law is Gaussian. At zero predictive variance the deterministic threshold decision applies, including the declared equality-at-zero rule. A two-point-prior counterexample is given in the review. None of the empirical IF comparisons below requires this scalar predictive law to be exact.

### 3.2 Actual retraining experiments and information available

All primary comparisons are on the same D0-positive queue within a bank/operation. Scores use `z(x)=logit(p(x))-logit(tau0)`, where tau0 is D0's CAL 80th probability percentile and is held fixed for reference models. The binary outcome defines supervised TRAIN labels; CALREF and AUDIT targets here are retraining decision flips, not clinical/credit outcomes. Each role uses 6 INT, 6 CALREF and 11 AUDIT models. Role models may share training rows according to the declared sampling law; TRAIN/CAL/EVAL roles remain separate.

| Dataset/design | Available training pool and D0 | Reference-model sampling |
|---|---|---|
| Diabetes Scheme D bootstrap | TRAIN 42,058; each bank selects 33,646 distinct TRAIN patients and fits D0 on them | Multinomial bootstrap of that bank's D0 training sample, 33,646 draws |
| Diabetes Scheme D updates | TRAIN split into A/B of 21,029 each; D0 uses 14,000 SRSWOR A patients | Half: 7,000 A + 7,000 B SRSWOR; full: 14,000 B SRSWOR |
| Credit bootstrap | TRAIN 18,000; A/B of 9,000 each; D0 is a 12,000-draw bootstrap of A (draw count is not unique-client count) | Bootstrap of the saved D0 multiset, 12,000 draws |
| Credit updates | Same Credit D0 and A/B pools as above | Half: 3,000 A + 3,000 B SRSWOR; full: 6,000 B SRSWOR |

Diabetes CAL/EVAL have 13,983/13,949 distinct selected patients; Credit CAL/EVAL have 6,000/6,000 records. The Diabetes bootstrap and update panels use different D0 training samples, preprocessors and alert queues. Credit bootstrap and updates have different reference sample sizes. Therefore cross-operation risk differences are descriptive design differences, not isolated causal effects of overlap fraction. The name half replacement denotes source-pool proportions, not exactly one-half retention of D0 training records.

The main predictive family is Logistic regression with objective `sum_i w_i logloss_i(beta) + (1/2)||beta_slopes||^2`; the intercept is unpenalized. Scheme D merges categorical levels with training support below 50 and leaves one-hot values unscaled; numerical preprocessing is fit only on the relevant training subset (bootstrap) or Pool A (updates). The operator-center baseline betaR minimizes the expected weighted training objective using available labeled A/B pools. It is not generally the exact expectation of fully refitted beta. This additional labeled-pool access and center fitting are part of the method, not free information.

For SRSWOR updates, IF uses `SigmaR=H_R^{-1} J_R H_R^{-1}`, with expected-objective Hessian H_R including ridge and `J_R=k(1-k/N_A)S_A+l(1-l/N_B)S_B`; S_A/S_B are sample score covariances with denominator N-1 at betaR. Conditional bootstrap uses its multinomial score covariance without an extra factor of two. The local projection is `vR(x)=phi(x)^T SigmaR phi(x)`. Margin, center-only and IF scores are z0, zR and zR/sqrt(vR), mapped to flip probabilities on CAL. These are local approximations assessed against full refits, not universal Gaussian or clinical guarantees.

### 3.3 Matched calibration, comparators, and uncertainty

C1 standardizes each score by its own CAL mean and population SD and fits the same nonincreasing two-parameter logistic-Brier map separately to old Pool-A bootstrap or new operation CALREF6 targets. Identical standardization is shared between the two targets for a score. All 144 saved slopes are strictly negative and the observed target-source changes leave ranking metrics unchanged. qMV and the CAL-tuned INT6 three-dimensional mean/variance comparator are retained, including failures. Learned comparators select regularization by five-fold CAL CV over the recorded extended grid; no EVAL flip target is used for selection in this analysis. Prior repeated Diabetes EVAL use remains part of its development history.

C2 uses 1,000 shared patient-frequency weight vectors per dataset across all six banks and operations, re-cutting weighted coverage with fractional exact ties each time. Percentile intervals are empirical row-resampling sensitivities conditional on the fitted models, calibrators and saved AUDIT library; they do not include model fitting, CAL selection or arbitrary future-population uncertainty. They resample the empirical EVAL distribution, rather than literally fixing its row multiplicities. For AUDIT Monte Carlo sensitivity, delete one whole model column **within one bank at a time**, leaving all other banks intact. If G[k,j] is the six-bank statistic after that deletion, the independent-bank estimate is `SE^2=sum_k ((B-1)/B) sum_j (G[k,j]-mean_j G[k,j])^2`. Independently permuting AUDIT columns within banks does not change this estimate. For R70, AURC and paired Brier this matches direct column-mean variance; the non-smooth ECE jackknife is diagnostic only. Patient sensitivity, finite-library MC sensitivity and bank training variation are reported separately, not combined into an omnibus confidence interval. The many exploratory intervals are not simultaneous coverage or confirmatory discovery tests.

'''
    text=text[:start]+methods+text[end:]
    paired=pd.read_csv(OUT/"Table1_paired_effects_reviewed.csv")
    table=["**Table 1B. Increment beyond strong comparators.** Differences are IF minus comparator; brackets are conditional empirical row-bootstrap 2.5/97.5 percentiles. All three pairs, including Margin, remain in the reviewed CSV.","",
           "| Dataset | Operation | IF−operator center ΔR70 [row sensitivity interval] | IF−tuned INT6 ΔR70 [row sensitivity interval] |",
           "|---|---|---:|---:|"]
    for dataset in ("Diabetes_Scheme_D","UCI_Credit"):
        for op in work.OPS:
            cells=[]
            for pair in ("Pair_2_IF_minus_OpCenter_Mean","Pair_3_IF_minus_Tuned_INT6_3D"):
                row=paired[(paired.dataset==dataset)&(paired.operation==op)&(paired.pair==pair)&(paired.metric=="R_070")].iloc[0]
                cells.append(f"{row.paired_mean:.6f} [{row.row_bootstrap_CI_2_5:.6f}, {row.row_bootstrap_CI_97_5:.6f}]")
            table.append("| "+" | ".join([dataset,op,*cells])+" |")
    table.append("\nDiabetes half replacement gives little incremental R70 reduction over the operator center and its row interval spans zero. Bootstrap center-only equals Margin, not IF. Credit supports incremental IF value on these tested designs; it does not imply universal ECE or clinical superiority. Corrected stratified AUDIT summaries are in `C2_stratified_audit_summary.csv`.\n\n")
    start=text.index("**Table 1B.");end=text.index("### 4.2",start)
    text=text[:start]+"\n".join(table)+text[end:]
    text=text.replace("### 4.4 Clinical outcome trade-off and partial scoring-path cost — Table 4",
                      "### 4.4 Historical Scheme A outcome trade-off and Scheme D partial cost — Table 4")
    marker="The clinical table keeps both denominators:"
    text=text.replace(marker,"**The utility panel is historical Scheme A scaled-OHE bootstrap evidence from S4, not Scheme D and not a pool-update experiment. Its fitted models and D0 alert queues differ from Table 1. No conclusion about the final Scheme D model's outcome utility follows from this panel.** The labeled CSV is `Table4_clinical_utility_labeled.csv`.\n\n"+marker)
    text=text.replace("**Table 4A. CAL-selected clinical utility, B=11, across banks.**",
                      "**Table 4A. Historical Scheme A CAL-selected outcome utility, B=11, across banks.**")
    text=text.replace("qMV projection component ms/row", "qMV full-path reference ms/row")
    text=text.replace("Tuned3D: qMV projection component only; saved 3D combination not separately timed",
                      "Tuned3D unmeasured; qMV full-path reference is not an isolated shared component")
    text=text.replace("Tuned3D shares the six INT projections and moment calculation with qMV; the saved 3D logistic combination is identified, while its separate incremental latency is not timed.",
                      "Tuned3D shares the six INT projections and moment calculation with qMV, but the measured qMV path also includes qMV-specific ratio/mapping work. Its full-path timing cannot isolate a shared component or give Tuned3D total latency; Tuned3D remains unmeasured.")
    text=text.replace("Tuned3D uses the same six linear projections/moments as qMV plus the saved 3D logistic combination; only the qMV shared projection component has a measured timing here.",
                      "Tuned3D uses the same six linear projections/moments as qMV, followed by a different 3D combination. Only the related qMV full path was timed; its number is a reference, not a measured shared-component time or a lower bound on Tuned3D.")
    text=text.replace("**Table 4B. Partial Scoring-Path & Setup Cost.**",
                      "All-EVAL timing rows benchmark computation on the full feature batch; calibration and filtering validity were evaluated only on D0-positive alerts, not established for non-alert probability outputs.\n\n**Table 4B. Partial Scoring-Path & Setup Cost.**")
    text=text.replace("`Table4_partial_scoring_cost.csv`", "`Table4_partial_scoring_cost_labeled.csv`")
    text=text.replace("`Table4_clinical_utility.csv`", "`Table4_clinical_utility_labeled.csv`")
    text=text.replace("patient row bootstrap intervals condition on a fixed observed EVAL cohort; leave-one-AUDIT intervals describe reference-column sensitivity only.",
                      "patient-row intervals resample the empirical EVAL distribution conditional on fitted models and saved references; stratified AUDIT delete-one results separately describe finite-reference MC sensitivity.")
    text=text.replace("Patient row bootstrap intervals condition on a fixed observed EVAL cohort; leave-one-AUDIT intervals describe reference-column sensitivity only.",
                      "Patient-row intervals resample the empirical EVAL distribution conditional on fitted models and saved references; stratified AUDIT delete-one results separately describe finite-reference MC sensitivity.")
    text=text.replace("## 6. Locked table and artifact map","## 6. Current table sources, pending final author lock")
    text=text.replace("`Table1_paired_effects.csv`","`Table1_paired_effects_reviewed.csv`")
    text=text.replace("`C2_audit_loao_summary.csv`","`C2_stratified_audit_summary.csv`")
    text=text.replace("`C2_audit_loao_replicates.csv`","`C2_stratified_audit_deletions.csv`")
    (OUT/"MANUSCRIPT_CORE_REVIEWED.md").write_text(text)


def main():
    start=time.monotonic(); OUT.mkdir(exist_ok=True)
    c1=review_c1();print("C1: maps, CAL standardization and metrics replayed without fitting",flush=True)
    datasets,_,identity=work._make_datasets()
    points=pd.read_csv(work.OUT/"Table1_main_per_bank.csv").set_index(["dataset","operation","bank","method"])
    point_diff=0.
    for ds,data in datasets.items():
        for op,cases in data["operations"].items():
            for bank,case in cases.items():
                for name,q in case["q"].items():
                    row=points.loc[(ds,op,bank,name)]
                    for key,value in independent_metrics(q,case["audit"]).items():
                        column={"Brier":"Brier_finite_AUDIT","AUC_flip":"AUC"}.get(key,key)
                        point_diff=max(point_diff,abs(value-float(row[column])))
    assert point_diff<1e-12
    boot=review_bootstrap(datasets);print("C2: bootstrap summaries/seed weights/selected full replicates checked",flush=True)
    jack=review_jackknife(datasets);print("C2: independent-bank AUDIT stratification corrected",flush=True)
    clinical=pd.read_csv(work.OUT/"Table4_clinical_utility.csv")
    clinical["representation"]="Scheme_A_Scaled_OHE"
    clinical["scope"]="historical S4 bootstrap utility; different models/alarms from Scheme D Table 1; not outcome validation of the final Scheme D method"
    clinical.to_csv(OUT/"Table4_clinical_utility_labeled.csv",index=False)
    cost=pd.read_csv(work.OUT/"Table4_partial_scoring_cost.csv")
    cost=cost.rename(columns={"online_shared_qMV_projection_ms_per_input_row":"qMV_fullpath_reference_CPU_ms_per_input_row"})
    cost.loc[cost.route=="Update_INT6_Tuned3D","scope"]="Tuned3D total and shared components unmeasured; related qMV full path includes qMV-specific mapping and is only a reference"
    cost.to_csv(OUT/"Table4_partial_scoring_cost_labeled.csv",index=False)
    render_reviewed_manuscript()
    summary={"status":"NUMERICAL_REVIEW_COMPLETE_WITH_STRATIFIED_MC_AND_SCOPE_CORRECTIONS",
             "C1":c1,"main_table_max_metric_difference":point_diff,"C2_bootstrap":boot,"C2_AUDIT":jack,
             "theory_counterexample":posterior_counterexample(),"new_prediction_model_fits":0,"new_CAL_fits":0,
             "public_Credit_D0_identity_inference_batches":6,"new_random_bootstrap_weight_vectors":0,
             "scope":"existing fixed-seed weight vectors replayed; no new random outer/model simulation; no ICU/confirmation or raw EVAL outcome values loaded",
             "review_wall_seconds":time.monotonic()-start}
    (OUT/"review_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
