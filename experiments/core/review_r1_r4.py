#!/usr/bin/env python3
"""Review R1-R4 caches and restore missing selected CAL heads, without model retraining."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core import run_r1_r3_replay as r123
from experiments.core import run_r4_credit_external as r4

OUT = r123.OUT / "review"


def readj(path):
    return json.loads(path.read_text())


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def metrics(q, audit):
    """Independent count-based AUC, probability-binned ECE, fractional-tie risk."""
    q = np.asarray(q, float)
    audit = np.asarray(audit)
    if (q.ndim != 1 or not len(q) or audit.ndim != 2 or audit.shape[0] != len(q)
            or not audit.shape[1] or not np.isfinite(q).all()
            or np.any((q < 0) | (q > 1)) or not np.isin(audit, [0, 1]).all()):
        raise ValueError("invalid probability scores or binary reference matrix")
    t = audit.mean(axis=1)
    levels, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
    flips = np.bincount(inverse, weights=t, minlength=len(levels))
    cumulative_n = np.r_[0, np.cumsum(counts)]
    cumulative_y = np.r_[0, np.cumsum(flips)]
    def risk(c):
        target = c * len(q)
        boundary = min(np.searchsorted(cumulative_n[1:], target, side="left"), len(counts)-1)
        frac = (target - cumulative_n[boundary]) / counts[boundary]
        return float((cumulative_y[boundary] + frac * flips[boundary]) / target)
    bins = np.minimum(np.floor(q * 10).astype(int), 9)
    ece = sum(abs(float(np.sum(q[bins == b] - t[bins == b]))) / len(q) for b in range(10))
    pos = flips * audit.shape[1]
    neg = counts * audit.shape[1] - pos
    denominator = float(pos.sum() * neg.sum())
    auc = float(np.sum(pos * (np.cumsum(neg) - .5 * neg)) / denominator) if denominator else None
    return {"R_070": risk(.7), "AURC100": float(np.mean([risk(j/100) for j in range(1, 101)])),
            "Brier": float(np.mean((q[:, None] - audit) ** 2)), "ECE10": float(ece), "AUC_flip": auc}


def shapley(cells, metric):
    values = np.asarray(cells, float)
    if values.shape != (2, 2, 2):
        raise ValueError("three-factor cube required")
    out = np.zeros(3)
    for order in itertools.permutations(range(3)):
        here = [0, 0, 0]
        for factor in order:
            nxt = here.copy(); nxt[factor] = 1
            out[factor] += (values[tuple(nxt)] - values[tuple(here)]) / 6
            here = nxt
    if not np.isclose(out.sum(), values[1,1,1] - values[0,0,0], atol=1e-12):
        raise AssertionError(metric)
    return out


def review_r2():
    rows = []; old = table(r123.OUT / "R2_cross_attribution_per_bank.csv")
    deltas = {m: 0. for m in metrics(np.array([.2]), np.array([[0, 1]]))}
    maps_meta = []
    for bank in range(6):
        pdir = r123.U1 / f"bank{bank}/prospective"
        va = r123.load_npz(pdir / "prospective_scores.npz")
        for op in ("Op_Half_Replace", "Op_Full_Replace"):
            odir = r123.U1 / f"bank{bank}/{op}"
            a = r123.load_npz(odir / "prediction_arrays.npz")
            paths = {("PoolA_Bootstrap", "raw"): pdir / "calibration_Prospective_Margin_D0.json",
                     ("PoolA_Bootstrap", "if"): pdir / "calibration_Prospective_D0_IF_1D.json",
                     ("Update_CALREF6", "raw"): odir / "calibration_OpCenter_Mean_1D.json",
                     ("Update_CALREF6", "if"): odir / "calibration_OpCenter_IF_1D.json"}
            for center, scale, source in itertools.product(("D0", "operator"), ("raw", "if"), ("PoolA_Bootstrap", "Update_CALREF6")):
                score = a["z0_eval_alert"] if center == "D0" else a["zR_eval_alert"]
                if scale == "if":
                    variance = va["v0_eval_alert"] if center == "D0" else a["vR_eval_alert"]
                    score = score / np.sqrt(np.maximum(variance, 1e-300))
                payload = readj(paths[source, scale])
                q = r123.apply_map(payload["mapping"], score)
                mm = metrics(q, a["audit_eval_alert"])
                previous = next(r for r in old if int(r["bank"]) == bank and r["operation"] == op
                                and (r["center"], r["score_scale"], r["calibration_source"]) == (center, scale, source))
                for key in deltas:
                    deltas[key] = max(deltas[key], abs(mm[key] - float(previous[key])))
                rows.append({"bank": bank, "operation": op, "center": center, "score_scale": scale,
                             "calibration_source": source, **mm})
                maps_meta.append({"bank": bank, "operation": op, "center": center, "score_scale": scale,
                                  "calibration_source": source, "map_path": str(paths[source, scale].relative_to(ROOT)),
                                  "map_native_center": "D0" if source == "PoolA_Bootstrap" else "operator",
                                  "map_transported_across_center": (source == "PoolA_Bootstrap") != (center == "D0")})
    save("R2_metrics_corrected.csv", rows)
    means = []
    for key, group in itertools.groupby(sorted(rows, key=lambda r: (r["operation"], r["center"], r["score_scale"], r["calibration_source"])),
                                       key=lambda r: (r["operation"], r["center"], r["score_scale"], r["calibration_source"])):
        group = list(group)
        means.append({**dict(zip(("operation", "center", "score_scale", "calibration_source"), key)),
                      **{name+"_mean": float(np.mean([r[name] for r in group])) for name in deltas}})
    save("R2_mean_metrics_corrected.csv", means)
    save("R2_map_transport.csv", maps_meta)
    attribution = []
    for bank in range(6):
        for op in ("Op_Half_Replace", "Op_Full_Replace"):
            for key in deltas:
                cube = np.empty((2,2,2))
                for i, center in enumerate(("D0", "operator")):
                    for j, scale in enumerate(("raw", "if")):
                        for k, source in enumerate(("PoolA_Bootstrap", "Update_CALREF6")):
                            cube[i,j,k] = next(r[key] for r in rows if r["bank"] == bank and r["operation"] == op
                                              and (r["center"], r["score_scale"], r["calibration_source"]) == (center, scale, source))
                vals = shapley(cube, key)
                attribution.append({"bank": bank, "operation": op, "metric": key,
                                    "center_and_matching_variance_context": vals[0], "IF_scaling": vals[1],
                                    "transported_map_source": vals[2], "total": cube[1,1,1]-cube[0,0,0]})
    save("R2_shapley_corrected.csv", attribution)
    return {"cells": len(rows), "max_absolute_changes_from_original": deltas,
            "transported_map_cells": sum(r["map_transported_across_center"] for r in maps_meta),
            "interpretation": "descriptive transported-map decomposition, not isolated calibration-target or causal effects"}


def review_r4():
    split, y, _ = r4.parse_without_eval_labels()
    source = table(r4.RUN_DIR / "R4_per_bank_metrics.csv")
    stat = {"models_replayed": 0, "max_logit_difference": 0., "max_variance_difference": 0.,
            "max_gradient_inf": 0., "max_metric_difference": 0., "max_1d_probability_difference": 0.,
            "max_restored_CV_probability_difference": 0., "restored_CV_heads": 0,
            "saved_final_CV_fits_missing": 0, "CV_fold_solves": 0, "CV_final_fits_derived_from_executed_code": 0,
            "EVAL_y_extracted": 0, "new_predictive_fits": 0, "new_calibration_fits": 0}
    metric_rows = []; variance_rows = []
    restored_path = OUT / "R4_restored_selected_CAL_heads.json"
    restored = readj(restored_path) if restored_path.exists() else {}
    def check_fit(path, x, yy, weights):
        obj = readj(path); beta = np.asarray(obj["beta"])
        gradient = np.asarray(x.T @ (weights * (expit(x @ beta) - yy))).reshape(-1)
        gradient[1:] += beta[1:]
        stat["models_replayed"] += 1
        stat["max_gradient_inf"] = max(stat["max_gradient_inf"], float(np.max(np.abs(gradient))))
        if r4.sha(weights.astype("<f8").tobytes()) != obj["sample_weight_sha256"]:
            raise AssertionError(f"weight identity: {path}")
        return beta
    for bank in range(6):
        bd = r4.RUN_DIR / f"bank{bank}"
        draws = r123.load_npz(bd / "draws.npz")
        train = split["TRAIN"].set_index("ID", drop=False)
        A = train.loc[draws["poolA_ids"]].reset_index(drop=True)
        B = train.loc[draws["poolB_ids"]].reset_index(drop=True)
        expected_a, expected_b = r4.bank_pool_ids(split["TRAIN"], bank)
        assert set(A.ID) == expected_a and set(B.ID) == expected_b
        ymap = dict(zip(split["TRAIN"].ID, y["TRAIN"]))
        ya = np.asarray([ymap[i] for i in A.ID]); yb = np.asarray([ymap[i] for i in B.ID])
        prep = r4.fit_preprocessor(A)  # Reconstruct TRAIN-only transforms; no predictive/map fit.
        xa, xb = r4.transform(A, prep), r4.transform(B, prep)
        xc, xe = r4.transform(split["CAL"], prep), r4.transform(split["EVAL"], prep)
        draw0 = np.random.default_rng(r4.seed(bank, "D0", "sample")).choice(len(A), 12000, replace=True)
        assert np.array_equal(draw0, draws["D0_poolA_draw"])
        beta0 = check_fit(bd / "D0.json", xa[draw0], ya[draw0], np.ones(12000))
        meta = readj(bd / "bank_manifest.json")
        gamma = meta["gamma0"]
        pc = expit(np.clip(np.asarray(xc @ beta0).reshape(-1), -40, 40))
        assert abs(float(np.quantile(pc, .8))-meta["tau0"]) < 1e-14
        zc = np.asarray(xc @ beta0).reshape(-1)-gamma
        ze = np.asarray(xe @ beta0).reshape(-1)-gamma
        xc, xe = xc[zc >= 0], xe[ze >= 0]
        for op in r4.OPS:
            od = bd / op; arrays = r123.load_npz(od / "prediction_arrays.npz")
            if op == "Bootstrap":
                center = beta0
                vc, ve, _, _ = r4.u1._fit_if(xa[draw0], ya[draw0], beta0, xc, xe)
            else:
                k, l = (3000,3000) if op == "Op_Half_Replace" else (0,6000)
                xall = sparse.vstack([xa, xb], format="csr"); yall = np.r_[ya,yb]
                ww = np.r_[np.full(len(A), k/len(A)), np.full(len(B), l/len(B))]
                center = check_fit(od / "OP_CENTER.json", xall, yall, ww)
                vc, ve, _, _ = r4.u1._finite_population_if(xa, ya, xb, yb, center, k, l, xc, xe)
            for candidate, cached in ((vc, arrays["vR_cal_alert"]), (ve, arrays["vR_eval_alert"])):
                stat["max_variance_difference"] = max(stat["max_variance_difference"], float(np.max(np.abs(candidate-cached))))
            preds = {"INT": [], "CALREF": [], "AUDIT": []}
            for role, index in r4.ROLES:
                rng = np.random.default_rng(r4.seed(bank, op, role, index))
                if op == "Bootstrap":
                    take = rng.integers(0, len(draw0), 12000)
                    weights = np.bincount(draw0[take], minlength=len(A)).astype(float)
                    xx, yy = xa, ya
                else:
                    ia = np.sort(rng.choice(len(A), 3000, replace=False)) if op == "Op_Half_Replace" else np.empty(0, int)
                    ib = np.sort(rng.choice(len(B), 3000 if op == "Op_Half_Replace" else 6000, replace=False))
                    xx = sparse.vstack([xa[ia], xb[ib]], format="csr"); yy = np.r_[ya[ia], yb[ib]]
                    weights = np.ones(len(yy))
                    info = readj(od / "models" / f"{role}-{index:02d}.json")
                    for key, ids in (("PoolA_ids_sha256", A.ID.iloc[ia]), ("PoolB_ids_sha256", B.ID.iloc[ib])):
                        assert r4.sha("\n".join(map(str, ids)).encode()) == info[key]
                beta = check_fit(od / "models" / f"{role}-{index:02d}.json", xx, yy, weights)
                preds[role].append((np.asarray(xc @ beta).reshape(-1)-gamma, np.asarray(xe @ beta).reshape(-1)-gamma))
            comparisons = [(np.asarray(xc @ beta0).reshape(-1)-gamma, arrays["z0_cal_alert"]),
                           (np.asarray(xe @ beta0).reshape(-1)-gamma, arrays["z0_eval_alert"]),
                           (np.asarray(xc @ center).reshape(-1)-gamma, arrays["zR_cal_alert"]),
                           (np.asarray(xe @ center).reshape(-1)-gamma, arrays["zR_eval_alert"])]
            for role, suffix, side in (("INT", "zINT_cal_alert", 0), ("INT", "zINT_eval_alert", 1),
                                       ("CALREF", "zCALREF_cal_alert", 0), ("AUDIT", "zAUDIT_eval_alert", 1)):
                comparisons.append((np.column_stack([p[side] for p in preds[role]]), arrays[suffix]))
            for calculated, cached in comparisons:
                stat["max_logit_difference"] = max(stat["max_logit_difference"], float(np.max(np.abs(calculated-cached))))
            assert np.array_equal(arrays["audit_eval_alert"], arrays["zAUDIT_eval_alert"] < 0)
            assert np.array_equal(arrays["tcal_calref"], (arrays["zCALREF_cal_alert"] < 0).mean(axis=1))
            names = arrays["scored_method_names"].tolist()
            for j, name in enumerate(names):
                q = arrays["scored"][:,j]; mm = metrics(q, arrays["audit_eval_alert"])
                old = next(r for r in source if int(r["bank"]) == bank and r["operation"] == op and r["method"] == name)
                stat["max_metric_difference"] = max(stat["max_metric_difference"], max(abs(mm[key]-float(old["AUC" if key == "AUC_flip" else key])) for key in mm))
                metric_rows.append({"bank": bank, "operation": op, "method": name, **mm})
                mp = od / f"calibration_{name}.json"
                if mp.exists():
                    z = arrays["z0_eval_alert"] if name == "Margin_D0_1D" else arrays["zR_eval_alert"]
                    if name == "OpCenter_IF_1D": z = z / np.sqrt(arrays["vR_eval_alert"])
                    if name == "Update_INT6_qMV": z = .5*(arrays["z0_eval_alert"]+arrays["zINT_eval_alert"].mean(axis=1))/np.sqrt(arrays["vtilde_eval_alert"])
                    replay = r123.apply_map(readj(mp)["mapping"], z)
                    stat["max_1d_probability_difference"] = max(stat["max_1d_probability_difference"], float(np.max(np.abs(q-replay))))
            cv = readj(od / "cv_selected.json")
            stat["saved_final_CV_fits_missing"] += sum("fit" not in fit for fit in cv)
            stat["CV_fold_solves"] += len(table(od / "cv_fold_scores.csv"))
            stat["CV_final_fits_derived_from_executed_code"] += len(cv)
            features = []
            for side in ("cal", "eval"):
                z0 = arrays[f"z0_{side}_alert"]; zR = arrays[f"zR_{side}_alert"]
                zm = arrays[f"zINT_{side}_alert"].mean(axis=1)
                lv = np.log(np.maximum(arrays[f"vtilde_{side}_alert"], 1e-300))
                features.append({"OpCenter_Tuned2D_Mean": np.column_stack([z0,zR]),
                                 "Update_INT6_Tuned2D_Mean": np.column_stack([z0,zm]),
                                 "Update_INT6_Tuned3D_MV": np.column_stack([z0,zm,lv])})
            for entry in cv:
                name = entry["learner"]; key = f"bank{bank}/{op}/{name}"
                if key not in restored:
                    stat["new_calibration_fits"] += 1
                    fit = r4.u1._fit_brier(features[0][name], arrays["tcal_calref"], entry["selected_lambda"])
                    restored[key] = {"fit": fit, "source_lambda": entry["selected_lambda"],
                                     "scope": "technical restoration from original CAL and selected lambda; no CV or EVAL selection"}
                    restored_path.write_text(json.dumps(restored, indent=2, allow_nan=False)+"\n")
                pred = r4.u1._predict_brier(restored[key]["fit"], features[1][name])
                diff = float(np.max(np.abs(pred-arrays["scored"][:,names.index(name)])))
                stat["max_restored_CV_probability_difference"] = max(stat["max_restored_CV_probability_difference"], diff)
                stat["restored_CV_heads"] += 1
            variance_rows.append({"bank": bank, "operation": op, **r4.summarize_variance(ve, arrays["zAUDIT_eval_alert"])})
    save("R4_metrics_replayed.csv", metric_rows); save("R4_variance_replayed.csv", variance_rows)
    pairs = []
    for op in r4.OPS:
        for baseline in ("Margin_D0_1D", "OpCenter_Mean_1D", "Update_INT6_Tuned3D_MV"):
            group = {int(r["bank"]): r for r in metric_rows if r["operation"] == op and r["method"] == baseline}
            for key in ("R_070", "AURC100", "Brier", "ECE10"):
                diffs = np.asarray([r[key]-group[r["bank"]][key] for r in metric_rows if r["operation"] == op and r["method"] == "OpCenter_IF_1D"])
                pairs.append({"operation": op, "method": "OpCenter_IF_1D", "baseline": baseline, "metric": key,
                              "paired_mean": diffs.mean(), "paired_min": diffs.min(), "paired_max": diffs.max(),
                              "sd_across_banks": diffs.std(ddof=1), "wins_out_of_6": int(np.sum(diffs < 0))})
    save("R4_strong_baseline_paired.csv", pairs)
    return stat


def review_bootstrap_baseline():
    new = table(r123.D1 / "D2_lambda_boundary_extension/metrics_by_bank.csv")
    out = []
    for baseline in ("Margin_CALREF", "Learned3D_MV_Unconstrained_CV"):
        for metric in ("R_070", "AURC100", "Brier"):
            rows = [r for r in new if r["scheme"] == "Scheme_D_Merged_Rare50_Unscaled"]
            a = {int(r["bank"]): r for r in rows if r["learner"] == "Conditional_IF_1D"}
            b = {int(r["bank"]): r for r in rows if r["learner"] == baseline}
            ds = np.asarray([float(a[i][metric])-float(b[i][metric]) for i in range(6)])
            out.append({"baseline": baseline, "metric": metric, "paired_mean": ds.mean(),
                        "paired_min": ds.min(), "paired_max": ds.max(), "wins_out_of_6": int(np.sum(ds < 0))})
    save("R3_bootstrap_latest_baselines.csv", out)


def review_cost_scope():
    source = table(r123.OUT / "R3_cycle_cost_per_bank.csv")
    unique = {}
    for row in source:
        key = (row["operation"], row["cohort"], row["route"], row["bank"])
        if key in unique:
            assert row == unique[key]
        unique[key] = row
    groups = {}
    for row in unique.values():
        groups.setdefault((row["operation"], row["cohort"], row["route"]), []).append(row)
    out = []
    for key, rows in sorted(groups.items()):
        assert {int(r["bank"]) for r in rows} == set(range(6))
        out.append({**dict(zip(("operation", "cohort", "route"), key)), "banks_n": len(rows),
                    "setup_CPU_seconds": np.mean([float(r["setup_cpu_seconds"]) for r in rows]),
                    "online_CPU_ms_per_input_row": np.mean([float(r["online_cpu_ms_per_input_patient"]) for r in rows]),
                    "scope": "partial scoring-path cost; existing D0 and CALREF assumed; CV and preprocessing fit excluded",
                    "INT_method": "1D qMV, not tuned 3D" if key[2] == "Update_INT6" else "not applicable",
                    "raw_preprocess": "separate borrowed D3 timing, not measured on this U1 end-to-end path"})
    save("R3_partial_cost_scope.csv", out)
    return {"original_rows": len(source), "unique_bank_route_rows": len(unique),
            "groups": len(out), "new_timing_runs": 0,
            "scope": "partial, not full cycle cost of strongest tuned methods"}


def main():
    OUT.mkdir(exist_ok=True)
    start = time.monotonic()
    r2 = review_r2(); print("R2 metric and map-source review complete", flush=True)
    r4_stats = review_r4(); print("R4 model/prediction/metric replay complete", flush=True)
    assert r4_stats["models_replayed"] == 432
    assert r4_stats["max_gradient_inf"] < 2e-5
    assert r4_stats["max_logit_difference"] < 1e-9
    assert r4_stats["max_variance_difference"] < 1e-9
    assert r4_stats["max_metric_difference"] < 1e-12
    assert r4_stats["max_1d_probability_difference"] < 1e-12
    assert r4_stats["max_restored_CV_probability_difference"] < 1e-12
    review_bootstrap_baseline()
    cost = review_cost_scope()
    summary = {"status": "REVIEWED_WITH_SCIENTIFIC_REPORTING_CORRECTIONS", "R2": r2, "R3_cost": cost, "R4": r4_stats,
               "review_wall_seconds": time.monotonic()-start,
               "new_predictive_fits": 0, "new_calibration_fits_this_review_invocation": r4_stats["new_calibration_fits"],
               "technical_CAL_restorations_cumulative": r4_stats["restored_CV_heads"],
               "clinical_or_confirmation_data_access": 0,
               "note": "Public Credit inference was replayed. Missing CAL heads restored using fixed selected lambdas only. Original reports/caches retained; no table lock."}
    (OUT / "review_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
