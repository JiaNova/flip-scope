#!/usr/bin/env python
"""R1-R3 report tables from literature and frozen U1/D1/D2/D3 artifacts.

This script performs no predictive-model fits. It reads only the previously
saved Diabetes prediction/variance arrays and summary artifacts. R2 applies
the already-fitted Pool-A and Update-CALREF one-dimensional maps as a crossed
diagnostic; the maps themselves are not re-estimated here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core import run_w1_w4 as base

U1 = ROOT / "reports/core/S1_S4_20261111/U1_POOL_UPDATE_20261112"
D1 = ROOT / "reports/core/S1_S4_20261111/D1_D2_D3_20261111"
D3 = D1 / "D3_online_latency_cachefree_20261111"
OUT = ROOT / "reports/core/R1_R4_20261112"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(p: Path):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows):
    if p.exists() or p.is_symlink():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = base._csv_bytes(rows, list(dict.fromkeys(k for r in rows for k in r)))
    base._write_new(p, raw)


def read_json(p: Path):
    if p.is_symlink() or not p.is_file():
        raise RuntimeError(f"unsafe/missing input JSON: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_npz(p: Path):
    if p.is_symlink() or not p.is_file():
        raise RuntimeError(f"unsafe/missing input NPZ: {p}")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def apply_map(mapping, x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite crossed score")
    center = float(mapping["center"])
    scale = float(mapping["scale"])
    a, b = np.asarray(mapping["coefficients"], float)
    if scale <= 0 or b > 1e-12:
        raise ValueError("invalid/nonmonotone frozen CALREF map")
    q = 1. / (1. + np.exp(-np.clip(a + b * (x - center) / scale, -40., 40.)))
    return q


def score_metrics(q, audit):
    q = np.asarray(q, float)
    audit = np.asarray(audit)
    if (q.ndim != 1 or not len(q) or audit.ndim != 2 or audit.shape[0] != len(q)
            or audit.shape[1] == 0 or not np.isfinite(q).all()
            or np.any((q < 0) | (q > 1)) or not np.isin(audit, [0, 1]).all()):
        raise ValueError("paired probability scores and binary AUDIT decisions required")
    audit = audit.astype(np.int8)
    t = audit.mean(axis=1)
    return {
        "R_070": base._risk(q, t, .70),
        "AURC100": float(np.mean([base._risk(q, t, k / 100.) for k in range(1, 101)])),
        "Brier": float(np.mean((q[:, None] - audit) ** 2)),
        # Bin by predicted probability, sum mass-weighted gaps, include q=1.
        "ECE10": base.pub._ece10(q, t),
        "AUC_flip": float(base._auc_finite(q, audit)),
        "n_alerts": int(len(q)),
    }


def r2_replay():
    rows = []
    # Descriptive transport diagnostic: each map was fitted on its native
    # center, so swapping maps also changes the CAL feature distribution.
    # Moreover, switching the standardized center changes its matching IF
    # covariance. These factors do not isolate calibration-target or causal effects.
    for bank in range(6):
        pdir = U1 / f"bank{bank}/prospective"
        pool_maps = {
            "raw": read_json(pdir / "calibration_Prospective_Margin_D0.json")["mapping"],
            "if": read_json(pdir / "calibration_Prospective_D0_IF_1D.json")["mapping"],
        }
        for op in ("Op_Half_Replace", "Op_Full_Replace"):
            odir = U1 / f"bank{bank}/{op}"
            arr = load_npz(odir / "prediction_arrays.npz")
            va = load_npz(pdir / "prospective_scores.npz")
            update_maps = {
                "raw": read_json(odir / "calibration_OpCenter_Mean_1D.json")["mapping"],
                "if": read_json(odir / "calibration_OpCenter_IF_1D.json")["mapping"],
            }
            raw_scores = {"D0": arr["z0_eval_alert"], "operator": arr["zR_eval_alert"]}
            if_scores = {
                "D0": arr["z0_eval_alert"] / np.sqrt(np.maximum(va["v0_eval_alert"], 1e-300)),
                "operator": arr["zR_eval_alert"] / np.sqrt(np.maximum(arr["vR_eval_alert"], 1e-300)),
            }
            audit = arr["audit_eval_alert"].astype(np.int8)
            for center in ("D0", "operator"):
                for scale in ("raw", "if"):
                    score = raw_scores[center] if scale == "raw" else if_scores[center]
                    for cal_source, maps in (("PoolA_Bootstrap", pool_maps), ("Update_CALREF6", update_maps)):
                        # Each source contributes its pre-existing map for the
                        # same score family; applying across centers is the
                        # intentional transport contrast.
                        q = apply_map(maps[scale], score)
                        met = score_metrics(q, audit)
                        rows.append({"bank": bank, "operation": op, "center": center,
                                     "score_scale": scale, "calibration_source": cal_source,
                                     "mapping_family": scale,
                                     "R_070": met["R_070"], "AURC100": met["AURC100"],
                                     "Brier": met["Brier"], "ECE10": met["ECE10"],
                                     "AUC_flip": met["AUC_flip"], "n_alerts": met["n_alerts"],
                                     "audit_target_sha256": hashlib.sha256(audit.tobytes()).hexdigest()})
            # Shapley attribution from the Track-II corner (D0/raw/PoolA) to
            # Track-I corner (operator/IF/update), summing exactly to the
            # observed endpoint contrast for each metric.
            keys = ["R_070", "AURC100", "Brier", "ECE10"]
            lookup = {(r["center"], r["score_scale"], r["calibration_source"]): r for r in rows
                      if r["bank"] == bank and r["operation"] == op}
            # Legacy column names retained for historical CSV consumers;
            # the interpretation metadata below is part of this diagnostic.
            shapley = {m: {f: 0. for f in ("calibration_target_drift", "operator_center", "IF_scale")}
                       for m in keys}
            factors = ("center", "score_scale", "calibration_source")
            level0 = {"center": "D0", "score_scale": "raw", "calibration_source": "PoolA_Bootstrap"}
            level1 = {"center": "operator", "score_scale": "if", "calibration_source": "Update_CALREF6"}
            fmap = {"center": "operator_center", "score_scale": "IF_scale", "calibration_source": "calibration_target_drift"}
            import itertools
            for perm in itertools.permutations(factors):
                cur = dict(level0)
                for f in perm:
                    nxt = dict(cur); nxt[f] = level1[f]
                    before = lookup[(cur["center"], cur["score_scale"], cur["calibration_source"])]
                    after = lookup[(nxt["center"], nxt["score_scale"], nxt["calibration_source"])]
                    for m in keys:
                        shapley[m][fmap[f]] += (after[m] - before[m]) / 6.
                    cur = nxt
            endpoints = {}
            for m in keys:
                before = lookup[("D0", "raw", "PoolA_Bootstrap")][m]
                after = lookup[("operator", "if", "Update_CALREF6")][m]
                endpoints[m] = {"track_ii_corner": before, "track_i_corner": after,
                                "total_track_i_minus_track_ii": after - before,
                                **shapley[m],
                                "shapley_sum": sum(shapley[m].values())}
            endpoints["interpretation"] = {
                "calibration_target_drift": "legacy name: transported map source, not isolated target effect",
                "operator_center": "center plus matching IF covariance when standardized",
                "scope": "descriptive decomposition, not causal attribution"}
            write_json(OUT / f"R2_shapley_bank{bank}_{op}.json", endpoints)
    write_csv(OUT / "R2_cross_attribution_per_bank.csv", rows)
    grouped = {}
    for r in rows:
        k = (r["operation"], r["center"], r["score_scale"], r["calibration_source"])
        grouped.setdefault(k, []).append(r)
    agg = []
    for k, vs in sorted(grouped.items()):
        row = dict(zip(("operation", "center", "score_scale", "calibration_source"), k))
        for m in ("R_070", "AURC100", "Brier", "ECE10", "AUC_flip"):
            vals = np.asarray([v[m] for v in vs], float)
            row[m + "_mean"] = float(vals.mean())
            row[m + "_min"] = float(vals.min())
            row[m + "_max"] = float(vals.max())
        agg.append(row)
    write_csv(OUT / "R2_cross_attribution_mean_range.csv", agg)
    return {"cells": len(rows), "banks": 6, "operations": 2, "model_fits": 0,
            "score_arrays_sha256": {
                str(p.relative_to(ROOT)): sha_file(p)
                for p in sorted(U1.glob("bank*/Op_*/prediction_arrays.npz"))},
            "poola_target_cache_sha256": {
                str(p.relative_to(ROOT)): sha_file(p)
                for p in sorted(U1.glob("bank*/prospective/prospective_scores.npz"))}}


def write_json(p: Path, obj):
    if p.exists() or p.is_symlink():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    base._write_json(p, obj)


def _paired_summary(rows, group_keys, method_key, method_value, baseline_key, baseline_value, metrics, stage):
    idx = {(tuple(r[k] for k in group_keys), r[method_key]): r for r in rows}
    methods = sorted({r[method_key] for r in rows if r[method_key] != baseline_value})
    out = []
    for group in sorted({tuple(r[k] for k in group_keys) for r in rows}, key=lambda x: tuple(map(str, x))):
        for method in methods:
            diffs = {m: [] for m in metrics}
            for b in sorted({r["bank"] for r in rows}):
                lookup_group = list(group)
                if "bank" not in group_keys:
                    lookup_group.insert(0, b)
                # group keys here include bank for true within-bank pairing.
                if "bank" in group_keys and lookup_group[0] != b:
                    continue
                mk = (tuple(lookup_group), method)
                bk = (tuple(lookup_group), baseline_value)
                if mk not in idx or bk not in idx:
                    continue
                for metric in metrics:
                    diffs[metric].append(float(idx[mk][metric]) - float(idx[bk][metric]))
            if not any(diffs[m] for m in metrics):
                continue
            row = {k: v for k, v in zip(group_keys, group)}
            row.update({"stage": stage, "method": method, "baseline": baseline_value})
            for m, vals in diffs.items():
                x = np.asarray(vals, float)
                row[f"delta_{m}_mean"] = float(x.mean()) if len(x) else None
                row[f"delta_{m}_min"] = float(x.min()) if len(x) else None
                row[f"delta_{m}_max"] = float(x.max()) if len(x) else None
                row[f"delta_{m}_sd_across_banks"] = float(x.std(ddof=1)) if len(x) > 1 else None
                row[f"wins_{m}"] = int(np.sum(x < 0)) if len(x) else 0
                row[f"paired_banks_n_{m}"] = int(len(x))
            out.append(row)
    return out


def r3_pairs_and_costs():
    # Same-pool bootstrap: the six-bank Scheme-D CAL-CV table stores the
    # comparators and the tuned 3D mean/variance learner on identical banks.
    boot = read_csv(D1 / "D2_lambda_boundary_extension/metrics_by_bank.csv")
    boot = [r for r in boot if r["scheme"] == "Scheme_D_Merged_Rare50_Unscaled"]
    boot_rows = [{"bank": int(r["bank"]), "method": r["learner"],
                  "R_070": float(r["R_070"]), "AURC100": float(r["AURC100"]),
                  "Brier": float(r["Brier"]), "AUC_flip": float(r["AUC_flip"]),
                  "ECE10": float(r["ECE10"])} for r in boot]
    boot_paired = []
    for baseline in ("Margin_CALREF", "Learned3D_MV_Unconstrained_CV"):
        boot_paired += _paired_summary(boot_rows, ["bank"], "method", None, None, baseline,
                                       ("R_070", "AURC100", "Brier"), "Bootstrap")

    cross = read_csv(U1 / "U1_per_bank_metrics.csv")
    cross = [r for r in cross if r["track"] == "I_Update_Pool_Aware"]
    cross_rows = [{"bank": int(r["bank"]), "operation": r["operation"], "method": r["method"],
                   **{m: float(r[m]) for m in ("R_070", "AURC100", "Brier", "AUC", "ECE10")}}
                  for r in cross]
    cross_paired = []
    for baseline in ("Margin_D0_1D", "OpCenter_Mean_1D"):
        cross_paired += _paired_summary(cross_rows, ["bank", "operation"], "method", None, None, baseline,
                                        ("R_070", "AURC100", "Brier"), "CrossPool")
    write_csv(OUT / "R3_bootstrap_paired_effects.csv", boot_paired)
    write_csv(OUT / "R3_crosspool_paired_effects.csv", cross_paired)

    # Full-cycle timing: use U1's measured 50-repetition Scheme-D setup and
    # online paths. Margin timing comes from the same 50-repetition D3
    # cache-free Scheme-D end-to-end route; its map-only setup is from U1.
    costs = read_csv(U1 / "U3_cost_by_bank_v2.csv")
    d3rows = read_csv(D3 / "D3_online_timings.csv")
    d3rows = [r for r in d3rows if r["scheme"] == "Scheme_D_Merged_Rare50_Unscaled" and
              r["method"] == "Method_1_Margin"]
    preprocess = read_csv(D3 / "D3_preprocess_timings.csv")
    preprocess = [r for r in preprocess if r["scheme"] == "Scheme_D_Merged_Rare50_Unscaled" and
                  r["method"] == "Step_0_Preprocess_raw_to_sparse_X_eval"]
    prep_s = float(np.mean([float(r["cpu_ms_mean"]) for r in preprocess])) / 1000.
    eval_n = 13949.
    prep_cpu_ms_per_row = prep_s * 1000. / eval_n
    cost_rows = []
    for op in ("Op_Half_Replace", "Op_Full_Replace"):
        for bank in range(6):
            maprow = read_json(U1 / f"bank{bank}/{op}/calibration_Margin_D0_1D.json")
            margin_setup = float(maprow.get("fit_cpu_seconds", 0.))
            for cohort in ("all_eval", "d0_alert_only"):
                scenario = "ALL_EVAL" if cohort == "all_eval" else "D0_ALERTS"
                dr = next(x for x in d3rows if int(x["bank"]) == bank and x["scenario"] == scenario)
                margin_online = float(dr["cpu_ms_mean"]) / eval_n
                line = {"bank": bank, "operation": op, "cohort": cohort, "route": "Margin",
                        "setup_cpu_seconds": margin_setup,
                        "setup_components": "already-fitted D0; one existing CALREF map",
                        "online_cpu_ms_per_input_patient": margin_online,
                        "online_source": "50-repeat cache-free D3 Scheme-D Method_1_Margin",
                        "preprocess_cpu_ms_per_input_patient": prep_cpu_ms_per_row,
                        "input_n_for_reference": eval_n}
                cost_rows.append(line)
                for r in costs:
                    if int(r["bank"]) != bank or r["operation"] != op or r["cohort"] != cohort:
                        continue
                    route = r["route"]
                    cost_rows.append({"bank": bank, "operation": op, "cohort": cohort, "route": route,
                        "setup_cpu_seconds": float(r["setup_cpu_seconds"]),
                        "setup_components": r["includes"],
                        "online_cpu_ms_per_input_patient": float(r["online_cpu_ms_per_input_row"]),
                        "online_source": "50-repeat cache-free U3 v2",
                        "preprocess_cpu_ms_per_input_patient": prep_cpu_ms_per_row,
                        "input_n_for_reference": float(r["input_rows"])})
    write_csv(OUT / "R3_cycle_cost_per_bank.csv", cost_rows)

    cost_groups = {}
    for r in cost_rows:
        cost_groups.setdefault((r["operation"], r["cohort"], r["route"]), []).append(r)
    ns = (10000, 50000, 200000, 1000000)
    summary = []
    for key, vs in sorted(cost_groups.items()):
        setup = float(np.mean([r["setup_cpu_seconds"] for r in vs]))
        online = float(np.mean([r["online_cpu_ms_per_input_patient"] for r in vs])) / 1000.
        prep = float(np.mean([r["preprocess_cpu_ms_per_input_patient"] for r in vs])) / 1000.
        row = {"operation": key[0], "cohort": key[1], "route": key[2], "setup_mean_cpu_s": setup,
               "online_mean_cpu_ms_per_patient": online * 1000.,
               "common_preprocess_mean_cpu_ms_per_patient": prep * 1000.,
               "preprocess_reference_full_eval_mean_cpu_ms": prep * eval_n * 1000.,
               "bank_n": len({r["bank"] for r in vs}), "timing_repetitions": 50,
               "cost_scope": "partial; CALREF generation and CV excluded; INT route is 1D qMV",
               "common_preprocess_source": "borrowed D3 reference, not U1 raw-input replay"}
        for n in ns:
            row[f"C_no_common_preprocess_N{n}_s"] = setup + n * online
            row[f"C_with_common_preprocess_N{n}_s"] = setup + n * (online + prep)
        summary.append(row)
    write_csv(OUT / "R3_cycle_cost_mean_range.csv", summary)

    # Crossovers computed from the six-bank mean setup/online costs.
    crossovers = []
    for op in ("Op_Half_Replace", "Op_Full_Replace"):
        for cohort in ("all_eval", "d0_alert_only"):
            group = [r for r in summary if r["operation"] == op and r["cohort"] == cohort]
            byroute = {r["route"]: r for r in group}
            for ra, rb in (("OpCenter_Mean", "OpCenter_IF"), ("OpCenter_Mean", "Update_INT6"),
                           ("OpCenter_IF", "Update_INT6"), ("Margin", "OpCenter_Mean"),
                           ("Margin", "OpCenter_IF"), ("Margin", "Update_INT6")):
                a, b = byroute[ra], byroute[rb]
                da = a["online_mean_cpu_ms_per_patient"]
                db = b["online_mean_cpu_ms_per_patient"]
                den = da - db
                numer = b["setup_mean_cpu_s"] - a["setup_mean_cpu_s"]
                nstar = numer / (den / 1000.) if abs(den) > 1e-15 and numer / den > 0 else None
                crossovers.append({"operation": op, "cohort": cohort, "route_A": ra, "route_B": rb,
                                   "mean_setup_A_s": a["setup_mean_cpu_s"], "mean_setup_B_s": b["setup_mean_cpu_s"],
                                   "mean_online_A_ms_per_patient": da, "mean_online_B_ms_per_patient": db,
                                   "N_star_rows": nstar,
                                   "interpretation": "positive only if route B has higher setup but lower per-patient online CPU; otherwise no positive crossing"})
    write_csv(OUT / "R3_cycle_cost_crossovers.csv", crossovers)
    return {"bootstrap_paired_rows": len(boot_paired), "crosspool_paired_rows": len(cross_paired),
            "cycle_cost_rows": len(cost_rows), "common_preprocess_mean_full_eval_cpu_seconds": prep_s,
            "crossovers": len(crossovers)}


def main():
    if OUT.exists():
        raise FileExistsError(f"create-only output already exists: {OUT}")
    OUT.mkdir(parents=True)
    r2 = r2_replay()
    r3 = r3_pairs_and_costs()
    summary = {"task": "R1-R3 frozen-cache analysis", "model_fits": 0,
               "new_prediction_inference": 0,
               "r2": r2, "r3": r3,
               "source_hashes": {"U1_per_bank_metrics.csv": sha_file(U1/"U1_per_bank_metrics.csv"),
                                 "U3_cost_by_bank_v2.csv": sha_file(U1/"U3_cost_by_bank_v2.csv"),
                                 "D2_per_bank_metrics.csv": sha_file(D1/"D2_CAL_CV/D2_per_bank_metrics.csv"),
                                 "D3_online_timings.csv": sha_file(D3/"D3_online_timings.csv"),
                                 "D3_preprocess_timings.csv": sha_file(D3/"D3_preprocess_timings.csv")}}
    write_json(OUT / "R1_R3_replay_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
