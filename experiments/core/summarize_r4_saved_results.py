#!/usr/bin/env python
"""Create-only aggregation/packaging of completed R4 caches; never fits models."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/core/R1_R4_20261112/R4"
RUN = OUT / "R4_credit_single_shot_v1"
SOURCE_BANKS = OUT


def read_json(p):
    if p.is_symlink() or not p.is_file():
        raise RuntimeError(f"missing/unsafe {p}")
    return json.loads(p.read_text())


def read_csv(p):
    if p.is_symlink() or not p.is_file():
        raise RuntimeError(f"missing/unsafe {p}")
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(p, data):
    if p.exists() or p.is_symlink():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    from experiments.core.run_w1_w4 import _write_new
    _write_new(p, raw)


def write_csv(p, rows):
    import io
    s = io.StringIO(newline="")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(s, fieldnames=fields); w.writeheader(); w.writerows(rows)
    raw = s.getvalue().encode()
    if p.is_symlink():
        raise RuntimeError(f"refuse symlink output: {p}")
    if p.exists():
        if p.read_bytes() != raw:
            raise RuntimeError(f"existing output differs; refuse replacement: {p}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    from experiments.core.run_w1_w4 import _write_new
    _write_new(p, raw)


def sha_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if not RUN.is_dir() or RUN.is_symlink():
        raise RuntimeError("single-shot folder missing/unsafe")
    # First validate all completed banks, then move them into the versioned
    # single-shot folder. Files are moved unchanged; no inference is performed.
    bank_sources = [SOURCE_BANKS / f"bank{i}" for i in range(6)]
    for i, src in enumerate(bank_sources):
        if src.is_symlink() or not src.is_dir():
            raise RuntimeError(f"bank {i} cache is missing/unsafe")
        for op in ("Bootstrap", "Op_Half_Replace", "Op_Full_Replace"):
            inv = read_json(src / op / "model_inventory.json")
            if len(inv["models"]) != 23 or not (src / op / "prediction_arrays.npz").is_file():
                raise RuntimeError(f"bank {i}/{op}: incomplete saved results")
            if len(read_csv(src / op / "cv_fold_scores.csv")) != 150:
                raise RuntimeError(f"bank {i}/{op}: CV audit rows missing")
        if (RUN / f"bank{i}").exists() or (RUN / f"bank{i}").is_symlink():
            raise FileExistsError(RUN / f"bank{i}")
    for i, src in enumerate(bank_sources):
        shutil.move(str(src), str(RUN / f"bank{i}"))

    per_bank = []; var_rows = []; model_cpu = []; map_cpu = []; counts = {"D0": 0, "OP_CENTER": 0, "INT": 0, "CALREF": 0, "AUDIT": 0, "CV": 0, "CAL_MAP": 0}
    for bank in range(6):
        bdir = RUN / f"bank{bank}"
        bm = read_json(bdir / "bank_manifest.json")
        per_bank.extend(read_csv(bdir / "bank_metrics.csv"))
        var_rows.extend(read_csv(bdir / "variance_closure.csv"))
        d0 = read_json(bdir / "D0.json"); model_cpu.append(float(d0["fit_cpu_seconds"])); counts["D0"] += 1
        for op in ("Bootstrap", "Op_Half_Replace", "Op_Full_Replace"):
            od = bdir / op
            folds = read_csv(od / "cv_fold_scores.csv"); counts["CV"] += len(folds)
            inv = read_json(od / "model_inventory.json")
            for row in inv["models"]:
                model_cpu.append(float(row["fit_cpu_seconds"]))
                counts[row["role"]] += 1
            if op != "Bootstrap":
                center = read_json(od / "OP_CENTER.json"); model_cpu.append(float(center["fit_cpu_seconds"])); counts["OP_CENTER"] += 1
            maps = read_json(od / "calibration_maps.json")
            for row in maps.values():
                map_cpu.append(float(row.get("fit_cpu_seconds", 0.)))
                counts["CAL_MAP"] += 1
        if int(bm["eval_alert_n"]) < 1:
            raise RuntimeError("unexpected empty evaluation alert cohort")

    if counts != {"D0": 6, "OP_CENTER": 12, "INT": 108, "CALREF": 108, "AUDIT": 198, "CV": 2700, "CAL_MAP": 72}:
        raise RuntimeError(f"actual count audit failed: {counts}")
    # Replace the erroring first-run post-fit aggregation using saved result
    # artifacts. R_100 is the saved mean AUDIT flip rate.
    grouped = {}
    for r in per_bank:
        grouped.setdefault((r["operation"], r["method"]), []).append(r)
    summary = []
    for key, rows in sorted(grouped.items()):
        out = {"operation": key[0], "method": key[1]}
        for metric in ("R_070", "AURC100", "AUC", "Brier", "ECE10", "R_100"):
            vals = np.asarray([float(r[metric]) for r in rows], float)
            name = "mean_flip_rate" if metric == "R_100" else metric
            out[name + "_mean"] = float(vals.mean()); out[name + "_min"] = float(vals.min()); out[name + "_max"] = float(vals.max())
        baseline_by_bank = {int(r["bank"]): r for r in per_bank if r["operation"] == key[0] and r["method"] == "Margin_D0_1D"}
        for metric in ("R_070", "AURC100", "Brier"):
            ds = np.asarray([float(r[metric]) - float(baseline_by_bank[int(r["bank"])][metric]) for r in rows], float)
            out[f"delta_{metric}_vs_Margin_mean"] = float(ds.mean())
            out[f"wins_vs_Margin_{metric}_out_of_6"] = int(np.sum(ds < 0))
        summary.append(out)
    write_csv(RUN / "R4_mean_range_metrics.csv", summary)
    write_csv(RUN / "R4_per_bank_metrics.csv", per_bank)
    write_csv(RUN / "R4_variance_closure.csv", var_rows)
    thresholds = []
    for bank in range(6):
        bm = read_json(RUN / f"bank{bank}/bank_manifest.json")
        thresholds.append({"bank": bank, "tau0": bm["tau0"], "cal_alert_n": bm["cal_alert_n"],
                           "eval_alert_n": bm["eval_alert_n"], "eval_alert_rate": bm["eval_alert_n"] / 6000})
    write_csv(RUN / "R4_bank_thresholds.csv", thresholds)

    # Base-fit CPU is exact from each persisted model. CV fold fitting was
    # not timed per fold by the original runner; do not estimate it.
    preflight = read_json(RUN / "preflight.json")
    all_output_files = sorted(p for p in RUN.rglob("*") if p.is_file())
    mtime_start = (RUN / "preflight.json").stat().st_mtime_ns
    mtime_end = max((RUN / "R4_per_bank_metrics.csv").stat().st_mtime_ns,
                    (RUN / "R4_variance_closure.csv").stat().st_mtime_ns)
    resource = {"completed_model_fit_counts": counts,
        "prediction_model_fit_total": int(counts["D0"] + counts["OP_CENTER"] + counts["INT"] + counts["CALREF"] + counts["AUDIT"]),
        "persisted_prediction_model_fit_cpu_seconds_sum": float(sum(model_cpu)),
        "persisted_CAL_map_fit_cpu_seconds_sum": float(sum(map_cpu)),
        "CV_fit_cpu_seconds": None,
        "CV_fit_cpu_unavailable_reason": "The runner recorded fold scores/counts but did not persist per-fold CPU timers; no value is imputed.",
        "process_wall_seconds": None,
        "process_wall_unavailable_reason": "The run's post-fit summary raised KeyError before persisting process timing; model artifacts are complete. The saved artifact mtime interval is recorded separately, not mislabeled as process wall time.",
        "preflight_to_result_artifact_mtime_seconds": max(0., (mtime_end - mtime_start) / 1e9),
        "process_cpu_seconds": None,
        "peak_rss_bytes": None,
        "resource_telemetry_status": "PARTIAL: per-model CPU persisted; process CPU/wall/peak RSS were not persisted due post-fit aggregation KeyError",
        "native_threads": 1, "gpu_used": False,
        "dataset_read_counts": {"source_rows": 30000, "TRAIN_y_extracted": 18000, "CAL_y_extracted": 6000, "EVAL_y_extracted": 0},
        "eval_labels_used": False, "single_shot_model_fits_complete": True,
        "model_fits_started_after_protocol_hash": preflight["source_hashes"]["protocol_sha256"],
        "runner_hash_executed": preflight["source_hashes"]["runner_sha256"],
        "runner_hash_after_documented_aggregation_fix": sha_file(ROOT / "reports/core/run_r4_credit_external.py"),
        "post_fit_aggregation_fix": "R_100 is reported as mean_flip_rate; original fit cache not rerun.",
        "run_output_files_count": len(all_output_files)}
    write_json(RUN / "R4_resource_and_counts.json", resource)
    print(json.dumps(resource, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
