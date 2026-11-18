#!/usr/bin/env python3
"""Small public pilot: CAL-selected diagonal-plus-low-rank IF covariance."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core import run_d1_d3 as work


def factors(sigma, rank):
    if rank == 0:
        low = np.zeros((len(sigma), 0))
    else:
        values, vectors = eigh(sigma, subset_by_index=[len(sigma) - rank, len(sigma) - 1])
        low = vectors * np.sqrt(np.maximum(values, 0.))[None, :]
    residual = np.diag(sigma) - np.sum(low * low, axis=1)
    assert residual.min() > -1e-8
    return np.maximum(residual, 0.), low


def variance(x, residual, low):
    diag = np.asarray(x.power(2) @ residual).reshape(-1)
    return diag + np.sum(np.asarray(x @ low) ** 2, axis=1)


def main():
    output = work.OUT / "IF_approx_pilot"
    output.mkdir(exist_ok=True)
    selected, _ = work._load_splits()
    train, cal, eva = [selected[key] for key in ("TRAIN", "CAL", "EVAL")]
    scheme = work.SCHEMES[3]
    records, timing, selections = [], [], []
    start = time.perf_counter()
    for bank in range(6):
        _, tr = work._train_draw(train, bank)
        pipe = work._load_pipeline(tr, scheme)
        xc, xe = work._transform(cal, pipe), work._transform(eva, pipe)
        bdir = work.OUT / scheme / f"bank{bank}"
        offline = work.OUT / "D3_online_latency_cachefree_20261111" / scheme / f"bank{bank}/offline_arrays.npz"
        with np.load(offline) as saved:
            sigma, beta = saved["sigma_theta"], saved["theta_D0"]
            ints, gamma = saved["theta_INT"], float(saved["gamma"][0])
            sigma_g = float(saved["sigma_g_sq"][0])
        maps = work._readj(bdir / "d1_calref_maps.json")
        with np.load(bdir / "predictions.npz") as saved:
            z0c = np.asarray(xc @ beta).reshape(-1) - gamma
            z0e = np.asarray(xe @ beta).reshape(-1) - gamma
            assert np.max(np.abs(z0c - saved["z0c"])) < 1e-8
            assert np.max(np.abs(z0e - saved["z0e"])) < 1e-8
            ac, ae = z0c >= 0, z0e >= 0
            target = (saved["zcalref"][ac] < 0).mean(axis=1)
            audit = (saved["zaudit"][ae, :11] < 0).astype(np.int8)
        candidates, fit_rows = {}, []
        for rank in (0, 8, 32):
            residual, low = factors(sigma, rank)
            vc = variance(xc[ac], residual, low)
            sc = z0c[ac] / np.sqrt(np.maximum(vc, 1e-12))
            losses = []
            for ti, vi in KFold(n_splits=5, shuffle=True, random_state=20261200 + bank).split(sc):
                q, _ = work._calibrated(sc[ti], target[ti], sc[vi])
                losses.extend((q - target[vi]) ** 2)
            _, mapping = work._calibrated(sc, target, sc)
            candidates[rank] = (residual, low, mapping)
            fit_rows.append({"rank": rank, "CAL_CV_Brier": float(np.mean(losses)), "mapping": mapping})
        rank = min(fit_rows, key=lambda row: (row["CAL_CV_Brier"], row["rank"]))["rank"]
        selections.append({"bank": bank, "selected_rank": rank, "candidates": fit_rows})
        (output / f"bank{bank}_CAL_selection.json").write_text(json.dumps(selections[-1], indent=2) + "\n")
        qm = work.base._apply_cal_map(maps["Margin"], z0e[ae])
        for r, (residual, low, mapping) in candidates.items():
            v = variance(xe[ae], residual, low)
            q = work.base._apply_cal_map(mapping, z0e[ae] / np.sqrt(np.maximum(v, 1e-12)))
            records.append({"bank": bank, "method": f"DiagLowRank_{r}", "selected_on_CAL": r == rank,
                            **work._metrics(q, audit, qm)})
        for method in ("Margin", "ConditionalIF"):
            if method == "Margin": q = qm
            else:
                v = np.asarray(xe[ae].multiply(np.asarray(xe[ae] @ sigma)).sum(axis=1)).reshape(-1)
                q = work.base._apply_cal_map(maps[method], z0e[ae] / np.sqrt(np.maximum(v, 1e-12)))
            records.append({"bank": bank, "method": method, "selected_on_CAL": False,
                            **work._metrics(q, audit, qm)})
        residual, low, approx_map = candidates[rank]
        np.savez_compressed(output / f"bank{bank}_approximation.npz", residual_diag=residual, low_rank_factor=low)
        def predict(method, scenario):
            x = xe
            z = np.asarray(x @ beta).reshape(-1) - gamma
            if scenario == "D0_ALERTS":
                mask = z >= 0
                x, z = x[mask], z[mask]
            if method == "Margin": return work.base._apply_cal_map(maps["Margin"], z)
            if method == "Six_INT_qMV":
                zi = np.asarray(x @ ints) - gamma
                m = .5 * (z + zi.mean(axis=1))
                v = .5 * np.var(zi, axis=1, ddof=1) + .5 * sigma_g
                return work.base._apply_cal_map(maps["qMV"], m / np.sqrt(v))
            if method == "Full_IF":
                v = np.asarray(x.multiply(np.asarray(x @ sigma)).sum(axis=1)).reshape(-1)
                mapping = maps["ConditionalIF"]
            else:
                v, mapping = variance(x, residual, low), approx_map
            return work.base._apply_cal_map(mapping, z / np.sqrt(np.maximum(v, 1e-12)))
        for scenario in ("ALL_EVAL", "D0_ALERTS"):
            for method in ("Margin", "Six_INT_qMV", "Full_IF", "CAL_selected_approx_IF"):
                timing.append({"bank": bank, "scenario": scenario, "method": method,
                               "selected_rank": rank, **work._time_call(lambda: predict(method, scenario), repeat=10)})
        print(f"Approximation pilot bank{bank}, CAL selected rank={rank}", flush=True)
    pd.DataFrame(records).to_csv(output / "metrics_by_bank.csv", index=False)
    pd.DataFrame(timing).to_csv(output / "timings.csv", index=False)
    summary = {"status": "EXPLORATORY_PILOT_COMPLETE", "prediction_model_fits": 0,
               "new_CAL_mapping_fits": 108, "CAL_selected_ranks": [s["selected_rank"] for s in selections],
               "timing_repetitions": 10, "wall_seconds": time.perf_counter() - start,
               "limits": ["Scheme D only, shared previously examined EVAL", "standard covariance approximation, no novelty claim",
                          "timings recompute variance from X and parameters; preprocessing excluded",
                          "analysis reads saved AUDIT targets outside timing; this process is not the original D3 no-cache-open audit"]}
    (output / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
