#!/usr/bin/env python3
"""Complete the CAL regularization boundary using saved D1 predictions only."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core import run_d1_d3 as work

NEW_LAMBDAS = (0., 1e-6, 1e-5)


def make_features(pred, iv):
    ac, ae = pred["z0c"] >= 0, pred["z0e"] >= 0
    pairs = []
    for is_cal, mask in ((True, ac), (False, ae)):
        z = pred["z0c" if is_cal else "z0e"][mask]
        zi = pred["zic" if is_cal else "zie"][mask]
        zb = zi.mean(axis=1)
        global_var = np.var(pred["zic"][ac], axis=1, ddof=1).mean()
        v = .5 * np.var(zi, axis=1, ddof=1) + .5 * global_var
        m = .5 * (z + zb)
        vif = iv["v_if_cal_alert" if is_cal else "v_if_eval_alert"]
        assert len(vif) == len(z)
        lv, li = np.log(np.maximum(v, 1e-300)), np.log(np.maximum(vif, 1e-300))
        pairs.append({
            "Learned2D_IF_Linear": np.column_stack((z, li)),
            "Learned_IF_Interaction": np.column_stack((z, li, z * li, z / np.sqrt(np.maximum(vif, 1e-300)))),
            "Learned2D_Mean_CV": np.column_stack((z, zb)),
            "Learned3D_MV_Unconstrained_CV": np.column_stack((z, zb, lv)),
            "Learned_MV_Interaction_CV": np.column_stack((z, zb, lv, m * lv, (m < 0) * lv)),
        })
    return ac, ae, pairs[0], pairs[1]


def main():
    output = work.OUT / "D2_lambda_boundary_extension"
    output.mkdir(exist_ok=True)
    old_cv = pd.read_csv(work.OUT / "D2_CAL_CV/D2_cv_fold_losses.csv")
    old_metrics = pd.read_csv(work.OUT / "D2_CAL_CV/D2_per_bank_metrics.csv")
    stats = {"new_CV_fits": 0, "new_full_CAL_fits": 0, "prediction_model_fits": 0,
             "added_lambdas": list(NEW_LAMBDAS), "selection_uses_EVAL": False}
    records, folds, fits = [], [], []
    start = time.perf_counter()
    try:
        for scheme in (work.SCHEMES[0], work.SCHEMES[3]):
            for bank in range(6):
                bdir = work.OUT / scheme / f"bank{bank}"
                pred = work._loadnpz(bdir / "predictions.npz")
                iv = work._loadnpz(bdir / "if_arrays.npz")
                ac, ae, xcal, xeval = make_features(pred, iv)
                target = (pred["zcalref"][ac] < 0).mean(axis=1)
                # No EVAL outcome or AUDIT value is consulted for selection.
                chosen = []
                for name, xc in xcal.items():
                    old = old_cv[(old_cv.scheme == scheme) & (old_cv.bank == bank) & old_cv.learner.eq(name)]
                    scores = old.groupby("lambda").holdout_brier.mean().to_dict()
                    for lam in NEW_LAMBDAS:
                        losses = []
                        for fold, (tr, va) in enumerate(KFold(n_splits=5, shuffle=True, random_state=20261200 + bank).split(xc)):
                            stats["new_CV_fits"] += 1
                            fit, _ = work._standardized_brier_fit(xc[tr], target[tr], lam)
                            q = work._brier_predict(fit, xc[va])
                            loss = float(np.mean((q - target[va]) ** 2))
                            losses.append(loss)
                            folds.append({"scheme": scheme, "bank": bank, "learner": name,
                                          "lambda": lam, "fold": fold, "holdout_brier": loss})
                        scores[lam] = float(np.mean(losses))
                    lam = min(scores, key=lambda key: (scores[key], key))
                    stats["new_full_CAL_fits"] += 1
                    fit, _ = work._standardized_brier_fit(xc, target, lam)
                    item = {"scheme": scheme, "bank": bank, "learner": name,
                            "selected_lambda": lam, "cv_scores": scores, "fit": fit}
                    chosen.append(item)
                    fits.append(item)
                # Publish learned parameters before evaluating the saved AUDIT.
                (output / f"{scheme}_bank{bank}_fits.json").write_text(json.dumps(chosen, indent=2, allow_nan=False) + "\n")
                audit = (pred["zaudit"][ae, :11] < 0).astype(np.int8)
                maps = work._readj(bdir / "d1_calref_maps.json")
                qm = work.base._apply_cal_map(maps["Margin"], pred["z0e"][ae])
                for item in chosen:
                    q = work._brier_predict(item["fit"], xeval[item["learner"]])
                    records.append({"scheme": scheme, "bank": bank, "learner": item["learner"],
                                    "selected_lambda": item["selected_lambda"], **work._metrics(q, audit, qm)})
                print(f"Completed {scheme} bank{bank}; CV fits={stats['new_CV_fits']}", flush=True)
        fixed = old_metrics[old_metrics.learner.isin(["Margin_CALREF", "Fused_Mean_1D", "qMV_CALREF", "Conditional_IF_1D"])]
        metrics = pd.concat([fixed, pd.DataFrame(records)], ignore_index=True)
        metrics.to_csv(output / "metrics_by_bank.csv", index=False)
        fields = ["R_070", "AURC100", "AUC_flip", "Brier", "ECE10"]
        summary = metrics.groupby(["scheme", "learner"])[fields].agg(["mean", "min", "max"])
        summary.columns = ["_".join(col) for col in summary.columns]
        summary.reset_index().to_csv(output / "mean_range_metrics.csv", index=False)
        pd.DataFrame(folds).to_csv(output / "added_CV_fold_losses.csv", index=False)
        stats["selected_lambda_counts"] = pd.Series([item["selected_lambda"] for item in fits]).value_counts().to_dict()
        stats["status"] = "COMPLETED"
    except Exception as exc:
        stats.update(status="FAILED", error=repr(exc))
        raise
    finally:
        stats["wall_seconds"] = time.perf_counter() - start
        (output / "execution_summary.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
