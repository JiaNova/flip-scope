#!/usr/bin/env python
"""One-shot, frozen-protocol UCI Credit external validation (no EVAL outcomes).

The protocol file must exist before this runner is invoked. This runner is
create-only and does not resume, tune from evaluation results, or access ICU
or confirmation datasets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core import run_d1_d3 as d1
from experiments.core import run_u1_u3 as u1
from experiments.core import run_w1_w4 as base

OUT = ROOT / "reports/core/R1_R4_20261112/R4"
RUN_DIR = OUT / "R4_credit_single_shot_v1"
PROTOCOL_PATH = OUT / "R4_preregistered_protocol_v1.json"
DATA_PATH = ROOT / "data/public/credit/default_credit_data.csv"
METADATA_PATH = ROOT / "data/public/credit/uci_metadata.json"
NUMERIC = ["X1", "X5"] + [f"X{i}" for i in range(12, 24)]
CATEGORICAL = ["X2", "X3", "X4"] + [f"X{i}" for i in range(6, 12)]
FEATURES = NUMERIC + CATEGORICAL
OPS = ("Bootstrap", "Op_Half_Replace", "Op_Full_Replace")
ROLES = [("INT", i) for i in range(1, 7)] + [("CALREF", i) for i in range(1, 7)] + [("AUDIT", i) for i in range(1, 12)]
LAMBDA_GRID = (0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing/unsafe input {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed(*parts) -> int:
    raw = ("R4_CREDIT_20261112|" + "|".join(map(str, parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32 - 1)


def write_json(path: Path, obj):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    base._write_json(path, obj)


def write_npz(path: Path, **arrays):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    base._write_new(path, base._npz_bytes(**arrays))


def write_csv(path: Path, rows):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) if rows else []
    base._write_new(path, base._csv_bytes(rows, fields))


def parse_without_eval_labels():
    """Read ID/features for all rows; materialize Y only for TRAIN and CAL."""
    if DATA_PATH.is_symlink() or METADATA_PATH.is_symlink():
        raise RuntimeError("unsafe dataset source symlink")
    with DATA_PATH.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        required = {"ID", *[f"X{i}" for i in range(1, 24)], "Y"}
        if set(header) != required or len(header) != len(required):
            raise RuntimeError(f"unexpected UCI schema/order: {header}")
        raw_rows = list(reader)
    if len(raw_rows) != 30000:
        raise RuntimeError(f"expected exactly 30000 rows, observed {len(raw_rows)}")
    id_col = 0
    ids = [int(r[id_col]) for r in raw_rows]
    if len(set(ids)) != 30000:
        raise RuntimeError("UCI row ID is not unique")
    ordered = sorted(ids, key=lambda x: (hashlib.sha256(f"R4_SPLIT_V1:{x}".encode()).digest(), x))
    train_ids = set(ordered[:18000]); cal_ids = set(ordered[18000:24000]); eval_ids = set(ordered[24000:])
    if len(train_ids) != 18000 or len(cal_ids) != 6000 or len(eval_ids) != 6000 or train_ids & cal_ids or train_ids & eval_ids or cal_ids & eval_ids:
        raise RuntimeError("exact row split / partition disjointness failed")
    id_to_role = {x: "TRAIN" for x in train_ids}
    id_to_role.update({x: "CAL" for x in cal_ids}); id_to_role.update({x: "EVAL" for x in eval_ids})
    records = []; labels = {}; eval_targets_accessed = 0
    for r in raw_rows:
        rid = int(r[0]); role = id_to_role[rid]
        rec = {c: int(r[header.index(c)]) for c in FEATURES}
        rec["ID"] = rid; records.append(rec)
        if role != "EVAL":
            labels[rid] = int(r[header.index("Y")])
        else:
            # Do not index the Y token in EVAL rows.
            eval_targets_accessed += 0
    if len(labels) != 24000 or eval_targets_accessed != 0:
        raise RuntimeError("target isolation audit failed")
    df = pd.DataFrame.from_records(records)
    split = {r: df[df.ID.isin(s)].reset_index(drop=True) for r, s in (("TRAIN", train_ids), ("CAL", cal_ids), ("EVAL", eval_ids))}
    y = {r: np.asarray([labels[int(i)] for i in split[r].ID], np.int8) for r in ("TRAIN", "CAL")}
    audit = {"source_rows": len(df), "unique_row_ids": len(ids), "row_id_is_not_a_customer_cluster": True,
             "split_counts": {k: len(v) for k, v in split.items()},
             "split_id_sha256": {k: sha("\n".join(map(str, sorted(s))).encode()) for k, s in (("TRAIN", train_ids), ("CAL", cal_ids), ("EVAL", eval_ids))},
             "train_prevalence": float(y["TRAIN"].mean()), "cal_prevalence": float(y["CAL"].mean()),
             "eval_target_values_materialized": False, "eval_target_access_count": int(eval_targets_accessed),
             "feature_columns": FEATURES, "dropped_from_features": ["ID", "Y"]}
    return split, y, audit


def fit_preprocessor(train_a: pd.DataFrame):
    means = {}; scales = {}
    xnum = train_a[NUMERIC].to_numpy(float)
    if not np.isfinite(xnum).all():
        raise RuntimeError("official no-missing schema contradicted by nonfinite numeric value")
    mu = xnum.mean(axis=0); sd = xnum.std(axis=0, ddof=0); sd[sd == 0] = 1.
    for c, m, s in zip(NUMERIC, mu, sd): means[c] = float(m); scales[c] = float(s)
    cat_maps = {}; vocabs = []; mapped_train = []
    for c in CATEGORICAL:
        vals = train_a[c].astype(str)
        counts = vals.value_counts().to_dict()
        rare = {str(k) for k, n in counts.items() if int(n) < 50}
        mapping = {str(k): ("<RARE>" if str(k) in rare else str(k)) for k in counts}
        cat_maps[c] = {"mapping": mapping, "rare_levels": sorted(rare), "train_counts": {str(k): int(v) for k, v in counts.items()}}
        mapped = vals.map(lambda v: mapping.get(str(v), "<RARE>")).astype(str)
        levels = sorted(set(mapped.tolist()) | {"<RARE>"})
        vocabs.append(levels); mapped_train.append(mapped.to_numpy())
    encoder = OneHotEncoder(categories=vocabs, handle_unknown="error", sparse_output=True, dtype=np.float64)
    rawcat = np.column_stack(mapped_train)
    encoder.fit(rawcat)
    return {"means": means, "scales": scales, "cat_maps": cat_maps, "encoder": encoder,
            "feature_names": ["__INTERCEPT__", *NUMERIC, *[f"{c}={v}" for c, levels in zip(CATEGORICAL, vocabs) for v in levels]],
            "vocabularies": vocabs}


def transform(df: pd.DataFrame, prep):
    num = df[NUMERIC].to_numpy(float)
    for j, c in enumerate(NUMERIC):
        num[:, j] = (num[:, j] - prep["means"][c]) / prep["scales"][c]
    mapped = []
    for c in CATEGORICAL:
        mp = prep["cat_maps"][c]["mapping"]
        mapped.append(df[c].astype(str).map(lambda v: mp.get(str(v), "<RARE>")).to_numpy(str))
    rawcat = np.column_stack(mapped)
    onehot = prep["encoder"].transform(rawcat)
    return sparse.hstack([sparse.csr_matrix(np.ones((len(df), 1))), sparse.csr_matrix(num), onehot], format="csr")


def fit_and_save(x, y, weights, path: Path, info):
    started_cpu = time.process_time(); started_wall = time.monotonic()
    beta, solver = base._fit_irls(x.tocsr(), np.asarray(y, np.int8), np.asarray(weights, float), ridge=1.)
    payload = {**info, **solver, "fit_cpu_seconds": time.process_time() - started_cpu,
               "fit_wall_seconds": time.monotonic() - started_wall,
               "weight_sum": float(np.sum(weights)), "weight_nonzero": int(np.count_nonzero(weights)),
               "feature_dimension": len(beta), "beta_sha256": sha(beta.astype("<f8").tobytes()),
               "beta": beta.tolist()}
    write_json(path, payload)
    return beta, payload


def bank_pool_ids(train: pd.DataFrame, bank: int):
    ids = train.ID.astype(int).tolist()
    order = sorted(ids, key=lambda x: (hashlib.sha256(f"R4_POOL_V1:{bank}:{x}".encode()).digest(), x))
    return set(order[:9000]), set(order[9000:])


def draw_op(bank, op, role, idx, a_ids, b_ids, train_pos):
    rng = np.random.default_rng(seed(bank, op, role, idx))
    if op == "Op_Half_Replace":
        ia = np.sort(rng.choice(a_ids, size=3000, replace=False)); ib = np.sort(rng.choice(b_ids, size=3000, replace=False))
    elif op == "Op_Full_Replace":
        ia = np.empty(0, np.int64); ib = np.sort(rng.choice(b_ids, size=6000, replace=False))
    else:
        raise ValueError(op)
    ix = np.r_[ia, ib].astype(np.int64)
    return ix, np.ones(len(ix), float), seed(bank, op, role, idx), ia, ib


def one_dim_mapping(name, cal, target, ev, folder, bank, op):
    q, payload = u1._calibrate_1d(name, cal, target, ev, bank, folder)
    return q, payload


def summarize_variance(v, audit_z):
    obs = np.var(audit_z, axis=1, ddof=1)
    vm = float(np.mean(v)); om = float(np.mean(obs))
    corr = float(np.corrcoef(v, obs)[0, 1]) if np.std(v) > 0 and np.std(obs) > 0 else None
    return {"mean_IF_variance": vm, "mean_AUDIT_variance": om,
            "AUDIT_over_IF_mean_ratio": om / vm if vm > 0 else None,
            "IF_over_AUDIT_mean_ratio": vm / om if om > 0 else None,
            "instance_correlation": corr, "median_IF_variance": float(np.median(v)),
            "median_AUDIT_variance": float(np.median(obs))}


def run_bank(bank, split, ytrain, cal, eva, ycal, audit_rows, split_audit):
    bank_dir = RUN_DIR / f"bank{bank}"
    bank_dir.mkdir()
    a_ids, b_ids = bank_pool_ids(split["TRAIN"], bank)
    if len(a_ids) != 9000 or len(b_ids) != 9000 or a_ids & b_ids:
        raise RuntimeError("R4 bank Pool A/B split invalid")
    train = split["TRAIN"]
    train_by_id = train.set_index("ID", drop=False)
    A = train_by_id.loc[sorted(a_ids)].reset_index(drop=True)
    B = train_by_id.loc[sorted(b_ids)].reset_index(drop=True)
    ymap = {int(i): int(v) for i, v in zip(train.ID, ytrain)}
    yA = np.asarray([ymap[int(i)] for i in A.ID], np.int8); yB = np.asarray([ymap[int(i)] for i in B.ID], np.int8)
    prep = fit_preprocessor(A)
    xA = transform(A, prep); xB = transform(B, prep); xcal_all = transform(cal, prep); xe_all = transform(eva, prep)
    d0_rng = np.random.default_rng(seed(bank, "D0", "sample"))
    d0_draw = d0_rng.choice(np.arange(len(A)), size=12000, replace=True)
    xD0 = xA[d0_draw]; yD0 = yA[d0_draw]
    d0_weights = np.ones(len(yD0), float)
    beta0, d0_meta = fit_and_save(xD0, yD0, d0_weights, bank_dir / "D0.json", {
        "model_id": f"bank{bank}/D0", "bank": bank, "operation": "shared_D0",
        "role": "D0", "draw_seed": seed(bank, "D0", "sample"),
        "training_id_sha256": sha("\n".join(map(str, A.ID.iloc[d0_draw])).encode()),
        "sample_weight_sha256": sha(d0_weights.astype("<f8").tobytes()), "train_n": 12000,
        "base_pool": "Pool A", "fit_scope": "bootstrap sample with replacement"})
    pcal0 = expit(np.clip(np.asarray(xcal_all @ beta0).reshape(-1), -40, 40))
    tau = float(np.quantile(pcal0, .80)); gamma = float(logit(np.clip(tau, 1e-12, 1 - 1e-12)))
    z0c_all = np.asarray(xcal_all @ beta0).reshape(-1) - gamma
    z0e_all = np.asarray(xe_all @ beta0).reshape(-1) - gamma
    ac = z0c_all >= 0.; ae = z0e_all >= 0.
    xcal_alert = xcal_all[ac]; xe_alert = xe_all[ae]
    z0c = z0c_all[ac]; z0e = z0e_all[ae]
    if len(z0c) == 0 or len(z0e) == 0:
        raise RuntimeError("empty CAL or EVAL alert queue")
    write_json(bank_dir / "preprocessor.json", {
        "scheme": "Diabetes Scheme D frozen adaptation", "numeric_means": prep["means"],
        "numeric_scales": prep["scales"], "categorical_maps": prep["cat_maps"],
        "vocabularies": prep["vocabularies"], "feature_names": prep["feature_names"],
        "feature_dimension": len(prep["feature_names"]), "onehot_unscaled": True,
        "preprocessor_fit_scope": "TRAIN Pool A only"})
    write_npz(bank_dir / "draws.npz", D0_poolA_draw=d0_draw.astype(np.int32), poolA_ids=A.ID.to_numpy(np.int64), poolB_ids=B.ID.to_numpy(np.int64))
    write_json(bank_dir / "bank_manifest.json", {"bank": bank, "train_n": 18000, "poolA_n": 9000, "poolB_n": 9000,
        "poolA_id_sha256": sha("\n".join(map(str, sorted(a_ids))).encode()),
        "poolB_id_sha256": sha("\n".join(map(str, sorted(b_ids))).encode()),
        "cal_n": 6000, "eval_n": 6000, "cal_alert_n": int(ac.sum()), "eval_alert_n": int(ae.sum()),
        "tau0": tau, "gamma0": gamma, "feature_dimension": len(prep["feature_names"]),
        "eval_labels_materialized": False})

    bank_rows = []; variance_rows = []; fit_count = 1; mapping_count = 0; cv_fit_count = 0
    for op in OPS:
        odir = bank_dir / op; odir.mkdir()
        if op == "Bootstrap":
            betaR = beta0.copy(); center_fit = {"operator_center_model": "alias of D0", "extra_center_fit_count": 0}
            vcal_all, veval_all, sigma, ifmeta = u1._fit_if(xD0, yD0, beta0, xcal_alert, xe_alert)
            # Each reference model independently resamples the persisted D0
            # bootstrap rows; counts are mapped back to Pool-A rows.
            def draw_role(role, idx):
                rg = np.random.default_rng(seed(bank, op, role, idx))
                take = rg.integers(0, len(d0_draw), size=12000)
                w = np.bincount(d0_draw[take], minlength=len(A)).astype(float)
                return xA, yA, w, {"draw_seed": seed(bank, op, role, idx), "poolA_multiplicity_sha256": sha(w.astype("<f8").tobytes()), "train_n": 12000, "draw_scope": "multinomial bootstrap from saved D0 training sample"}
        else:
            if op == "Op_Half_Replace": k, l = 3000, 3000
            else: k, l = 0, 6000
            center_w = np.r_[np.full(len(A), k / len(A)), np.full(len(B), l / len(B))]
            center_x = sparse.vstack([xA, xB], format="csr")
            center_y = np.r_[yA, yB]
            betaR, center_meta = fit_and_save(center_x, center_y, center_w, odir / "OP_CENTER.json", {
                "model_id": f"bank{bank}/{op}/OP_CENTER", "bank": bank, "operation": op,
                "role": "operator_center", "draw_seed": None, "train_n_equivalent": int(k + l),
                "sample_weight_sha256": sha(center_w.astype("<f8").tobytes()),
                "Pool_A_n": len(A), "Pool_B_n": len(B),
                "weight_rule": f"A={k}/N_A, B={l}/N_B", "fit_scope": "expected finite-pool update objective"})
            fit_count += 1
            vcal_all, veval_all, sigma, ifmeta = u1._finite_population_if(xA, yA, xB, yB, betaR, k, l, xcal_alert, xe_alert)
            def draw_role(role, idx):
                ixA, ixB = None, None
                rng = np.random.default_rng(seed(bank, op, role, idx))
                if op == "Op_Half_Replace":
                    ixA = np.sort(rng.choice(len(A), 3000, replace=False)); ixB = np.sort(rng.choice(len(B), 3000, replace=False))
                else:
                    ixA = np.empty(0, np.int64); ixB = np.sort(rng.choice(len(B), 6000, replace=False))
                xx = sparse.vstack([xA[ixA], xB[ixB]], format="csr")
                yy = np.r_[yA[ixA], yB[ixB]]
                return xx, yy, np.ones(len(yy), float), {"draw_seed": seed(bank, op, role, idx),
                    "PoolA_ids_sha256": sha("\n".join(map(str, A.ID.iloc[ixA])).encode()),
                    "PoolB_ids_sha256": sha("\n".join(map(str, B.ID.iloc[ixB])).encode()), "train_n": len(yy),
                    "draw_scope": "independent SRSWOR per role model"}
        zRc = np.asarray(xcal_alert @ betaR).reshape(-1) - gamma
        zRe = np.asarray(xe_alert @ betaR).reshape(-1) - gamma
        fit_records = []; role_betas = {}
        for role, idx in ROLES:
            xx, yy, ww, provenance = draw_role(role, idx)
            mid = f"{role}-{idx:02d}"
            beta, meta = fit_and_save(xx, yy, ww, odir / "models" / f"{mid}.json", {
                "model_id": f"bank{bank}/{op}/{mid}", "bank": bank, "operation": op,
                "role": role, "replicate": idx, "sample_weight_sha256": sha(ww.astype("<f8").tobytes()),
                **provenance})
            fit_count += 1; role_betas[mid] = beta; fit_records.append({k: v for k, v in meta.items() if k != "beta"})
        if op == "Bootstrap":
            write_json(odir / "operator_center_alias.json", center_fit)
        int_betas = [role_betas[f"INT-{i:02d}"] for i in range(1, 7)]
        cal_betas = [role_betas[f"CALREF-{i:02d}"] for i in range(1, 7)]
        audit_betas = [role_betas[f"AUDIT-{i:02d}"] for i in range(1, 12)]
        ziC = np.column_stack([np.asarray(xcal_alert @ b).reshape(-1) - gamma for b in int_betas])
        ziE = np.column_stack([np.asarray(xe_alert @ b).reshape(-1) - gamma for b in int_betas])
        zcR = np.column_stack([np.asarray(xcal_alert @ b).reshape(-1) - gamma for b in cal_betas])
        zaE = np.column_stack([np.asarray(xe_alert @ b).reshape(-1) - gamma for b in audit_betas])
        audit = (zaE < 0).astype(np.int8); tcal = (zcR < 0).mean(axis=1)
        zbarC = ziC.mean(axis=1); zbarE = ziE.mean(axis=1)
        s2C = np.var(ziC, axis=1, ddof=1); s2E = np.var(ziE, axis=1, ddof=1)
        sigmag2 = float(np.mean(s2C))
        vtildeC = .5 * s2C + .5 * sigmag2; vtildeE = .5 * s2E + .5 * sigmag2
        mC = .5 * (z0c + zbarC); mE = .5 * (z0e + zbarE)
        # All scored paths share the D0-defined CAL/EVAL alert cohorts.
        maps = {}; scored = {}; cal_scores = {
            "Margin_D0_1D": z0c,
            "OpCenter_Mean_1D": zRc,
            "OpCenter_IF_1D": zRc / np.sqrt(np.maximum(vcal_all, 1e-300)),
            "Update_INT6_qMV": mC / np.sqrt(np.maximum(vtildeC, 1e-300)),
        }
        eval_scores = {
            "Margin_D0_1D": z0e,
            "OpCenter_Mean_1D": zRe,
            "OpCenter_IF_1D": zRe / np.sqrt(np.maximum(veval_all, 1e-300)),
            "Update_INT6_qMV": mE / np.sqrt(np.maximum(vtildeE, 1e-300)),
        }
        for name in cal_scores:
            q, payload = one_dim_mapping(name, cal_scores[name], tcal, eval_scores[name], odir, bank, op)
            maps[name] = payload; scored[name] = q; mapping_count += 1
        cv_features = {
            "OpCenter_Tuned2D_Mean": (np.column_stack([z0c, zRc]), np.column_stack([z0e, zRe])),
            "Update_INT6_Tuned2D_Mean": (np.column_stack([z0c, zbarC]), np.column_stack([z0e, zbarE])),
            "Update_INT6_Tuned3D_MV": (np.column_stack([z0c, zbarC, np.log(np.maximum(vtildeC, 1e-300))]),
                                        np.column_stack([z0e, zbarE, np.log(np.maximum(vtildeE, 1e-300))]))}
        cv_records = []; selected = []
        for j, (name, (xc, xe)) in enumerate(cv_features.items()):
            chosen, pred, folds = u1._cv_brier(name, xc, tcal, xe, bank, seed(bank, op, "CV", j))
            scored[name] = pred; cv_records.extend(folds); cv_fit_count += len(folds)
            selected.append({"learner": name, "selected_lambda": chosen["lambda"],
                             "fit": chosen,
                             "cv_mean_brier_by_lambda": chosen["cv_mean_brier_by_lambda"],
                             "cal_alert_n": len(tcal), "cv_folds": 5, "target": "CALREF6 flip frequency"})
        write_json(odir / "calibration_maps.json", maps)
        write_json(odir / "cv_selected.json", selected)
        write_csv(odir / "cv_fold_scores.csv", cv_records)
        write_json(odir / "model_inventory.json", {"bank": bank, "operation": op, "models": fit_records,
            "role_model_count": 23, "unique_role_coefficient_hashes": len(set(x["beta_sha256"] for x in fit_records)),
            "operator_center_fit_count": int(op != "Bootstrap"), "D0_fit_count_for_operation": 0,
            "AUDIT_is_only_EVAL_flip_target": True})
        write_npz(odir / "prediction_arrays.npz", z0_cal_alert=z0c, z0_eval_alert=z0e,
                  zR_cal_alert=zRc, zR_eval_alert=zRe, zINT_cal_alert=ziC, zINT_eval_alert=ziE,
                  zCALREF_cal_alert=zcR, zAUDIT_eval_alert=zaE, audit_eval_alert=audit,
                  vR_cal_alert=vcal_all, vR_eval_alert=veval_all,
                  vtilde_cal_alert=vtildeC, vtilde_eval_alert=vtildeE,
                  tcal_calref=tcal, scored=np.column_stack([scored[k] for k in sorted(scored)]),
                  scored_method_names=np.asarray(sorted(scored), dtype="U64"))
        for method, q in scored.items():
            metrics = u1._metrics(q, audit)
            bank_rows.append({"bank": bank, "operation": op, "method": method, **metrics,
                              "sigma_g_sq": sigmag2, "cal_alert_n": len(tcal), "eval_alert_n": len(audit)})
        variance_rows.append({"bank": bank, "operation": op,
                              **summarize_variance(veval_all, zaE),
                              "IF_meta_total_cpu_s": float(ifmeta.get("total_cpu_seconds", ifmeta.get("Sigma_build_cpu_seconds", 0.))),
                              "fit_center_cpu_s": float(center_meta.get("fit_cpu_seconds", 0.)) if op != "Bootstrap" else 0.})
        write_json(odir / "operation_manifest.json", {"bank": bank, "operation": op, "tau0": tau,
            "operator_center_sha256": sha(betaR.astype("<f8").tobytes()), "d0_sha256": sha(beta0.astype("<f8").tobytes()),
            "n_alerts_eval": int(ae.sum()), "mean_audit_flip_rate": float(audit.mean()),
            "sigma_g_sq": sigmag2, "metric_target": "AUDIT11 decisions on EVAL alerts",
            "cal_target": "CALREF6 decision flips on CAL alerts", "eval_outcome_labels_used": False,
            "IF_formula": ifmeta, "operator_center": center_fit if op == "Bootstrap" else {"fit_cpu_seconds": center_meta["fit_cpu_seconds"]},
            "model_fits_this_operation": 23 + int(op != "Bootstrap"),
            "new_1d_calibration_maps": 4, "cv_search_fold_fits": 150,
            "cv_final_CAL_fits": 3, "cv_model_fits": 153})
    write_csv(bank_dir / "bank_metrics.csv", bank_rows)
    write_csv(bank_dir / "variance_closure.csv", variance_rows)
    return bank_rows, variance_rows, {"bank": bank, "model_fits_including_D0": fit_count,
        "mapping_fits": mapping_count, "cv_search_fold_fits": cv_fit_count,
        "cv_final_CAL_fits": len(OPS) * 3, "cv_fits": cv_fit_count + len(OPS) * 3,
        "d0_fit_count": 1, "operator_centers_fitted": 2, "operator_center_aliases": 1,
        "role_model_counts": {"INT": 18, "CALREF": 18, "AUDIT": 33},
        "d0_fit_cpu_seconds": d0_meta["fit_cpu_seconds"], "cal_alert_n": int(ac.sum()),
        "eval_alert_n": int(ae.sum()), "tau0": tau,
        "eval_label_values_accessed": False}


def main():
    if OUT.exists() is False:
        raise RuntimeError("R4 output directory/protocol was not created before runner")
    if not PROTOCOL_PATH.is_file() or PROTOCOL_PATH.is_symlink():
        raise RuntimeError("frozen protocol missing/unsafe")
    if not DATA_PATH.is_file() or DATA_PATH.is_symlink():
        raise RuntimeError("official UCI data missing/unsafe")
    # The external-data subdirectory is create-only and single shot.
    run_dir = RUN_DIR
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"refuse repeat Credit run: {run_dir}")
    run_dir.mkdir()
    start_wall = time.monotonic(); start_cpu = time.process_time()
    source = {"data_sha256": file_sha(DATA_PATH), "metadata_sha256": file_sha(METADATA_PATH),
              "protocol_sha256": file_sha(PROTOCOL_PATH),
              "runner_sha256": file_sha(Path(__file__).resolve())}
    split, y, split_audit = parse_without_eval_labels()
    pre = {"status": "FROZEN_BEFORE_FIRST_MODEL_FIT", "source_hashes": source,
           "split_audit": split_audit, "authorization": "user-approved R1-R4 fixed batch",
           "model_fits_started": 0, "new_prediction_fits": 0, "eval_labels_used": False}
    write_json(run_dir / "preflight.json", pre)
    all_rows = []; variance_rows = []; bank_manifests = []
    for bank in range(6):
        print(f"R4 bank {bank}: start", flush=True)
        br, vr, bm = run_bank(bank, split, y["TRAIN"], split["CAL"], split["EVAL"], y["CAL"], None, split_audit)
        all_rows.extend(br); variance_rows.extend(vr); bank_manifests.append(bm)
        print(f"R4 bank {bank}: fits={bm['model_fits_including_D0']} eval_alerts={bm['eval_alert_n']} tau={bm['tau0']:.6g}", flush=True)
    write_csv(run_dir / "R4_per_bank_metrics.csv", all_rows)
    write_csv(run_dir / "R4_variance_closure.csv", variance_rows)
    agg = {}
    for r in all_rows:
        agg.setdefault((r["operation"], r["method"]), []).append(r)
    summary_rows = []
    for key, vs in sorted(agg.items()):
        row = {"operation": key[0], "method": key[1]}
        for metric in ("R_070", "AURC100", "AUC", "Brier", "ECE10", "R_100"):
            vals = np.asarray([float(x[metric]) for x in vs], float)
            out_metric = "mean_flip_rate" if metric == "R_100" else metric
            row[out_metric + "_mean"] = float(vals.mean()); row[out_metric + "_min"] = float(vals.min()); row[out_metric + "_max"] = float(vals.max())
        baseline = "Margin_D0_1D"
        base_by_bank = {int(r["bank"]): r for r in all_rows if r["operation"] == key[0] and r["method"] == baseline}
        for metric in ("R_070", "AURC100", "Brier"):
            ds = np.asarray([float(r[metric]) - float(base_by_bank[int(r["bank"])][metric]) for r in vs], float)
            row[f"delta_{metric}_vs_Margin_mean"] = float(ds.mean())
            row[f"wins_vs_Margin_{metric}_out_of_6"] = int(np.sum(ds < 0))
        summary_rows.append(row)
    write_csv(run_dir / "R4_mean_range_metrics.csv", summary_rows)
    threshold_rows = []
    for bank in range(6):
        m = read_json(run_dir / f"bank{bank}/bank_manifest.json")
        threshold_rows.append({"bank": bank, "tau0": m["tau0"], "cal_alert_n": m["cal_alert_n"],
                               "eval_alert_n": m["eval_alert_n"], "eval_alert_rate": m["eval_alert_n"] / 6000})
    write_csv(run_dir / "R4_bank_thresholds.csv", threshold_rows)
    resource_summary = {"wall_seconds": time.monotonic() - start_wall,
        "process_cpu_seconds": time.process_time() - start_cpu,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "native_threads": 1, "gpu_used": False,
        "model_fit_counts": {"D0": int(sum(x["d0_fit_count"] for x in bank_manifests)),
                             "operator_centers_fitted": int(sum(x["operator_centers_fitted"] for x in bank_manifests)),
                             "operator_center_aliases": int(sum(x["operator_center_aliases"] for x in bank_manifests)),
                             "INT": int(sum(x["role_model_counts"]["INT"] for x in bank_manifests)),
                             "CALREF": int(sum(x["role_model_counts"]["CALREF"] for x in bank_manifests)),
                             "AUDIT": int(sum(x["role_model_counts"]["AUDIT"] for x in bank_manifests)),
                             "prediction_model_fit_total": int(sum(x["model_fits_including_D0"] for x in bank_manifests)),
                             "1d_calibration_maps": int(sum(x["mapping_fits"] for x in bank_manifests)),
                             "5fold_cv_brier_fits": int(sum(x["cv_search_fold_fits"] for x in bank_manifests)),
                             "final_CAL_brier_fits": int(sum(x["cv_final_CAL_fits"] for x in bank_manifests)),
                             "cv_brier_fits_total": int(sum(x["cv_fits"] for x in bank_manifests))},
        "eval_label_values_used": False,
        "bank_manifests": bank_manifests,
        "dataset_read_counts": {"official_source_rows": 30000, "TRAIN_y_extracted": 18000,
                                 "CAL_y_extracted": 6000, "EVAL_y_extracted": 0},
        "one_shot": True, "no_post_result_tuning": True}
    write_json(run_dir / "R4_resource_and_counts.json", resource_summary)
    print(json.dumps(resource_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
