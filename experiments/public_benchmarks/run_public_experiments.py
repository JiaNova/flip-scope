#!/usr/bin/env python3
"""Public Diabetes replication, Adult P1 surrogate, and Adult Logistic P2.

The protocol is fixed in protocol.json.  This runner has explicit stages so
Bank 0 resources can be measured before the remaining banks are fitted.  It
does not read ICU/confirmation data and does not modify the prior Adult P1
models or outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from typing import Any, Mapping

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtr
from scipy.sparse import csr_matrix, hstack
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inflight import public_adult as adult  # noqa: E402
from experiments.adult_support.run_p1_instance_transfer import (  # noqa: E402
    _ece10, compute_tie_aware_risk,
)


PROTOCOL_PATH = ROOT / "reports/public_benchmarks/protocol.json"
OUT_ROOT = PROTOCOL_PATH.parent
DATA_DIR = ROOT / "data/public/diabetes"
RUN_ROOT = ROOT / "runs/public_benchmarks"
ADULT_DATA = ROOT / "data/public/adult"
ADULT_P1 = ROOT / "reports/adult_support/P1_EXECUTION_20261110"
DIABETES_FEATURES = [
    "race", "gender", "age", "weight", "admission_type_id",
    "discharge_disposition_id", "admission_source_id", "time_in_hospital",
    "payer_code", "medical_specialty", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses", "max_glu_serum",
    "A1Cresult", "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed",
]
DIABETES_NUMERIC = (
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
)
DIABETES_CATEGORICAL = tuple(x for x in DIABETES_FEATURES if x not in DIABETES_NUMERIC)
DISPOSITION_EXCLUDED = {11, 13, 14, 19, 20, 21}
P3_METHODS = (
    "margin_logistic", "q_global_1", "q_sqrt2", "int_freq",
    "int_freq_margin_tiebreak", "int_mean_logistic", "q_V", "q_MV_shrunk",
    "q_MV_raw", "q_MV_surrogate",
)
P2_METHODS = (
    "margin_logistic", "q_global_1", "q_sqrt2", "raw_tweedie", "int_freq",
    "int_freq_margin_tiebreak", "int_mean_logistic", "q_V", "q_MV_shrunk",
    "q_MV_raw", "q_IF_analytic",
)
RISK_COVERAGES = (0.50, 0.70, 0.80, 0.85, 0.90, 1.00)
EPS = 1e-12


class ExperimentError(RuntimeError):
    pass


def canonical(obj: Any) -> bytes:
    return (json.dumps(_json_safe(obj), sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_text(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ExperimentError(f"required JSON missing/not regular: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_directory(path: Path, files: Mapping[str, bytes]) -> None:
    adult.controlled.atomic_publish_directory(path, dict(files))


def save_numpy_archive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    write_new_directory(path, {"arrays.npz": buffer.getvalue()})


def load_protocol() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("protocol_id") != "M3R_P3_P2_PUBLIC_REPLICATION_20261110_V1":
        raise ExperimentError("fixed P3/P2 protocol identity mismatch")
    if protocol.get("status") != "FIXED_BEFORE_NEW_MODEL_FITS":
        raise ExperimentError("P3/P2 protocol does not state pre-fit fixed status")
    return protocol


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def stable_group_hash(patient: str, encounter: str) -> str:
    return hashlib.sha256(f"select_{patient}_{encounter}".encode("utf-8")).hexdigest()


def safe_logit(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p <= 0.0) | (p >= 1.0)):
        raise ExperimentError("model probabilities must be finite and in (0,1)")
    return np.log(p) - np.log1p(-p)


def _parse_diabetes_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"encounter_id", "patient_nbr", "discharge_disposition_id", "readmitted"}
    if len(df) != 101766 or df.shape[1] != 50 or not required.issubset(df.columns):
        raise ExperimentError(f"UCI Diabetes schema mismatch: rows={len(df)}, columns={df.shape[1]}")
    if set(DIABETES_FEATURES) != set(df.columns) - {
            "encounter_id", "patient_nbr", "readmitted"}:
        raise ExperimentError("fixed 47-feature list does not match UCI CSV columns")
    disposition = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
    if disposition.isna().any():
        raise ExperimentError("discharge disposition IDs contain missing/non-numeric values")
    excluded_mask = disposition.astype(int).isin(DISPOSITION_EXCLUDED)
    exclusions = {str(code): int(np.count_nonzero(disposition.astype(int).to_numpy() == code))
                  for code in sorted(DISPOSITION_EXCLUDED)}
    eligible = df.loc[~excluded_mask].copy()
    eligible["discharge_disposition_id"] = disposition.loc[~excluded_mask].astype(int).astype(str)
    if not set(eligible["readmitted"].unique()).issubset({"<30", ">30", "NO"}):
        raise ExperimentError("unexpected readmitted value; do not remap silently")
    eligible["y_readmit30"] = (eligible["readmitted"] == "<30").astype(np.int8)

    selected_rows = []
    for patient, part in eligible.groupby("patient_nbr", sort=False):
        ranked = [(stable_group_hash(str(patient), str(encounter)), str(encounter), index)
                  for index, encounter in zip(part.index, part["encounter_id"].tolist())]
        _, _, chosen_index = min(ranked)
        selected_rows.append(chosen_index)
    selected = eligible.loc[selected_rows].copy()
    selected = selected.sort_values("patient_nbr", kind="mergesort").reset_index(drop=True)
    if selected["patient_nbr"].duplicated().any():
        raise AssertionError("patient-level single-encounter selection duplicated patients")
    selected["split_hash_mod_10000"] = selected["patient_nbr"].map(
        lambda p: stable_hash_int(f"split_v2_{p}") % 10000).astype(np.int64)
    selected["split"] = np.select(
        [selected["split_hash_mod_10000"] < 6000,
         selected["split_hash_mod_10000"] < 8000], ["TRAIN", "CAL"], default="EVAL")
    selected["bank_id"] = -1
    train_index = selected.index[selected["split"] == "TRAIN"]
    for row_index in train_index:
        patient = str(selected.at[row_index, "patient_nbr"])
        scores = [stable_hash_int(f"P3_bank_split_20261110_{bank}_{patient}")
                  for bank in range(3)]
        selected.at[row_index, "bank_id"] = int(np.argmin(scores))

    before_patients = int(df["patient_nbr"].nunique())
    after_patients = int(eligible["patient_nbr"].nunique())
    split = {}
    for name in ("TRAIN", "CAL", "EVAL"):
        rows = selected.loc[selected["split"] == name]
        split[name] = {"patients": int(len(rows)),
                       "positive_count": int(rows["y_readmit30"].sum()),
                       "positive_rate": float(rows["y_readmit30"].mean()) if len(rows) else None}
    bank_counts = {str(bank): int(np.count_nonzero(selected["bank_id"].to_numpy() == bank))
                   for bank in range(3)}
    audit = {
        "raw_records": int(len(df)),
        "raw_columns": int(df.shape[1]),
        "unique_patients_before_filter": before_patients,
        "excluded_records_by_discharge_id": exclusions,
        "excluded_record_count": int(excluded_mask.sum()),
        "eligible_records_after_filter": int(len(eligible)),
        "eligible_unique_patients_after_filter": after_patients,
        "selected_single_encounters": int(len(selected)),
        "unique_selected_patients": int(selected["patient_nbr"].nunique()),
        "split_patient_counts_and_positive_rates": split,
        "train_bank_patient_counts": bank_counts,
        "target": "readmitted '<30' positive; '>30' and 'NO' negative",
        "selection": "minimum SHA-256 encounter key per patient; not first visit and not label-selected",
        "patient_split_rule": "SHA-256 split_v2_{patient_nbr} modulo 10000; <6000 TRAIN, <8000 CAL, else EVAL",
        "feature_exclusions": ["encounter_id", "patient_nbr", "readmitted", "y_readmit30",
                               "split_hash_mod_10000", "split", "bank_id"],
        "no_predictions_or_model_fits_before_audit": True,
    }
    return selected, audit


def prepare_diabetes() -> dict:
    protocol = load_protocol()
    if not DATA_DIR.exists():
        url = protocol["P3_diabetes"]["official_archive"]
        request = urllib.request.Request(url, headers={"User-Agent": "M3R-public-research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise ExperimentError(f"UCI source HTTP status {response.status}")
            archive = response.read(30_000_001)
        if len(archive) > 30_000_000:
            raise ExperimentError("UCI source archive exceeds 30MB safety bound")
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            by_base = {Path(n).name: n for n in zipped.namelist() if not n.endswith("/")}
            required = {"diabetic_data.csv", "IDS_mapping.csv"}
            if not required.issubset(by_base):
                raise ExperimentError(f"official UCI archive members missing: {required-set(by_base)}")
            csv_bytes = zipped.read(by_base["diabetic_data.csv"])
            ids_bytes = zipped.read(by_base["IDS_mapping.csv"])
        license_text = (
            "UCI Diabetes 130-US Hospitals for Years 1999-2008\n"
            "Citation: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). "
            "Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5230J.\n"
            "License: Creative Commons Attribution 4.0 International (CC BY 4.0). Attribution required.\n"
            f"Official page: {protocol['P3_diabetes']['source_page']}\n"
            f"Official archive: {url}\n"
        ).encode("utf-8")
        manifest = {
            "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
            "dataset_id": 296,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_page": protocol["P3_diabetes"]["source_page"],
            "source_archive": url,
            "archive_sha256": sha_bytes(archive),
            "diabetic_data_csv_sha256": sha_bytes(csv_bytes),
            "ids_mapping_csv_sha256": sha_bytes(ids_bytes),
            "license": "CC BY 4.0",
            "csv_bytes": len(csv_bytes),
        }
        write_new_directory(DATA_DIR, {
            "uci_archive.zip": archive,
            "diabetic_data.csv": csv_bytes,
            "IDS_mapping.csv": ids_bytes,
            "LICENSE_NOTICE.txt": license_text,
            "download_manifest.json": canonical(manifest),
        })
    manifest = read_json(DATA_DIR / "download_manifest.json")
    for filename, key in (("diabetic_data.csv", "diabetic_data_csv_sha256"),
                          ("IDS_mapping.csv", "ids_mapping_csv_sha256")):
        actual = sha_bytes((DATA_DIR / filename).read_bytes())
        if actual != manifest.get(key):
            raise ExperimentError(f"cached public UCI input hash mismatch: {filename}")
    df = pd.read_csv(DATA_DIR / "diabetic_data.csv", dtype=str, keep_default_na=False,
                     low_memory=False)
    selected, audit = _parse_diabetes_rows(df)
    audit.update({"source_archive_sha256": manifest["archive_sha256"],
                  "diabetic_data_csv_sha256": manifest["diabetic_data_csv_sha256"],
                  "ids_mapping_csv_sha256": manifest["ids_mapping_csv_sha256"],
                  "dataset_license": manifest["license"],
                  "preprocessing_or_model_fit_calls": 0})
    id_map = (DATA_DIR / "IDS_mapping.csv").read_text(encoding="utf-8", errors="replace")
    audit["official_disposition_mapping_text_sha256"] = sha_text(id_map)
    selected_manifest = selected[["patient_nbr", "encounter_id", "split", "y_readmit30", "bank_id"]].copy()
    selected_manifest.insert(0, "patient_key_sha256", selected_manifest["patient_nbr"].map(
        lambda x: sha_text("patient:" + str(x))))
    selected_manifest.insert(1, "encounter_key_sha256", selected_manifest["encounter_id"].map(
        lambda x: sha_text("encounter:" + str(x))))
    selected_manifest = selected_manifest.drop(columns=["patient_nbr", "encounter_id"])
    stream = io.StringIO(newline="")
    selected_manifest.to_csv(stream, index=False)
    write_new_directory(OUT_ROOT / "diabetes_data_audit", {
        "audit.json": canonical(audit),
        "selected_patient_split_manifest.csv": stream.getvalue().encode("utf-8"),
        "protocol_sha256.txt": (sha_bytes(PROTOCOL_PATH.read_bytes()) + "\n").encode(),
    })
    print(json.dumps({"stage": "prepare-diabetes", "status": "READY_BEFORE_MODELS",
                      "audit": audit}, ensure_ascii=False))
    return audit


def load_selected_diabetes() -> tuple[pd.DataFrame, dict]:
    download = read_json(DATA_DIR / "download_manifest.json")
    audit = read_json(OUT_ROOT / "diabetes_data_audit/audit.json")
    if sha_bytes((DATA_DIR / "diabetic_data.csv").read_bytes()) != download["diabetic_data_csv_sha256"]:
        raise ExperimentError("Diabetes source changed after the pre-fit eligibility audit")
    df = pd.read_csv(DATA_DIR / "diabetic_data.csv", dtype=str, keep_default_na=False,
                     low_memory=False)
    selected, recomputed = _parse_diabetes_rows(df)
    if (recomputed["eligible_records_after_filter"] != audit["eligible_records_after_filter"] or
            recomputed["unique_selected_patients"] != audit["unique_selected_patients"] or
            recomputed["split_patient_counts_and_positive_rates"] != audit[
                "split_patient_counts_and_positive_rates"]):
        raise ExperimentError("Diabetes pre-fit audit no longer matches recomputed patient splits")
    return selected, audit


def _diabetes_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for column in DIABETES_NUMERIC:
        values = df[column].replace("?", np.nan)
        result[column] = pd.to_numeric(values, errors="coerce")
    return result


def fit_diabetes_preprocessor(train: pd.DataFrame) -> dict:
    numeric = _diabetes_numeric_frame(train)
    medians = {}
    for column in DIABETES_NUMERIC:
        median = numeric[column].median()
        if not np.isfinite(median):
            raise ExperimentError(f"TRAIN has no observed value for numeric field {column}")
        medians[column] = float(median)
    vocabularies = {}
    for column in DIABETES_CATEGORICAL:
        values = {"__MISSING__", "__UNK__"}
        values.update("__MISSING__" if v in ("", "?") else str(v) for v in train[column])
        vocabularies[column] = sorted(values)
    return {"schema": 1, "fit_rows": int(len(train)), "numeric_medians": medians,
            "categorical_vocabularies": vocabularies,
            "numeric_features": list(DIABETES_NUMERIC),
            "categorical_features": list(DIABETES_CATEGORICAL)}


def transform_diabetes_lgb(df: pd.DataFrame, preprocessor: Mapping) -> pd.DataFrame:
    frame = pd.DataFrame(index=np.arange(len(df)))
    for column in DIABETES_FEATURES:
        if column in DIABETES_NUMERIC:
            values = pd.to_numeric(df[column].replace("?", np.nan), errors="coerce")
            frame[column] = values.fillna(float(preprocessor["numeric_medians"][column])).astype(float).to_numpy()
        else:
            vocab = preprocessor["categorical_vocabularies"][column]
            allowed = set(vocab)
            values = []
            for raw in df[column].astype(str).tolist():
                value = "__MISSING__" if raw in ("", "?") else raw
                values.append(value if value in allowed else "__UNK__")
            ids = {value: index for index, value in enumerate(vocab)}
            codes = [ids[value] for value in values]
            frame[column] = pd.Categorical(codes, categories=list(range(len(vocab))))
    return frame[DIABETES_FEATURES]


def fit_sparse_proxy_preprocessor(train: pd.DataFrame, preprocessor: Mapping):
    numeric = transform_diabetes_lgb(train, preprocessor)[list(DIABETES_NUMERIC)].to_numpy(dtype=float)
    numeric_scaler = StandardScaler().fit(numeric)
    cat_values = np.empty((len(train), len(DIABETES_CATEGORICAL)), dtype=object)
    for j, column in enumerate(DIABETES_CATEGORICAL):
        allowed = set(preprocessor["categorical_vocabularies"][column])
        raw_values = []
        for raw in train[column].astype(str).tolist():
            v = "__MISSING__" if raw in ("", "?") else raw
            raw_values.append(v if v in allowed else "__UNK__")
        cat_values[:, j] = raw_values
    categories = [preprocessor["categorical_vocabularies"][c] for c in DIABETES_CATEGORICAL]
    encoder = OneHotEncoder(categories=categories, handle_unknown="error", sparse_output=True,
                            dtype=np.float64)
    cat_sparse = encoder.fit_transform(cat_values)
    cat_scaler = StandardScaler(with_mean=False).fit(cat_sparse)
    return {"numeric_scaler": numeric_scaler, "encoder": encoder,
            "cat_scaler": cat_scaler}


def transform_sparse_proxy(df: pd.DataFrame, preprocessor: Mapping, fitted: Mapping,
                           z0: np.ndarray, p0: np.ndarray):
    numeric = transform_diabetes_lgb(df, preprocessor)[list(DIABETES_NUMERIC)].to_numpy(dtype=float)
    numeric_scaled = fitted["numeric_scaler"].transform(numeric)
    cat_values = np.empty((len(df), len(DIABETES_CATEGORICAL)), dtype=object)
    for j, column in enumerate(DIABETES_CATEGORICAL):
        allowed = set(preprocessor["categorical_vocabularies"][column])
        vals = []
        for raw in df[column].astype(str).tolist():
            v = "__MISSING__" if raw in ("", "?") else raw
            vals.append(v if v in allowed else "__UNK__")
        cat_values[:, j] = vals
    cats = fitted["cat_scaler"].transform(fitted["encoder"].transform(cat_values))
    extra = np.column_stack((z0, np.abs(z0), p0 * (1.0 - p0)))
    return hstack((csr_matrix(numeric_scaled), cats, csr_matrix(extra)), format="csr")


def _model_seed_p3(bank: int, role: str, idx: int) -> int:
    role_code = {"D0": 0, "INT": 100, "AUDIT": 200}[role]
    return 2026111000 + 10000 * bank + role_code + idx


def _p3_bank_dir(bank: int) -> Path:
    return RUN_ROOT / "diabetes" / f"bank-{bank}"


def _p3_model_dir(bank: int, model_id: str) -> Path:
    return _p3_bank_dir(bank) / "models" / model_id


def _train_lgb_model(model_id: str, train: pd.DataFrame, labels: np.ndarray,
                     weights: np.ndarray, bank: int, seed: int) -> dict:
    params = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
        "num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 40,
        "lambda_l2": 1.0, "feature_fraction": 1.0, "bagging_fraction": 1.0,
        "bagging_freq": 0, "deterministic": True, "force_col_wise": True,
        "num_threads": 1, "device_type": "cpu", "verbosity": -1,
        "seed": seed, "data_random_seed": seed, "feature_fraction_seed": seed,
        "bagging_seed": seed, "drop_seed": seed,
    }
    dataset = lgb.Dataset(train, label=labels, weight=weights,
                          categorical_feature=list(DIABETES_CATEGORICAL),
                          free_raw_data=False)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    booster = lgb.train(params, dataset, num_boost_round=100)
    wall = time.monotonic() - started_wall
    cpu = time.process_time() - started_cpu
    model_bytes = booster.model_to_string().encode("utf-8")
    model_sha = sha_bytes(model_bytes)
    manifest = {"model_id": model_id, "bank_id": bank,
                "role": "CURRENT_SINGLE" if model_id == "D0" else
                        ("INTERNAL_CALIBRATION_REFERENCE" if model_id.startswith("INT") else
                         "HELD_OUT_AUDIT_REFERENCE"),
                "fit_rows": int(len(train)), "fit_weight_sum": float(weights.sum()),
                "seed": int(seed), "boosting_rounds": 100,
                "lightgbm_version": lgb.__version__, "model_sha256": model_sha,
                "fit_wall_seconds": wall, "fit_cpu_seconds": cpu,
                "feature_columns": list(DIABETES_FEATURES),
                "excluded_features": ["encounter_id", "patient_nbr", "readmitted"],
                "gpu_used": False, "native_threads": 1}
    path = _p3_model_dir(bank, model_id)
    write_new_directory(path, {"booster.txt": model_bytes,
                               "model_manifest.json": canonical(manifest)})
    return manifest


def _save_bank_fit_summary(bank: int, summary: dict) -> None:
    path = _p3_bank_dir(bank) / "fit_measurement"
    write_new_directory(path, {"summary.json": canonical(summary)})


def fit_diabetes_bank(bank: int) -> dict:
    if not 0 <= bank < 3:
        raise ValueError("bank must be 0, 1, or 2")
    selected, audit = load_selected_diabetes()
    train = selected.loc[selected["split"] == "TRAIN"].copy()
    train["bank_id"] = pd.to_numeric(train["bank_id"], errors="raise").astype(int)
    bank_train = train.loc[train["bank_id"] == bank].sort_values("patient_nbr", kind="mergesort").copy()
    if len(bank_train) < 1000:
        raise ExperimentError(f"P3 bank {bank} train pool too small: {len(bank_train)}")
    bank_dir = _p3_bank_dir(bank)
    bank_dir.mkdir(parents=True, exist_ok=True)
    if bank_dir.is_symlink():
        raise ExperimentError(f"symlink bank path rejected: {bank_dir}")
    preprocessor = fit_diabetes_preprocessor(bank_train)
    preprocessor_bytes = canonical(preprocessor)
    preproc_dir = bank_dir / "preprocessor"
    if not preproc_dir.exists():
        write_new_directory(preproc_dir, {"preprocessor.json": preprocessor_bytes})
    elif sha_bytes((preproc_dir / "preprocessor.json").read_bytes()) != sha_bytes(preprocessor_bytes):
        raise ExperimentError(f"P3 bank {bank} preprocessor cache mismatch")
    train_frame = transform_diabetes_lgb(bank_train, preprocessor)
    labels = bank_train["y_readmit30"].to_numpy(dtype=np.int8)
    group_ids = bank_train["patient_nbr"].astype(str).tolist()
    if len(set(group_ids)) != len(group_ids):
        raise ExperimentError("P3 bootstrap unit contains duplicate patient IDs")
    pre_fit = {"bank_id": bank,
               "bank_patient_count": int(len(bank_train)),
               "bank_patient_ids_sha256": sha_bytes("\n".join(group_ids).encode()),
               "preprocessor_sha256": sha_bytes(preprocessor_bytes),
               "training_positive_rate": float(labels.mean()),
               "source_audit_sha256": sha_bytes((OUT_ROOT / "diabetes_data_audit/audit.json").read_bytes()),
               "model_fit_attempts": 0, "model_fit_successes": 0,
               "model_fits": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    fit_started_wall, fit_started_cpu = time.monotonic(), time.process_time()
    model_specs = [("D0", "D0", 0)]
    model_specs.extend((f"INT-{j:02d}", "INT", j) for j in range(1, 7))
    model_specs.extend((f"AUDIT-{j:02d}", "AUDIT", j) for j in range(1, 12))
    for model_id, role, idx in model_specs:
        model_dir = _p3_model_dir(bank, model_id)
        if model_dir.exists():
            manifest = read_json(model_dir / "model_manifest.json")
            actual = sha_bytes((model_dir / "booster.txt").read_bytes())
            if actual != manifest.get("model_sha256") or manifest.get("model_id") != model_id:
                raise ExperimentError(f"P3 existing model cache invalid: bank={bank}/{model_id}")
            pre_fit["model_fits"].append({"model_id": model_id, "status": "reused_existing_fit",
                                          "model_sha256": actual})
            continue
        seed = _model_seed_p3(bank, role, idx)
        if model_id == "D0":
            weights = np.ones(len(bank_train), dtype=np.float64)
            multiplicities = None
        else:
            bootstrap_seed = seed + 50000
            rng = np.random.default_rng(bootstrap_seed)
            multiplicities = np.bincount(rng.integers(0, len(bank_train), size=len(bank_train)),
                                         minlength=len(bank_train)).astype(np.int64)
            weights = multiplicities.astype(np.float64)
        pre_fit["model_fit_attempts"] += 1
        manifest = _train_lgb_model(model_id, train_frame, labels, weights, bank, seed)
        if multiplicities is not None:
            weight_bytes = canonical({"patient_ids_sha256": pre_fit["bank_patient_ids_sha256"],
                                      "bootstrap_seed": bootstrap_seed,
                                      "multiplicities": multiplicities.tolist()})
            model_path = model_dir / "bootstrap_weights"
            write_new_directory(model_path, {"weights.json": weight_bytes})
            manifest["bootstrap_seed"] = int(bootstrap_seed)
            manifest["bootstrap_multiplicities_sha256"] = sha_bytes(weight_bytes)
            # The booster is already atomically saved; record the weight digest beside it.
            # A create-only sidecar directory preserves both artifacts without touching the model.
            write_new_directory(model_dir / "fit_details", {
                "fit_details.json": canonical({"bootstrap_seed": bootstrap_seed,
                                                "bootstrap_multiplicities_sha256": sha_bytes(weight_bytes),
                                                "unique_positive_weight_patients": int(np.count_nonzero(multiplicities)),
                                                "bootstrap_weight_sum": int(multiplicities.sum())})})
        pre_fit["model_fit_successes"] += 1
        pre_fit["model_fits"].append({"model_id": model_id, "status": "fit_success",
                                      "model_sha256": manifest["model_sha256"],
                                      "fit_wall_seconds": manifest["fit_wall_seconds"],
                                      "fit_cpu_seconds": manifest["fit_cpu_seconds"]})
    pre_fit["total_fit_wall_seconds"] = float(time.monotonic() - fit_started_wall)
    pre_fit["total_fit_cpu_seconds"] = float(time.process_time() - fit_started_cpu)
    pre_fit["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    pre_fit["status"] = "BANK_FIT_COMPLETE"
    _save_bank_fit_summary(bank, pre_fit)
    print(json.dumps({"stage": f"fit-diabetes-bank-{bank}", "status": pre_fit["status"],
                      "fit_attempts_this_command": pre_fit["model_fit_attempts"],
                      "fit_successes_this_command": pre_fit["model_fit_successes"],
                      "wall_seconds": pre_fit["total_fit_wall_seconds"],
                      "cpu_seconds": pre_fit["total_fit_cpu_seconds"],
                      "peak_rss_bytes": pre_fit["peak_rss_bytes"]}, ensure_ascii=False))
    return pre_fit


def _assert_bank0_resource_ok() -> dict:
    summary = read_json(_p3_bank_dir(0) / "fit_measurement/summary.json")
    if summary.get("status") != "BANK_FIT_COMPLETE":
        raise ExperimentError("P3 Bank 0 fit summary is not complete")
    limit = int(load_protocol()["P3_diabetes"]["banks"][
        "bank0_resource_review_threshold_seconds"])
    if float(summary["total_fit_wall_seconds"]) > limit:
        raise ExperimentError(f"P3 Bank 0 exceeded predeclared {limit}s controllability threshold; remaining banks not started")
    return summary


def fit_diabetes_rest() -> list[dict]:
    bank0 = _assert_bank0_resource_ok()
    summaries = [bank0]
    for bank in (1, 2):
        summaries.append(fit_diabetes_bank(bank))
    return summaries


def _p3_load_bank_models(bank: int) -> dict[str, lgb.Booster]:
    models = {}
    for model_id in ["D0"] + [f"INT-{i:02d}" for i in range(1, 7)] + [
            f"AUDIT-{i:02d}" for i in range(1, 12)]:
        path = _p3_model_dir(bank, model_id)
        if path.is_symlink() or not (path / "booster.txt").is_file():
            raise ExperimentError(f"P3 model cache miss; stop without fitting: bank={bank}/{model_id}")
        manifest = read_json(path / "model_manifest.json")
        if sha_bytes((path / "booster.txt").read_bytes()) != manifest.get("model_sha256"):
            raise ExperimentError(f"P3 model cache hash mismatch: bank={bank}/{model_id}")
        models[model_id] = lgb.Booster(model_file=str(path / "booster.txt"))
    return models


def _risk_metrics(scores: np.ndarray, targets: np.ndarray, auc_labels: np.ndarray,
                  margin: np.ndarray, q_sqrt2: np.ndarray) -> dict:
    q = np.asarray(scores, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    y = np.asarray(auc_labels, dtype=np.int8)
    if q.ndim != 1 or t.shape != q.shape or y.ndim != 2 or y.shape[0] != q.size:
        raise ExperimentError("risk metric arrays are not aligned")
    risk = {f"R_{int(c*100):03d}": compute_tie_aware_risk(q, t, c)
            for c in RISK_COVERAGES}
    risk["AURC100"] = float(np.mean([compute_tie_aware_risk(q, t, k/100) for k in range(1, 101)]))
    risk["AURC_70_100"] = float(np.mean([compute_tie_aware_risk(q, t, k/100)
                                         for k in range(70, 101)]))
    try:
        auc = float(roc_auc_score(y.reshape(-1), np.repeat(q, y.shape[1])))
    except ValueError:
        auc = None
    return {"n_alert_groups": int(len(q)), **risk, "AUC_vs_AUDIT_flip": auc,
            "Brier_finite_AUDIT": float(np.mean((q[:, None]-y) ** 2)),
            "ECE10": _ece10(q, t),
            "paired_delta_Brier_vs_margin_logistic": float(np.mean(
                q*q - margin*margin - 2*(q-margin)*t)),
            "paired_delta_Brier_vs_q_sqrt2": float(np.mean(
                q*q - q_sqrt2*q_sqrt2 - 2*(q-q_sqrt2)*t)),
            "mean_AUDIT_flip_rate": float(np.mean(t))}


def _p3_proxy_fit(cal_features: pd.DataFrame, train_features: pd.DataFrame,
                  preprocessor: Mapping, z0cal: np.ndarray, p0cal: np.ndarray,
                  zintcal: np.ndarray, sigma2: float):
    xfit = fit_sparse_proxy_preprocessor(train_features, preprocessor)
    cal_x = transform_sparse_proxy(cal_features, preprocessor, xfit, z0cal, p0cal)
    zbar = np.mean(zintcal, axis=1)
    s2 = np.var(zintcal, axis=1, ddof=1)
    delta = 0.5 * (zbar - z0cal)
    eta = np.log(0.5 * s2 + 0.5 * sigma2)
    delta_model = Ridge(alpha=1.0, fit_intercept=True, solver="lsqr").fit(cal_x, delta)
    eta_model = Ridge(alpha=1.0, fit_intercept=True, solver="lsqr").fit(cal_x, eta)
    return xfit, delta_model, eta_model


def evaluate_diabetes_p3() -> dict:
    selected, audit = load_selected_diabetes()
    cal = selected.loc[selected["split"] == "CAL"].copy().reset_index(drop=True)
    evaluation = selected.loc[selected["split"] == "EVAL"].copy().reset_index(drop=True)
    train_all = selected.loc[selected["split"] == "TRAIN"].copy()
    bank_results = []
    prediction_calls = 0
    fresh_prediction_calls = 0
    reused_prediction_calls = 0
    map_fit_calls = 0
    ridge_fit_calls = 0
    model_hashes = {}
    instance_csvs = {}
    prediction_cache_provenance = []
    for bank in range(3):
        bank_fit = read_json(_p3_bank_dir(bank) / "fit_measurement/summary.json")
        preprocessor = read_json(_p3_bank_dir(bank) / "preprocessor/preprocessor.json")
        bank_train = train_all.loc[pd.to_numeric(train_all["bank_id"]) == bank].copy()
        models = _p3_load_bank_models(bank)
        cal_matrix = transform_diabetes_lgb(cal, preprocessor)
        eval_matrix = transform_diabetes_lgb(evaluation, preprocessor)
        combined = pd.concat([cal_matrix, eval_matrix], ignore_index=True)
        ncal = len(cal)
        predictions = {}
        for model_id in models:
            model_hashes[f"bank{bank}/{model_id}"] = read_json(
                _p3_model_dir(bank, model_id) / "model_manifest.json")["model_sha256"]
        pred_dir = _p3_bank_dir(bank) / "prediction_cache"
        prediction_source = "fresh_inference"
        if pred_dir.exists():
            prediction_source = "reused_from_prior_unpublished_evaluation_attempt"
            if pred_dir.is_symlink() or not (pred_dir / "arrays.npz").is_file():
                raise ExperimentError(f"P3 prediction cache is not a regular saved cache: {pred_dir}")
            with np.load(pred_dir / "arrays.npz", allow_pickle=False) as archive:
                cached_ncal = int(np.asarray(archive["cal_count"]).reshape(-1)[0])
                if cached_ncal != ncal:
                    raise ExperimentError(f"P3 prediction cache CAL size mismatch: bank {bank}")
                for model_id in models:
                    key = f"p_{model_id}"
                    if key not in archive.files:
                        raise ExperimentError(f"P3 prediction cache model missing: bank {bank}/{model_id}")
                    values = np.asarray(archive[key], dtype=np.float64)
                    if (values.shape != (len(combined),) or not np.isfinite(values).all() or
                            np.any((values <= 0.0) | (values >= 1.0))):
                        raise ExperimentError(f"invalid P3 cached predictions bank {bank}/{model_id}")
                    predictions[model_id] = values.copy()
            reused_prediction_calls += len(predictions)
        else:
            for model_id, booster in models.items():
                values = np.asarray(booster.predict(combined, num_threads=1), dtype=np.float64)
                if (values.shape != (len(combined),) or not np.isfinite(values).all() or
                        np.any((values <= 0.0) | (values >= 1.0))):
                    raise ExperimentError(f"invalid P3 predictions bank {bank}/{model_id}")
                predictions[model_id] = values
                prediction_calls += 1
                fresh_prediction_calls += 1
        cal_p = {m: v[:ncal] for m, v in predictions.items()}
        eval_p = {m: v[ncal:] for m, v in predictions.items()}
        tau = float(np.quantile(cal_p["D0"], 0.80, method="linear"))
        if not (0.0 < tau < 1.0):
            raise ExperimentError(f"P3 threshold is outside (0,1) for bank {bank}")
        gamma = float(logit(tau))
        z0cal = safe_logit(cal_p["D0"]) - gamma
        z0eval = safe_logit(eval_p["D0"]) - gamma
        zintcal = np.column_stack([safe_logit(cal_p[f"INT-{i:02d}"]) - gamma
                                   for i in range(1, 7)])
        zinteval = np.column_stack([safe_logit(eval_p[f"INT-{i:02d}"]) - gamma
                                    for i in range(1, 7)])
        zauditcal = np.column_stack([safe_logit(cal_p[f"AUDIT-{i:02d}"]) - gamma
                                     for i in range(1, 12)])
        zauditeval = np.column_stack([safe_logit(eval_p[f"AUDIT-{i:02d}"]) - gamma
                                      for i in range(1, 12)])
        ical = z0cal >= 0.0
        ieval = z0eval >= 0.0
        if not np.any(ical) or not np.any(ieval):
            raise ExperimentError(f"P3 bank {bank} has empty CAL/EVAL alert queue")
        tcal_all = (zauditcal < 0.0).mean(axis=1)
        teval_all = (zauditeval < 0.0).mean(axis=1)
        tcal = tcal_all[ical]
        teval = teval_all[ieval]
        audit_eval_alert = (zauditeval[ieval] < 0.0).astype(np.int8)
        z0c = z0cal[ical]
        z0e = z0eval[ieval]
        zi_c = zintcal[ical]
        zi_e = zinteval[ieval]
        zbar_c = zi_c.mean(axis=1)
        zbar_e = zi_e.mean(axis=1)
        svar_c = np.var(zi_c, axis=1, ddof=1)
        svar_e = np.var(zi_e, axis=1, ddof=1)
        sigma2 = float(np.mean(svar_c))
        if not math.isfinite(sigma2) or sigma2 <= 0.0:
            raise ExperimentError(f"P3 CAL sigma_g^2 is nonpositive for bank {bank}")
        margin_map = adult.fit_mapping("logistic_brier_margin", z0c, tcal)
        mean_map = adult.fit_mapping("logistic_brier_margin", zbar_c, tcal)
        map_fit_calls += 2
        proxy_transform, delta_model, eta_model = _p3_proxy_fit(
            cal, bank_train, preprocessor, z0cal, cal_p["D0"], zintcal, sigma2)
        ridge_fit_calls += 2
        eval_proxy_x = transform_sparse_proxy(evaluation, preprocessor, proxy_transform,
                                              z0eval, eval_p["D0"])
        delta_hat = delta_model.predict(eval_proxy_x)
        eta_hat = eta_model.predict(eval_proxy_x)
        m_hat = z0eval + delta_hat
        v_hat = np.clip(np.exp(eta_hat), 0.1 * sigma2, 10.0 * sigma2)
        k = np.sum(zi_e < 0.0, axis=1)
        vtilde = 0.5 * svar_e + 0.5 * sigma2
        mtilde = 0.5 * z0e + 0.5 * zbar_e
        scores = {
            "margin_logistic": adult.apply_mapping(margin_map, z0e),
            "q_global_1": ndtr(-z0e / math.sqrt(sigma2)),
            "q_sqrt2": ndtr(-z0e / math.sqrt(2.0 * sigma2)),
            "int_freq": (k + 0.5) / 7.0,
            "int_freq_margin_tiebreak": (k + 0.25 + 0.5 * expit(-z0e)) / 7.0,
            "int_mean_logistic": adult.apply_mapping(mean_map, zbar_e),
            "q_V": ndtr(-z0e / np.sqrt(vtilde)),
            "q_MV_shrunk": ndtr(-mtilde / np.sqrt(vtilde)),
            "q_MV_raw": ndtr(-zbar_e / np.sqrt(vtilde)),
            "q_MV_surrogate": ndtr(-m_hat[ieval] / np.sqrt(v_hat[ieval])),
        }
        if tuple(scores) != P3_METHODS:
            raise AssertionError("P3 method list/order does not match the fixed protocol")
        metrics = {method: _risk_metrics(scores[method], teval, audit_eval_alert,
                                         scores["margin_logistic"], scores["q_sqrt2"])
                   for method in P3_METHODS}
        y_eval = evaluation["y_readmit30"].to_numpy(dtype=np.int8)
        p_eval = eval_p["D0"]
        try:
            outcome_auc = float(roc_auc_score(y_eval, p_eval))
        except ValueError:
            outcome_auc = None
        bank_result = {
            "bank_id": bank,
            "train_patients": int(len(bank_train)),
            "CAL_patients": int(len(cal)), "EVAL_patients": int(len(evaluation)),
            "tau_0_CAL_q80": tau, "gamma_0": gamma,
            "CAL_alert_groups": int(ical.sum()),
            "EVAL_alert_groups": int(ieval.sum()),
            "EVAL_alert_rate": float(ieval.mean()),
            "sigma_g_sq_CAL_alerts": sigma2,
            "INT_mean_z_CAL_alert": float(zbar_c.mean()),
            "INT_variance_mean_CAL_alert": sigma2,
            "D0_readmission_EVAL": {
                "AUC": outcome_auc,
                "Brier": float(np.mean((p_eval-y_eval)**2)),
                "positive_rate": float(y_eval.mean()),
                "alert_rate": float(ieval.mean()),
            },
            "surrogate": {
                "ridge_alpha": 1.0, "ridge_fit_calls": 2,
                "fit_rows": int(len(cal)), "targets_used": ["delta", "eta"],
                "audit_or_eval_outputs_used_for_fit": False,
                "test_INT_model_inference_calls": 0,
                "test_inference_inputs": ["feature matrix", "D0 p", "D0 z"],
            },
            "metrics": metrics,
            "model_hashes": {name: digest for name, digest in model_hashes.items()
                             if name.startswith(f"bank{bank}/")},
            "fit_measurement": bank_fit,
            "prediction_batches": len(predictions),
        }
        bank_results.append(bank_result)
        output_ids = evaluation.loc[ieval, "patient_nbr"].astype(str).map(
            lambda x: sha_text("patient:"+x)).tolist()
        instance_stream = io.StringIO(newline="")
        writer = csv.DictWriter(instance_stream,
                                fieldnames=["patient_key_sha256", "t_audit", *P3_METHODS])
        writer.writeheader()
        for i, patient_hash in enumerate(output_ids):
            row = {"patient_key_sha256": patient_hash, "t_audit": repr(float(teval[i]))}
            row.update({m: repr(float(scores[m][i])) for m in P3_METHODS})
            writer.writerow(row)
        instance_csvs[f"bank{bank}_eval_alert_scores.csv"] = instance_stream.getvalue().encode()
        if not pred_dir.exists():
            pred_arrays = {"cal_count": np.asarray([ncal]),
                           **{f"p_{key}": value for key, value in predictions.items()}}
            save_numpy_archive(pred_dir, pred_arrays)
        prediction_cache_provenance.append({
            "bank_id": bank,
            "source": prediction_source,
            "archive_sha256": sha_bytes((pred_dir / "arrays.npz").read_bytes()),
            "cal_rows": ncal,
            "evaluation_rows": len(evaluation),
            "model_output_arrays": len(predictions),
            "artifact_binding_limitation": "cache predates its report manifest; hash and model manifests are recorded now, but no per-array model-hash sidecar was saved at inference time",
        })
    aggregate = {}
    metric_keys = list(bank_results[0]["metrics"][P3_METHODS[0]].keys())
    for method in P3_METHODS:
        aggregate[method] = {}
        for key in metric_keys:
            vals = [float(row["metrics"][method][key]) for row in bank_results
                    if row["metrics"][method].get(key) is not None]
            aggregate[method][key] = {
                "mean": float(np.mean(vals)) if vals else None,
                "min": float(np.min(vals)) if vals else None,
                "max": float(np.max(vals)) if vals else None,
                "available_banks": len(vals),
            }
    report = {
        "status": "P3_THREE_BANK_EVALUATION_COMPLETE",
        "purpose": "external public nonclinical stability benchmark; finite AUDIT flip target",
        "protocol_sha256": sha_bytes(PROTOCOL_PATH.read_bytes()),
        "audit": audit,
        "counts": {"model_fits": 54,
                   "model_inference_batches_cal_plus_eval": prediction_calls + reused_prediction_calls,
                   "new_inference_batches_this_report_run": fresh_prediction_calls,
                   "reused_saved_inference_batches": reused_prediction_calls,
                   "mapping_fit_calls": map_fit_calls, "ridge_fit_calls": ridge_fit_calls,
                   "test_INT_inference_for_surrogate": 0},
        "bank_results": bank_results,
        "prediction_cache_provenance": prediction_cache_provenance,
        "aggregate_mean_min_max_across_banks": aggregate,
        "limitations": [
            "The three bank training patient pools are disjoint, but CAL/EVAL patients are shared across banks; bank range is descriptive, not an independent-site confidence interval.",
            "AUDIT bootstrap models define a finite retraining reference and are not an unknown retraining population.",
            "UCI does not provide hospital identifiers or calendar timestamps in this export; this is not held-out-hospital or temporal validation.",
            "The target is risk of alert-decision change under the declared bootstrap operation, not readmission probability; D0 readmission AUC/Brier are reported separately.",
        ],
        "resource_summary": {
            "fit_wall_seconds_sum": float(sum(x["fit_measurement"]["total_fit_wall_seconds"] for x in bank_results)),
            "fit_cpu_seconds_sum": float(sum(x["fit_measurement"]["total_fit_cpu_seconds"] for x in bank_results)),
            "fit_peak_rss_max_bytes": int(max(x["fit_measurement"]["peak_rss_bytes"] for x in bank_results)),
            "inference_batches": prediction_calls + reused_prediction_calls,
            "new_inference_batches_this_report_run": fresh_prediction_calls,
            "reused_saved_inference_batches": reused_prediction_calls,
            "native_threads": 1, "gpu_used": False,
        },
    }
    csv_stream = io.StringIO(newline="")
    cols = ["bank_id", "method", "n_alert_groups", *[f"R_{int(c*100):03d}" for c in RISK_COVERAGES],
            "AURC100", "AURC_70_100", "AUC_vs_AUDIT_flip", "Brier_finite_AUDIT",
            "paired_delta_Brier_vs_margin_logistic", "paired_delta_Brier_vs_q_sqrt2", "ECE10",
            "mean_AUDIT_flip_rate"]
    writer = csv.DictWriter(csv_stream, fieldnames=cols)
    writer.writeheader()
    for bank in bank_results:
        for method in P3_METHODS:
            writer.writerow({"bank_id": bank["bank_id"], "method": method,
                             **bank["metrics"][method]})
    agg_stream = io.StringIO(newline="")
    agg_cols = ["method"] + [f"{m}_{s}" for m in metric_keys for s in ("mean", "min", "max")]
    aw = csv.DictWriter(agg_stream, fieldnames=agg_cols)
    aw.writeheader()
    for method in P3_METHODS:
        row = {"method": method}
        for key in metric_keys:
            for stat in ("mean", "min", "max"):
                row[f"{key}_{stat}"] = aggregate[method][key][stat]
        aw.writerow(row)
    write_new_directory(OUT_ROOT / "p3_three_bank_results", {
        "p3_report.json": canonical(report),
        "p3_per_bank_metrics.csv": csv_stream.getvalue().encode(),
        "p3_mean_min_max.csv": agg_stream.getvalue().encode(),
        **instance_csvs,
    })
    print(json.dumps({"stage": "report-p3", "status": report["status"],
                      "total_inference_batches": report["counts"]["model_inference_batches_cal_plus_eval"],
                      "new_inference_batches_this_report_run": fresh_prediction_calls,
                      "reused_saved_inference_batches": reused_prediction_calls,
                      "EVAL_alert_counts": [x["EVAL_alert_groups"] for x in bank_results],
                      "aggregate_q_MV_shrunk_Brier": aggregate["q_MV_shrunk"]["Brier_finite_AUDIT"]},
                     ensure_ascii=False))
    return report


def evaluate_adult_surrogate() -> dict:
    """Fit only CAL512 ridge proxies and score saved P1 dev/locked rows."""
    prep = read_json(ADULT_DATA / "preparation_manifest.json")
    preprocessing = read_json(ADULT_DATA / "preprocessing.json")
    raw_data = (ADULT_DATA / "adult.data").read_bytes()
    raw_test = (ADULT_DATA / "adult.test").read_bytes()
    data_groups = adult._parse_adult(raw_data, "adult.data", keep_income=False)
    test_groups = adult._parse_adult(raw_test, "adult.test", keep_income=False)
    config, _ = adult.load_config()
    parts = adult._group_assignment_manifest(data_groups, test_groups, config)
    train_groups = parts["train"]
    cal_groups = parts["calibration"]
    lock = [g for g in parts["calibration_candidates"]
            if g["group_id"] not in {x["group_id"] for x in cal_groups}]
    if len(cal_groups) != 512 or len(lock) != 9141:
        raise ExperimentError("Adult P1 surrogate roles no longer match saved 512/9141 cohort design")
    train_matrix = adult._transform_features(train_groups, preprocessing)
    cal_matrix = adult._transform_features(cal_groups, preprocessing)
    dev_ids_rows = list(csv.DictReader((ADULT_P1 / "development_instance_scores.csv").open()))
    lock_ids_rows = list(csv.DictReader((ADULT_P1 / "locked_validation_instance_scores.csv").open()))
    dev_map = {x["group_id"]: x for x in dev_ids_rows} if "group_id" in (dev_ids_rows[0] if dev_ids_rows else {}) else {
        x["feature_group_sha256"]: x for x in dev_ids_rows}
    lock_map = {x["feature_group_sha256"]: x for x in lock_ids_rows}
    dev_lookup = {g["group_id"]: g for g in parts["evaluation"]}
    lock_lookup = {g["group_id"]: g for g in lock}
    if not set(dev_map).issubset(dev_lookup) or not set(lock_map).issubset(lock_lookup):
        raise ExperimentError("Adult P1 saved group identity does not align with public feature source")
    cal_p0_raw, cal_p0_sha = adult._read_fixed(Path("runs/adult_pilot/predictions/D0-calibration/calibration.json"))
    cal_p0_arr = np.asarray(json.loads(cal_p0_raw), dtype=np.float64)
    z0cal = safe_logit(cal_p0_arr)
    p_int = []
    int_hashes = {}
    for j in range(1, 7):
        rel = Path(f"runs/adult_pilot/predictions/INT-{j:02d}-calibration/calibration.json")
        raw_values, digest = adult._read_fixed(rel)
        p_int.append(np.asarray(json.loads(raw_values), dtype=np.float64))
        int_hashes[f"INT-{j:02d}"] = digest
    zintcal = np.column_stack([safe_logit(p) for p in p_int])
    ical = z0cal >= 0.0
    sigma2 = float(np.mean(np.var(zintcal[ical], axis=1, ddof=1)))
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ExperimentError("Adult CAL512 saved bank has invalid sigma_g_sq")
    adult_numeric_idx = [adult.FEATURES.index(c) for c in adult.NUMERIC]
    adult_categorical_idx = [adult.FEATURES.index(c) for c in adult.CATEGORICAL]
    adult_num_scaler = StandardScaler().fit(train_matrix[:, adult_numeric_idx])
    adult_train_cats = np.column_stack([train_matrix[:, j].astype(int)
                                        for j in adult_categorical_idx])
    adult_categories = [np.arange(len(preprocessing["categorical_vocabularies"][c]))
                        for c in adult.CATEGORICAL]
    adult_encoder = OneHotEncoder(categories=adult_categories, handle_unknown="ignore",
                                  sparse_output=True, dtype=np.float64)
    adult_train_ohe = adult_encoder.fit_transform(adult_train_cats)
    adult_cat_scaler = StandardScaler(with_mean=False).fit(adult_train_ohe)
    adult_sparse_fit = {"numeric_scaler": adult_num_scaler,
                        "encoder": adult_encoder, "cat_scaler": adult_cat_scaler}
    # The saved LightGBM feature matrix stores numeric values and categorical integer codes.
    # Convert those codes to feature names using the saved TRAIN-only vocabularies.
    def adult_proxy_x(groups: list[dict], matrix: np.ndarray, z: np.ndarray):
        n = len(groups)
        numeric = matrix[:, adult_numeric_idx]
        numeric_scaled = adult_sparse_fit["numeric_scaler"].transform(numeric)
        categories = np.empty((n, len(adult.CATEGORICAL)), dtype=object)
        for j, col in enumerate(adult.CATEGORICAL):
            categories[:, j] = matrix[:, adult_categorical_idx[j]].astype(int)
        cat = adult_sparse_fit["cat_scaler"].transform(adult_sparse_fit["encoder"].transform(categories))
        p = expit(z)
        return hstack((csr_matrix(numeric_scaled), cat,
                       csr_matrix(np.column_stack((z, np.abs(z), p*(1-p))))), format="csr")
    cal_x = adult_proxy_x(cal_groups, cal_matrix, z0cal)
    zbar_cal = zintcal.mean(axis=1)
    s2_cal = np.var(zintcal, axis=1, ddof=1)
    delta = 0.5*(zbar_cal-z0cal)
    eta = np.log(0.5*s2_cal+0.5*sigma2)
    delta_model = Ridge(alpha=1.0, fit_intercept=True, solver="lsqr").fit(cal_x, delta)
    eta_model = Ridge(alpha=1.0, fit_intercept=True, solver="lsqr").fit(cal_x, eta)
    p1_report = read_json(ADULT_P1 / "p1_report.json")
    outputs = {}
    for cohort, rows, lookup in (("development", dev_ids_rows, dev_lookup),
                                 ("locked_validation", lock_ids_rows, lock_lookup)):
        ids = [row["feature_group_sha256"] for row in rows]
        groups = [lookup[key] for key in ids]
        matrix = adult._transform_features(groups, preprocessing)
        z0 = np.asarray([float(row["z0"]) for row in rows])
        x = adult_proxy_x(groups, matrix, z0)
        dhat = delta_model.predict(x)
        ehat = eta_model.predict(x)
        vhat = np.clip(np.exp(ehat), 0.1*sigma2, 10.0*sigma2)
        qsur = ndtr(-(z0+dhat)/np.sqrt(vhat))
        qmv = np.asarray([float(row["q_MV_shrunk"]) for row in rows])
        margin = np.asarray([float(row["margin_logistic"]) for row in rows])
        t = np.asarray([float(row["audit_flip_rate"]) for row in rows])
        metrics = {
            "q_MV_surrogate": {"R_070": compute_tie_aware_risk(qsur,t,.70),
                               "R_085": compute_tie_aware_risk(qsur,t,.85),
                               "AURC100": float(np.mean([compute_tie_aware_risk(qsur,t,k/100)
                                                         for k in range(1,101)])),
                               "paired_delta_Brier_vs_margin_logistic": float(np.mean(
                                   qsur*qsur-margin*margin-2*(qsur-margin)*t)),
                               "paired_delta_Brier_vs_q_MV_shrunk": float(np.mean(
                                   qsur*qsur-qmv*qmv-2*(qsur-qmv)*t))},
            "q_MV_shrunk": {"R_070": compute_tie_aware_risk(qmv,t,.70),
                            "R_085": compute_tie_aware_risk(qmv,t,.85),
                            "AURC100": float(np.mean([compute_tie_aware_risk(qmv,t,k/100)
                                                      for k in range(1,101)])),
                            "paired_delta_Brier_vs_margin_logistic": float(np.mean(
                                qmv*qmv-margin*margin-2*(qmv-margin)*t))},
            "margin_logistic": {"R_070": compute_tie_aware_risk(margin,t,.70),
                               "R_085": compute_tie_aware_risk(margin,t,.85),
                               "AURC100": float(np.mean([compute_tie_aware_risk(margin,t,k/100)
                                                         for k in range(1,101)])),
                               "paired_delta_Brier_vs_margin_logistic": 0.0},
        }
        retention = {}
        for endpoint in ("R_070", "AURC100", "paired_delta_Brier_vs_margin_logistic"):
            teacher_gain = (metrics["margin_logistic"][endpoint] - metrics["q_MV_shrunk"][endpoint]
                            if endpoint != "paired_delta_Brier_vs_margin_logistic" else
                            -metrics["q_MV_shrunk"][endpoint])
            surrogate_gain = (metrics["margin_logistic"][endpoint] - metrics["q_MV_surrogate"][endpoint]
                              if endpoint != "paired_delta_Brier_vs_margin_logistic" else
                              -metrics["q_MV_surrogate"][endpoint])
            retention[endpoint] = {"teacher_gain_vs_margin": float(teacher_gain),
                                   "surrogate_gain_vs_margin": float(surrogate_gain),
                                   "retained_fraction": (float(surrogate_gain/teacher_gain)
                                                         if teacher_gain > 0 else None)}
        outputs[cohort] = {"n_alert_groups": len(rows), "metrics": metrics,
                           "gain_retention": retention,
                           "test_INT_model_inference_calls": 0}
        instance_stream = io.StringIO(newline="")
        w = csv.DictWriter(instance_stream, fieldnames=["feature_group_sha256", "t_audit",
                                                       "q_MV_shrunk", "q_MV_surrogate"])
        w.writeheader()
        for i, key in enumerate(ids):
            w.writerow({"feature_group_sha256":key, "t_audit":repr(float(t[i])),
                        "q_MV_shrunk":repr(float(qmv[i])), "q_MV_surrogate":repr(float(qsur[i]))})
        outputs[cohort]["instance_csv"] = instance_stream.getvalue().encode()
    report = {
        "status": "ADULT_P1_ZERO_INT_SURROGATE_COMPLETE",
        "sigma_g_sq_from_saved_CAL512_alerts": sigma2,
        "CAL512_alert_count": int(ical.sum()),
        "ridge_fit_calls": 2,
        "ridge_alpha": 1.0,
        "training_rows": 512,
        "test_INT_model_inference_calls": 0,
        "test_INT_outputs_reused_only_for_teacher_comparison": True,
        "calibration_D0_cache_sha256": cal_p0_sha,
        "calibration_INT_cache_sha256": int_hashes,
        "P1_report_sha256": sha_bytes((ADULT_P1/"p1_report.json").read_bytes()),
        "cohorts": {k: {kk:vv for kk,vv in v.items() if kk != "instance_csv"}
                    for k,v in outputs.items()},
        "limitations": ["Uses the P1 finite 11-AUDIT target and same Adult source/model bank.",
                        "Surrogate predictions use D0 and features only at evaluation; six-INT teacher values are used only to quantify retained gain.",
                        "This is a post-P1 proxy fit, not a new independent validation."],
    }
    files = {"adult_surrogate_report.json": canonical(report),
             "adult_development_surrogate_scores.csv": outputs["development"]["instance_csv"],
             "adult_locked_surrogate_scores.csv": outputs["locked_validation"]["instance_csv"]}
    write_new_directory(OUT_ROOT/"adult_p1_zero_int_surrogate", files)
    print(json.dumps({"stage":"adult-surrogate", "status":report["status"],
                      "sigma_g_sq":sigma2,
                      "locked_retention":outputs["locked_validation"]["gain_retention"]},ensure_ascii=False))
    return report


def _adult_groups_for_p2():
    data_raw = (ADULT_DATA / "adult.data").read_bytes()
    test_raw = (ADULT_DATA / "adult.test").read_bytes()
    data_groups = adult._parse_adult(data_raw,"adult.data",keep_income=True)
    test_groups = adult._parse_adult(test_raw,"adult.test",keep_income=False)
    config,_=adult.load_config()
    parts=adult._group_assignment_manifest(data_groups,test_groups,config)
    return data_groups,test_groups,parts


def _p2_bank_pools(train_groups: list[dict]) -> list[tuple[list[dict],list[dict]]]:
    chosen=sorted(train_groups,key=lambda g:(sha_text("P2_bank_pool_20261110_"+g["group_id"]),g["group_id"]))[:12000]
    if len(chosen)!=12000: raise ExperimentError("Adult TRAIN has fewer than 12000 unique feature groups")
    banks=[]
    for bank in range(3):
        block=chosen[bank*4000:(bank+1)*4000]
        split=sorted(block,key=lambda g:(sha_text(f"P2_AB_split_20261110_{bank}_"+g["group_id"]),g["group_id"]))
        a,b=split[:2000],split[2000:]
        if len(a)!=2000 or len(b)!=2000 or ({x['group_id'] for x in a}&{x['group_id'] for x in b}):
            raise ExperimentError("P2 A/B split invalid")
        banks.append((a,b))
    ids=[g["group_id"] for a,b in banks for g in a+b]
    if len(set(ids))!=12000: raise ExperimentError("P2 bank pools must be disjoint")
    return banks


def _adult_lr_transform_fit(a: list[dict], *, numeric_feature_indices=None):
    prep=adult._fit_preprocessor(a)
    mat=adult._transform_features(a,prep)
    # A frozenset has process-dependent iteration order. New fits use canonical
    # feature order; legacy coefficient files require their persisted order.
    nidx=([i for i,c in enumerate(adult.FEATURES) if c in adult.NUMERIC]
          if numeric_feature_indices is None else list(numeric_feature_indices))
    if set(nidx)!={i for i,c in enumerate(adult.FEATURES) if c in adult.NUMERIC}:
        raise ExperimentError("numeric feature indices do not identify the six Adult numeric features")
    cidx=[adult.FEATURES.index(c) for c in adult.CATEGORICAL]
    num_scaler=StandardScaler().fit(mat[:,nidx])
    cats=np.column_stack([mat[:,j].astype(int) for j in cidx])
    categories=[np.arange(len(prep["categorical_vocabularies"][c])) for c in adult.CATEGORICAL]
    encoder=OneHotEncoder(categories=categories,handle_unknown="ignore",sparse_output=False,dtype=np.float64)
    cats_enc=encoder.fit_transform(cats)
    cat_scaler=StandardScaler().fit(cats_enc)
    return {"prep":prep,"nidx":nidx,"cidx":cidx,"num_scaler":num_scaler,
            "encoder":encoder,"cat_scaler":cat_scaler}


def _adult_lr_features(groups: list[dict], fit: Mapping) -> np.ndarray:
    mat=adult._transform_features(groups,fit["prep"])
    num=fit["num_scaler"].transform(mat[:,fit["nidx"]])
    cats=np.column_stack([mat[:,j].astype(int) for j in fit["cidx"]])
    cats=fit["cat_scaler"].transform(fit["encoder"].transform(cats))
    return np.column_stack((np.ones(len(mat)),num,cats))


def _group_soft_labels(groups: list[dict]) -> np.ndarray:
    return np.asarray([float(np.mean(g["labels"])) for g in groups],dtype=np.float64)


def _fit_regularized_logistic(x: np.ndarray,y: np.ndarray,weights: np.ndarray|None=None):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or w.shape!=y.shape or np.any(w<0) or w.sum()<=0:
        raise ExperimentError("invalid P2 regularized Logistic training arrays")
    penalty=np.ones(x.shape[1]); penalty[0]=0.0
    def objective(beta):
        eta=x@beta
        p=expit(eta)
        loss=np.sum(w*(np.logaddexp(0.0,eta)-y*eta))+0.5*np.sum(penalty*beta*beta)
        grad=x.T@(w*(p-y))+penalty*beta
        return float(loss),grad
    result=minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",
                    options={"maxiter":500,"ftol":1e-11,"gtol":1e-7,"maxls":30})
    if not result.success or not np.isfinite(result.x).all():
        raise ExperimentError(f"P2 Logistic fit failed: {result.message}")
    return np.asarray(result.x,dtype=np.float64),{"n_iter":int(result.nit),"objective":float(result.fun),"message":str(result.message)}


def _p2_fit_path(bank:int,op:str,model:str)->Path:
    return RUN_ROOT/"adult_p2"/f"bank-{bank}"/"models"/op/model


def _p2_write_model(path:Path,beta:np.ndarray,meta:dict):
    if path.exists():
        existing=read_json(path/"model.json")
        if existing.get("beta_sha256")!=sha_bytes(np.asarray(beta,dtype="<f8").tobytes()):
            raise ExperimentError(f"P2 model cache mismatch: {path}")
        return
    value={**meta,"beta":beta.tolist(),"beta_sha256":sha_bytes(np.asarray(beta,dtype="<f8").tobytes())}
    write_new_directory(path,{"model.json":canonical(value)})


def fit_p2_bank(bank:int)->dict:
    data_groups,test_groups,parts=_adult_groups_for_p2()
    pools=_p2_bank_pools(parts["train"])
    a,b=pools[bank]
    fit=_adult_lr_transform_fit(a)
    x_a=_adult_lr_features(a,fit); y_a=_group_soft_labels(a)
    x_b=_adult_lr_features(b,fit); y_b=_group_soft_labels(b)
    start_wall,start_cpu=time.monotonic(),time.process_time()
    models={}; costs=[]
    d0_path=_p2_fit_path(bank,"shared","D0")
    d0_start=time.monotonic(); d0_cpu=time.process_time()
    beta_d0,info=_fit_regularized_logistic(x_a,y_a)
    d0_meta={"model_id":"D0","role":"CURRENT_SINGLE","bank_id":bank,
             "train_pool":"A","n_groups":len(a),"lambda":1.0,**info}
    _p2_write_model(d0_path,beta_d0,d0_meta)
    models["D0"]=(beta_d0,d0_meta)
    costs.append({"model":"D0","wall_seconds":time.monotonic()-d0_start,
                  "cpu_seconds":time.process_time()-d0_cpu})
    subsets={"alpha_0":(x_b,y_b,np.ones(len(b)),"B"),
             "alpha_0_5":(None,None,None,"A_half+B_half"),
             "bootstrap":(x_a,y_a,None,"A_multinomial")}
    half_a=sorted(range(len(a)),key=lambda i:(sha_text(f"P2_alpha_half_A_{bank}_{a[i]['group_id']}"),a[i]["group_id"]))[:1000]
    half_b=sorted(range(len(b)),key=lambda i:(sha_text(f"P2_alpha_half_B_{bank}_{b[i]['group_id']}"),b[i]["group_id"]))[:1000]
    x_half=np.vstack((x_a[half_a],x_b[half_b])); y_half=np.r_[y_a[half_a],y_b[half_b]]
    subsets["alpha_0_5"]=(x_half,y_half,np.ones(2000),"A_half+B_half")
    for op_code,op in enumerate(("alpha_0","alpha_0_5","bootstrap"),start=1):
        x,y,weights,source=subsets[op]
        for role,number in (("INT",6),("AUDIT",11)):
            for j in range(1,number+1):
                role_code=100 if role=="INT" else 200
                seed=2026111100+10000*bank+1000*op_code+role_code+j
                mult=None
                if op=="bootstrap":
                    rng=np.random.default_rng(seed+50000)
                    mult=np.bincount(rng.integers(0,len(a),size=len(a)),minlength=len(a)).astype(np.int64)
                    weights=mult.astype(float)
                model_id=f"{role}-{j:02d}"
                path=_p2_fit_path(bank,op,model_id)
                fstart=time.monotonic(); fcputime=time.process_time()
                beta,fit_info=_fit_regularized_logistic(x,y,weights)
                meta={"model_id":model_id,"role":role,"bank_id":bank,"operation":op,
                      "train_source":source,"fit_groups":int(len(y)),"seed":seed,
                      "bootstrap_seed":seed+50000 if mult is not None else None,
                      "bootstrap_multiplicity_sha256":sha_bytes(mult.astype("<i8").tobytes()) if mult is not None else None,
                      "lambda":1.0,**fit_info}
                _p2_write_model(path,beta,meta)
                if mult is not None:
                    write_new_directory(path/"bootstrap_weights",{
                        "multiplicities.json":canonical({"multiplicities":mult.tolist(),
                                                          "group_ids_sha256":sha_bytes("\n".join(g["group_id"] for g in a).encode())})})
                models[f"{op}/{model_id}"]=(beta,meta)
                costs.append({"model":f"{op}/{model_id}","wall_seconds":time.monotonic()-fstart,
                              "cpu_seconds":time.process_time()-fcputime})
    # Persist A/B identities and the bank preprocessor before any calibration/evaluation use.
    pool_file=RUN_ROOT/"adult_p2"/f"bank-{bank}"/"pool_manifest"
    ids_payload={"A_group_ids": [g["group_id"] for g in a],
                 "B_group_ids": [g["group_id"] for g in b],
                 "selected_CAL512_group_ids": [g["group_id"] for g in parts["calibration"]],
                 "EVAL_group_ids": [g["group_id"] for g in parts["evaluation"]]}
    if not pool_file.exists():
        write_new_directory(pool_file,{"pool_manifest.json":canonical(ids_payload),
                                       "preprocessor.json":canonical(fit["prep"]),
                                       "fit_design.json":canonical({"numeric_feature_indices":fit["nidx"],
                                                                     "categorical_feature_indices":fit["cidx"],
                                                                     "numeric_mean":fit["num_scaler"].mean_.tolist(),
                                                                     "numeric_scale":fit["num_scaler"].scale_.tolist(),
                                                                     "category_mean":fit["cat_scaler"].mean_.tolist(),
                                                                     "category_scale":fit["cat_scaler"].scale_.tolist(),
                                                                     "categorical_vocabularies":fit["prep"]["categorical_vocabularies"]})})
    summary={"bank_id":bank,"status":"P2_BANK_FITS_COMPLETE",
             "models_fit_this_command":len(costs),"model_fit_costs":costs,
             "total_fit_wall_seconds":float(time.monotonic()-start_wall),
             "total_fit_cpu_seconds":float(time.process_time()-start_cpu),
             "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
             "A_groups":len(a),"B_groups":len(b),"A_sha256":sha_bytes("\n".join(g["group_id"] for g in a).encode()),
             "B_sha256":sha_bytes("\n".join(g["group_id"] for g in b).encode()),
             "fit_count":52,"mapping_fit_count_expected":6}
    write_new_directory(RUN_ROOT/"adult_p2"/f"bank-{bank}"/"fit_measurement",{"summary.json":canonical(summary)})
    print(json.dumps({"stage":f"fit-p2-bank-{bank}",**summary},ensure_ascii=False))
    return summary


def _p2_bank0_ok():
    summary=read_json(RUN_ROOT/"adult_p2/bank-0/fit_measurement/summary.json")
    if summary.get("status")!="P2_BANK_FITS_COMPLETE": raise ExperimentError("P2 Bank0 is incomplete")
    if float(summary["total_fit_wall_seconds"])>120:
        raise ExperimentError("P2 Bank0 exceeded predeclared 120s controllability threshold; remaining banks not started")
    return summary


def fit_p2_rest():
    out=[_p2_bank0_ok()]
    for bank in (1,2): out.append(fit_p2_bank(bank))
    return out


def _p2_load_beta(bank: int, operation: str, model_id: str) -> tuple[np.ndarray, dict]:
    path = _p2_fit_path(bank, operation, model_id)
    if path.is_symlink() or not (path / "model.json").is_file():
        raise ExperimentError(f"P2 model cache miss; stop without fitting: bank={bank}/{operation}/{model_id}")
    value = read_json(path / "model.json")
    beta = np.asarray(value["beta"], dtype=np.float64)
    if (beta.ndim != 1 or not np.isfinite(beta).all() or
            sha_bytes(beta.astype("<f8").tobytes()) != value.get("beta_sha256")):
        raise ExperimentError(f"P2 coefficient cache hash mismatch: {bank}/{operation}/{model_id}")
    return beta, value


def _kde_log_derivatives(z_eval: np.ndarray, z_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Gaussian KDE log-density derivatives with a fixed Silverman robust bandwidth."""
    z_eval = np.asarray(z_eval, dtype=np.float64)
    z_ref = np.asarray(z_ref, dtype=np.float64)
    if z_ref.ndim != 1 or z_ref.size < 2 or not np.isfinite(z_ref).all():
        raise ExperimentError("IF KDE requires at least two finite CAL margins")
    sd = float(np.std(z_ref, ddof=1))
    q25, q75 = np.quantile(z_ref, [0.25, 0.75])
    h = 1.06 * min(sd, float((q75-q25)/1.349)) * z_ref.size ** (-0.2)
    if not math.isfinite(h) or h <= 0.0:
        raise ExperimentError("IF KDE bandwidth is zero/nonfinite")
    lp = np.empty(z_eval.size, dtype=np.float64)
    lpp = np.empty(z_eval.size, dtype=np.float64)
    inv_h2 = 1.0/(h*h)
    for start in range(0, z_eval.size, 512):
        z = z_eval[start:start+512]
        delta = z[:, None] - z_ref[None, :]
        logw = -0.5 * np.square(delta/h)
        logw -= logw.max(axis=1, keepdims=True)
        weights = np.exp(logw)
        weights /= weights.sum(axis=1, keepdims=True)
        # d log m / dz = E[-(z-z_j)/h^2]
        mean_score = np.sum(weights * (-delta*inv_h2), axis=1)
        second_score = np.sum(weights * np.square(delta*inv_h2), axis=1)
        lp[start:start+len(z)] = mean_score
        lpp[start:start+len(z)] = -inv_h2 + second_score - mean_score*mean_score
    return lp, lpp, float(h)


def _quadratic_rows(a: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk,ik->i", a, matrix, a, optimize=True)


def _corr_summary(a: np.ndarray, b: np.ndarray) -> dict:
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    ok=np.isfinite(a)&np.isfinite(b)
    if int(ok.sum())<3 or np.std(a[ok])==0 or np.std(b[ok])==0:
        return {"n":int(ok.sum()),"pearson_r":None,"spearman_r":None}
    return {"n":int(ok.sum()),"pearson_r":float(pearsonr(a[ok],b[ok]).statistic),
            "spearman_r":float(spearmanr(a[ok],b[ok]).statistic)}


def _p2_if_design(bank: int, a_groups: list[dict], b_groups: list[dict],
                  fit: Mapping, beta0: np.ndarray, x_eval: np.ndarray,
                  op: str) -> dict:
    """Estimate V0, VR, C using centered Logistic score influences and design weights."""
    x_a=_adult_lr_features(a_groups,fit)
    y_a=_group_soft_labels(a_groups)
    x_b=_adult_lr_features(b_groups,fit)
    y_b=_group_soft_labels(b_groups)
    p_a=expit(x_a@beta0); p_b=expit(x_b@beta0)
    u_a=(p_a-y_a)[:,None]*x_a
    u_b=(p_b-y_b)[:,None]*x_b
    u_a=u_a-u_a.mean(axis=0,keepdims=True)
    u_b=u_b-u_b.mean(axis=0,keepdims=True)
    hessian=x_a.T@((p_a*(1-p_a))[:,None]*x_a)
    penalty=np.ones(x_a.shape[1]); penalty[0]=0.0
    hessian+=np.diag(penalty)
    try:
        h_inv=np.linalg.inv(hessian)
    except np.linalg.LinAlgError as exc:
        raise ExperimentError("P2 Logistic Hessian is singular") from exc
    # H is the Hessian of the *total* penalized loss, so centered score
    # covariance is a sum.  Dividing J by n^2 here would apply an extra
    # mean-loss normalization and shrink V0/VR/C by n^2.
    j_a=u_a.T@u_a
    j_b=u_b.T@u_b
    half_a=sorted(range(len(a_groups)),key=lambda i:(sha_text(f"P2_alpha_half_A_{bank}_{a_groups[i]['group_id']}"),a_groups[i]["group_id"]))[:1000]
    half_b=sorted(range(len(b_groups)),key=lambda i:(sha_text(f"P2_alpha_half_B_{bank}_{b_groups[i]['group_id']}"),b_groups[i]["group_id"]))[:1000]
    if op=="alpha_0":
        j_r,j_c=j_b,np.zeros_like(j_a)
    elif op=="alpha_0_5":
        j_as=u_a[half_a].T@u_a[half_a]
        j_bs=u_b[half_b].T@u_b[half_b]
        j_r,j_c=j_as+j_bs,j_as
    elif op=="bootstrap":
        # Unconditional first-order bootstrap endpoint: data-sampling variance plus
        # conditional bootstrap variance, with C/V0=1 and VR/V0=2.
        j_r,j_c=2.0*j_a,j_a
    else:
        raise ValueError(op)
    a_eval=x_eval@h_inv
    v0=_quadratic_rows(a_eval,j_a)
    vr=_quadratic_rows(a_eval,j_r)
    cov=_quadratic_rows(a_eval,j_c)
    # Signed first-order IF contributions; the sign cancels in all covariance terms.
    return {"V0":v0,"VR":vr,"C":cov,
            "C_over_V0":np.divide(cov,v0,out=np.full_like(cov,np.nan),where=v0>0),
            "VR_over_V0":np.divide(vr,v0,out=np.full_like(vr,np.nan),where=v0>0),
            "hessian_condition_number":float(np.linalg.cond(hessian)),
            "score_rows_A":int(len(a_groups)),"score_rows_B":int(len(b_groups)),
            "bootstrap_asymptotic_endpoint":op=="bootstrap"}


def _summary_vec(x: np.ndarray) -> dict:
    x=np.asarray(x,dtype=np.float64); x=x[np.isfinite(x)]
    if not x.size:return {"n":0,"mean":None,"median":None,"min":None,"max":None}
    return {"n":int(x.size),"mean":float(np.mean(x)),"median":float(np.median(x)),
            "min":float(np.min(x)),"max":float(np.max(x))}


def evaluate_p2() -> dict:
    """Historical fixed-subset report; corrected random operators are in run_followup.py."""
    eval_wall_start=time.monotonic()
    eval_cpu_start=time.process_time()
    data_groups,test_groups,parts=_adult_groups_for_p2()
    pools=_p2_bank_pools(parts["train"])
    cal_groups=parts["calibration"]
    eval_groups=parts["evaluation"]
    bank_records=[]; inference_calls=0; reused_inference_calls=0
    mapping_fits=0; raw_tweedie_calls=0
    prediction_payloads={}
    metric_keys=None
    for bank,(a_groups,b_groups) in enumerate(pools):
        design=read_json(RUN_ROOT/"adult_p2"/f"bank-{bank}"/"pool_manifest/fit_design.json")
        fit=_adult_lr_transform_fit(a_groups,numeric_feature_indices=design["numeric_feature_indices"])
        x_a=_adult_lr_features(a_groups,fit); y_a=_group_soft_labels(a_groups)
        x_b=_adult_lr_features(b_groups,fit); y_b=_group_soft_labels(b_groups)
        x_cal=_adult_lr_features(cal_groups,fit)
        x_eval=_adult_lr_features(eval_groups,fit)
        x_all=np.vstack((x_cal,x_eval)); ncal=len(cal_groups)
        beta0,meta0=_p2_load_beta(bank,"shared","D0")
        logit_cache_dir=RUN_ROOT/"adult_p2"/f"bank-{bank}"/"logit_prediction_cache"
        ops=("alpha_0","alpha_0_5","bootstrap")
        model_keys=["shared/D0"]+[
            f"{op}/{role}-{j:02d}" for op in ops
            for role,nmodels in (("INT",6),("AUDIT",11))
            for j in range(1,nmodels+1)]
        expected_model_hashes=[]
        for key in model_keys:
            op_key,model_id=key.split("/",1)
            _,model_meta=_p2_load_beta(bank,op_key,model_id)
            expected_model_hashes.append(model_meta["beta_sha256"])
        if logit_cache_dir.exists():
            if logit_cache_dir.is_symlink() or not (logit_cache_dir/"arrays.npz").is_file():
                raise ExperimentError(f"P2 logit prediction cache is not regular: bank={bank}")
            with np.load(logit_cache_dir/"arrays.npz",allow_pickle=False) as archive:
                if int(np.asarray(archive["cal_count"]).reshape(-1)[0])!=ncal:
                    raise ExperimentError(f"P2 cached logit CAL size mismatch: bank={bank}")
                saved_hashes=[str(x) for x in archive["model_beta_sha256"].tolist()]
                if saved_hashes!=expected_model_hashes:
                    raise ExperimentError(f"P2 cached logit/model digest mismatch: bank={bank}")
                pred0=np.asarray(archive["z_shared_D0"],dtype=np.float64)
                op_refs={op:{"INT":np.asarray(archive[f"z_{op}_INT"],dtype=np.float64),
                             "AUDIT":np.asarray(archive[f"z_{op}_AUDIT"],dtype=np.float64)}
                         for op in ops}
            if not np.allclose(pred0,x_all@beta0,rtol=0,atol=1e-8):
                raise ExperimentError("historical P2 prediction cache has a numeric-column-order error; use run_followup.py p2, which restores saved feature order and recomputes predictions")
            if pred0.shape!=(len(x_all),) or any(
                    values[role].shape!=(len(x_all),nmodels)
                    for values in op_refs.values()
                    for role,nmodels in (("INT",6),("AUDIT",11))):
                raise ExperimentError(f"P2 logit prediction cache shape mismatch: bank={bank}")
            reused_inference_calls+=len(model_keys)
        else:
            pred0=x_all@beta0; inference_calls+=1
            op_refs={}
            cache_payload={"cal_count":np.asarray([ncal]),
                           "model_beta_sha256":np.asarray(expected_model_hashes,dtype="U64"),
                           "z_shared_D0":pred0}
            for op in ops:
                refs={"INT":[],"AUDIT":[]}
                for role,nmodels in (("INT",6),("AUDIT",11)):
                    for j in range(1,nmodels+1):
                        beta,_=_p2_load_beta(bank,op,f"{role}-{j:02d}")
                        margin=x_all@beta; inference_calls+=1
                        refs[role].append(margin)
                op_refs[op]={role:np.column_stack(values) for role,values in refs.items()}
                cache_payload[f"z_{op}_INT"]=op_refs[op]["INT"]
                cache_payload[f"z_{op}_AUDIT"]=op_refs[op]["AUDIT"]
            save_numpy_archive(logit_cache_dir,cache_payload)
        z0cal,z0eval=pred0[:ncal],pred0[ncal:]
        cal_alert=z0cal>=0; eval_alert=z0eval>=0
        if not np.any(cal_alert) or not np.any(eval_alert):
            raise ExperimentError(f"P2 bank {bank} has an empty CAL/EVAL D0 alert queue")
        prediction_payloads[f"bank{bank}"]={"z0cal":z0cal,"z0eval":z0eval}
        for op,refs in op_refs.items():
            ints_cal=refs["INT"][:ncal][cal_alert]-0.0
            ints_eval=refs["INT"][ncal:][eval_alert]-0.0
            audits_cal=refs["AUDIT"][:ncal][cal_alert]-0.0
            audits_eval=refs["AUDIT"][ncal:][eval_alert]-0.0
            z0c=z0cal[cal_alert]; z0e=z0eval[eval_alert]
            tcal=(audits_cal<0).mean(axis=1)
            teval=(audits_eval<0).mean(axis=1)
            yaudit=(audits_eval<0).astype(np.int8)
            zbarc=ints_cal.mean(axis=1); zbare=ints_eval.mean(axis=1)
            s2c=np.var(ints_cal,axis=1,ddof=1); s2e=np.var(ints_eval,axis=1,ddof=1)
            sigma2=float(np.mean(s2c))
            if not np.isfinite(sigma2) or sigma2<=0:
                raise ExperimentError(f"P2 bank {bank}/{op} CAL sigma_g_sq is invalid")
            margin_map=adult.fit_mapping("logistic_brier_margin",z0c,tcal)
            mean_map=adult.fit_mapping("logistic_brier_margin",zbarc,tcal)
            mapping_fits+=2
            vtilde=.5*s2e+.5*sigma2
            mtilde=.5*z0e+.5*zbare
            k=(ints_eval<0).sum(axis=1)
            q_sqrt2=ndtr(-z0e/math.sqrt(2*sigma2))
            scores={
                "margin_logistic":adult.apply_mapping(margin_map,z0e),
                "q_global_1":ndtr(-z0e/math.sqrt(sigma2)),
                "q_sqrt2":q_sqrt2,
                "int_freq":(k+.5)/7.,
                "int_freq_margin_tiebreak":(k+.25+.5*expit(-z0e))/7.,
                "int_mean_logistic":adult.apply_mapping(mean_map,zbare),
                "q_V":ndtr(-z0e/np.sqrt(vtilde)),
                "q_MV_shrunk":ndtr(-mtilde/np.sqrt(vtilde)),
                "q_MV_raw":ndtr(-zbare/np.sqrt(vtilde)),
            }
            tweedie,twdiag=adult.flip.tweedie_flip_probability(
                z0e,z0c,sigma2,return_diagnostics=True)
            scores["raw_tweedie"]=np.asarray(tweedie,dtype=np.float64)
            raw_tweedie_calls+=1

            # Influence function uses the D0 A-pool Hessian/residuals.  Evaluation
            # calculations use only x and the already fitted D0 coefficients.
            eval_design=x_eval[eval_alert]
            if_result=_p2_if_design(bank,a_groups,b_groups,fit,beta0,eval_design,op)
            v0,vr,cov=if_result["V0"],if_result["VR"],if_result["C"]
            lprime,lsecond,kde_bandwidth=_kde_log_derivatives(z0e,z0c)
            mean_if=z0e+(v0-cov)*lprime
            var_if=v0+vr-2*cov+(v0-cov)**2*lsecond
            valid=np.isfinite(mean_if)&np.isfinite(var_if)&(v0>0)&(var_if>=0)
            q_if=np.full_like(z0e,np.nan)
            pos=valid&(var_if>0)
            q_if[pos]=ndtr(-mean_if[pos]/np.sqrt(var_if[pos]))
            zero=valid&(var_if==0)
            q_if[zero]=(mean_if[zero]<0).astype(float)
            scores["q_IF_analytic"]=q_if
            if set(scores)!=set(P2_METHODS):
                raise AssertionError("P2 predeclared method set mismatch")
            metrics={name:_risk_metrics(value,teval,yaudit,scores["margin_logistic"],q_sqrt2)
                     for name,value in scores.items() if name!="q_IF_analytic"}
            if valid.all():
                metrics["q_IF_analytic"]=_risk_metrics(q_if,teval,yaudit,
                                                        scores["margin_logistic"],q_sqrt2)
                if_status="ESTIMABLE_FULL_ALERT_SUPPORT"
            elif valid.any():
                common={name:_risk_metrics(scores[name][valid],teval[valid],yaudit[valid],
                                            scores["margin_logistic"][valid],q_sqrt2[valid])
                        for name in ("q_IF_analytic","q_MV_shrunk")}
                metrics["q_IF_analytic"]={"status":"PARTIAL_IF_SUPPORT",
                                          "valid_alert_groups":int(valid.sum()),
                                          "total_alert_groups":int(len(valid)),
                                          "common_support_metrics":common}
                if_status="PARTIAL_IF_SUPPORT"
            else:
                metrics["q_IF_analytic"]={"status":"NOT_ESTIMABLE",
                                          "valid_alert_groups":0,
                                          "total_alert_groups":int(len(valid)),
                                          "reason":"nonpositive V0 or conditional variance W on all alert groups"}
                if_status="NOT_ESTIMABLE"
            audit_variance=np.var(audits_eval,axis=1,ddof=1)
            v0_audit_corr=_corr_summary(v0,audit_variance)
            diag={"bank_id":bank,"operation":op,"sigma_g_sq_CAL":sigma2,
                  "CAL_alert_groups":int(cal_alert.sum()),"EVAL_alert_groups":int(eval_alert.sum()),
                  "tau_0":.5,"gamma_0":0.0,
                  "V0":_summary_vec(v0),"VR":_summary_vec(vr),"C":_summary_vec(cov),
                  "C_over_V0":_summary_vec(if_result["C_over_V0"]),
                  "VR_over_V0":_summary_vec(if_result["VR_over_V0"]),
                  "V0_vs_AUDIT_empirical_variance_correlation":v0_audit_corr,
                  "IF_density_bandwidth":kde_bandwidth,
                  "W_positive_count":int(np.count_nonzero(var_if>0)),
                  "W_zero_count":int(np.count_nonzero(var_if==0)),
                  "W_negative_or_invalid_count":int(len(var_if)-np.count_nonzero(valid)),
                  "q_IF_status":if_status,
                  "H_condition_number":if_result["hessian_condition_number"],
                  "audit_empirical_variance_mean":float(np.mean(audit_variance)),
                  "empirical_INT_variance_mean_EVAL":float(np.mean(s2e)),
                  "mapping_fit_calls":2,"raw_tweedie_calls":1,
                  "q_IF_INT_inference_calls":0}
            bank_records.append({"bank_id":bank,"operation":op,"metrics":metrics,
                                "covariance_diagnostics":diag,
                                "methods":{k:{"status":if_status if k=="q_IF_analytic" else "ESTIMABLE"}
                                           for k in scores}})
            cache={"z0cal":z0cal,"z0eval":z0eval,"cal_alert_mask":cal_alert,
                   "eval_alert_mask":eval_alert,"sigma2":np.asarray([sigma2]),
                   "zint_cal":refs["INT"][:ncal],
                   "zint_eval":refs["INT"][ncal:],
                   "zaudit_cal":refs["AUDIT"][:ncal],
                   "zaudit_eval":refs["AUDIT"][ncal:],
                   "q_if":q_if,"if_valid":valid,"V0":v0,"VR":vr,"C":cov}
            prediction_payloads[f"bank{bank}/{op}"]=cache
    # The deterministic alpha=1 endpoint reuses D0 exactly: no duplicate fits,
    # no pseudo-independent AUDIT references, and no fitted maps.
    alpha1_records=[]
    for bank in range(3):
        z0=np.asarray(prediction_payloads[f"bank{bank}"]["z0eval"])
        alert=z0>=0.0
        n=int(alert.sum())
        if n==0: raise ExperimentError("P2 alpha=1 has no D0 alerts")
        zeros=np.zeros((n,11),dtype=np.int8); t=np.zeros(n)
        q=np.zeros(n)
        alpha1_records.append({"bank_id":bank,"operation":"alpha_1_deterministic_reuse",
            "status":"EXACT_ZERO_FLIP_CONTROL_NO_REFITS",
            "D0_alert_groups":n,"C_over_V0":1.0,"VR_over_V0":1.0,
            "conditional_variance_W":0.0,"flip_risk":0.0,
            "Brier_finite_AUDIT":0.0,"AUC_vs_AUDIT_flip":None,
            "mapping_fit_calls":0,"prediction_model_fits":0})
    # Preserve prediction arrays for audit and zero-fit rebuilds.
    for bank in range(3):
        payload={}
        for key,val in prediction_payloads.items():
            if key==f"bank{bank}": payload.update({f"shared_{k}":v for k,v in val.items()})
            elif key.startswith(f"bank{bank}/"): payload.update({key.split("/",1)[1]+"_"+k:v for k,v in val.items()})
        cache_dir=RUN_ROOT/"adult_p2"/f"bank-{bank}"/"prediction_cache"
        if not cache_dir.exists(): save_numpy_archive(cache_dir,payload)
    # Aggregate numeric metrics separately for each operation and method.
    aggregate={}
    ops=("alpha_0","alpha_0_5","bootstrap")
    metric_names=sorted({k for row in bank_records for method in row["metrics"].values()
                         for k,v in method.items() if isinstance(v,(int,float)) and v is not None})
    for op in ops:
        aggregate[op]={}
        for method in P2_METHODS:
            selected=[r for r in bank_records if r["operation"]==op and method in r["metrics"]]
            aggregate[op][method]={}
            for key in metric_names:
                vals=[]
                for row in selected:
                    value=row["metrics"][method].get(key)
                    if isinstance(value,(int,float)) and value is not None and math.isfinite(float(value)):
                        vals.append(float(value))
                aggregate[op][method][key]={"mean":float(np.mean(vals)) if vals else None,
                    "min":float(np.min(vals)) if vals else None,"max":float(np.max(vals)) if vals else None,
                    "available_banks":len(vals)}
    eval_wall_seconds=time.monotonic()-eval_wall_start
    eval_cpu_seconds=time.process_time()-eval_cpu_start
    peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    # Three earlier report attempts failed before publication.  Record their
    # actual inference/calibration work so final counts include all retries.
    prior_failed_attempts=[
        {"status":"FAILED_BEFORE_PUBLICATION_METHOD_SET_ASSERTION",
         "bank_id":0,"model_inference_batches":52,"mapping_fit_calls":2,
         "raw_tweedie_calls":1,"prediction_model_fits":0},
        {"status":"FAILED_ALPHA_0_5_IF_COVARIANCE_SCALING_NAME_ERROR",
         "bank_id":0,"model_inference_batches":52,"mapping_fit_calls":4,
         "raw_tweedie_calls":2,"prediction_model_fits":0,
         "raw_prediction_cache_persisted_before_failure":True},
        {"status":"FAILED_BANK_1_PROBABILITY_SATURATION_AT_LOGIT_CONVERSION",
         "bank_id":1,"model_inference_batches":52,"mapping_fit_calls":0,
         "raw_tweedie_calls":0,"prediction_model_fits":0,
         "bank_0_raw_cache_reused":True,"bank_1_probability_cache_preserved":True},
        {"status":"FAILED_BEFORE_PUBLICATION_CSV_MISSING_MEAN_AUDIT_FIELD",
         "bank_ids":[0,1,2],"model_inference_batches":156,"mapping_fit_calls":18,
         "raw_tweedie_calls":9,"prediction_model_fits":0,
         "logit_prediction_caches_persisted":True},
    ]
    prior_model_batches=sum(x["model_inference_batches"] for x in prior_failed_attempts)
    prior_mapping_fits=sum(x["mapping_fit_calls"] for x in prior_failed_attempts)
    prior_tweedie_calls=sum(x["raw_tweedie_calls"] for x in prior_failed_attempts)
    report={"status":"P2_THREE_BANK_COUPLING_AND_IF_COMPLETE",
        "protocol_sha256":sha_bytes(PROTOCOL_PATH.read_bytes()),
        "model_fits":156,"model_inference_batches":prior_model_batches+inference_calls,
        "model_inference_batches_successful_report_pass":inference_calls,
        "model_inference_batches_reused_logit_cache":reused_inference_calls,
        "mapping_fit_calls":prior_mapping_fits+mapping_fits,
        "mapping_fit_calls_successful_report_pass":mapping_fits,
        "raw_tweedie_calls":prior_tweedie_calls+raw_tweedie_calls,
        "prior_failed_attempts":prior_failed_attempts,
        "if_analytic_int_model_inference_calls":0,
        "alpha_1":alpha1_records,
        "per_bank_operation_results":bank_records,
        "aggregate_mean_min_max":aggregate,
        "resources":{"evaluation_wall_seconds":eval_wall_seconds,
            "evaluation_cpu_seconds":eval_cpu_seconds,"evaluation_peak_rss_bytes":peak_rss_bytes,
            "native_threads":1,"gpu_used":False,
            "fit_wall_seconds_sum":sum(read_json(RUN_ROOT/"adult_p2"/f"bank-{b}"/"fit_measurement"/"summary.json")["total_fit_wall_seconds"] for b in range(3)),
            "fit_cpu_seconds_sum":sum(read_json(RUN_ROOT/"adult_p2"/f"bank-{b}"/"fit_measurement"/"summary.json")["total_fit_cpu_seconds"] for b in range(3)),
            "fit_peak_rss_max_bytes":max(read_json(RUN_ROOT/"adult_p2"/f"bank-{b}"/"fit_measurement"/"summary.json")["peak_rss_bytes"] for b in range(3))},
        "limitations":["Adult CAL512 and adult.test were used in P1 previously; P2 is exploratory and not an unseen confirmation.",
            "The three 4,000-group banks are disjoint source blocks; CAL/EVAL sets and the same fixed algorithm family are shared.",
            "q_IF is a first-order regularized Logistic approximation with Silverman KDE derivatives; nonpositive W is explicitly marked nonestimable.",
            "For alpha=0 the disjoint B pool is only a finite public training pool, not a new-population sampling guarantee."]}
    per=io.StringIO(newline="")
    fields=["bank_id","operation","method","status","n_alert_groups","if_valid_alert_groups","if_total_alert_groups","if_not_estimable_reason",*[f"R_{int(c*100):03d}" for c in RISK_COVERAGES],
            "AURC100","AURC_70_100","AUC_vs_AUDIT_flip","Brier_finite_AUDIT",
            "paired_delta_Brier_vs_margin_logistic","paired_delta_Brier_vs_q_sqrt2","ECE10",
            "mean_AUDIT_flip_rate"]
    w=csv.DictWriter(per,fieldnames=fields); w.writeheader()
    for row in bank_records:
        for method in P2_METHODS:
            m=row["metrics"].get(method,{})
            flat={}
            if m.get("status")=="PARTIAL_IF_SUPPORT":
                flat={"status":"PARTIAL_IF_SUPPORT",
                      "if_valid_alert_groups":m.get("valid_alert_groups"),
                      "if_total_alert_groups":m.get("total_alert_groups")}
            elif m.get("status")=="NOT_ESTIMABLE":
                flat={"status":"NOT_ESTIMABLE",
                      "if_valid_alert_groups":m.get("valid_alert_groups"),
                      "if_total_alert_groups":m.get("total_alert_groups"),
                      "if_not_estimable_reason":m.get("reason")}
            else:
                flat=m
            w.writerow({"bank_id":row["bank_id"],"operation":row["operation"],"method":method,
                        "status":row["methods"].get(method,{}).get("status","NOT_ESTIMABLE"),**flat})
    covstream=io.StringIO(newline="")
    covfields=["bank_id","operation","sigma_g_sq_CAL","CAL_alert_groups","EVAL_alert_groups",
        "C_over_V0_mean","C_over_V0_min","C_over_V0_max","VR_over_V0_mean","VR_over_V0_min","VR_over_V0_max",
        "V0_mean","V0_min","V0_max","pearson_r_V0_AUDITvar","spearman_r_V0_AUDITvar",
        "q_IF_status","W_positive_count","W_negative_or_invalid_count","IF_density_bandwidth"]
    cw=csv.DictWriter(covstream,fieldnames=covfields);cw.writeheader()
    for row in bank_records:
        d=row["covariance_diagnostics"]
        rr={"bank_id":row["bank_id"],"operation":row["operation"],"sigma_g_sq_CAL":d["sigma_g_sq_CAL"],
            "CAL_alert_groups":d["CAL_alert_groups"],"EVAL_alert_groups":d["EVAL_alert_groups"],
            "q_IF_status":d["q_IF_status"],"W_positive_count":d["W_positive_count"],
            "W_negative_or_invalid_count":d["W_negative_or_invalid_count"],"IF_density_bandwidth":d["IF_density_bandwidth"]}
        for metric in ("C_over_V0","VR_over_V0","V0"):
            for stat in ("mean","min","max"): rr[f"{metric}_{stat}"]=d[metric][stat]
        rr["pearson_r_V0_AUDITvar"]=d["V0_vs_AUDIT_empirical_variance_correlation"]["pearson_r"]
        rr["spearman_r_V0_AUDITvar"]=d["V0_vs_AUDIT_empirical_variance_correlation"]["spearman_r"]
        cw.writerow(rr)
    write_new_directory(OUT_ROOT/"p2_three_bank_results",{
        "p2_report.json":canonical(report),
        "p2_per_bank_operation_metrics.csv":per.getvalue().encode(),
        "p2_covariance_diagnostics.csv":covstream.getvalue().encode(),
    })
    print(json.dumps({"stage":"report-p2","status":report["status"],
        "model_fits":156,"total_inference_batches":report["model_inference_batches"],
        "new_inference_batches_this_report_run":inference_calls,
        "reused_logit_cache_batches":reused_inference_calls,
        "mapping_fits_total_including_retries":report["mapping_fit_calls"],
        "qIF_partial_count":sum(1 for r in bank_records if r["covariance_diagnostics"]["q_IF_status"]!="ESTIMABLE_FULL_ALERT_SUPPORT")},ensure_ascii=False))
    return report


def selfcheck():
    r=compute_tie_aware_risk(np.array([0.,0.,1.]),np.array([1.,0.,1.]),.5)
    if abs(r-.5)>1e-15: raise AssertionError("tie-aware risk check failed")
    assert set(DIABETES_FEATURES).isdisjoint({"encounter_id","patient_nbr","readmitted"})
    qlex=np.array([(k+.25+.5*expit(-z))/7 for k in range(7) for z in (0.,1.,2.)])
    for k in range(6):
        lo=(k+.25)/7; hi=(k+.75)/7
        assert np.all((qlex[3*k:3*k+3]>lo-.5/7-1e-12))
        assert qlex[3*k] > qlex[3*k+1] > qlex[3*k+2]
        assert hi < (k+1+.25)/7
    assert float(ndtr(-(-0.7)/math.sqrt(.2)))>.5
    return {"tie_risk":r,"lex_tie_break":True,"signed_probability_above_half":True,
            "random_draws":0,"model_fits":0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["selfcheck","prepare-diabetes","fit-diabetes-bank0",
        "fit-diabetes-rest","report-p3","adult-surrogate","fit-p2-bank0","fit-p2-rest","report-p2"])
    args=parser.parse_args()
    load_protocol()
    if args.stage=="selfcheck": print(json.dumps(selfcheck(),ensure_ascii=False)); return 0
    if args.stage=="prepare-diabetes": prepare_diabetes(); return 0
    if args.stage=="fit-diabetes-bank0": fit_diabetes_bank(0); return 0
    if args.stage=="fit-diabetes-rest": fit_diabetes_rest(); return 0
    if args.stage=="report-p3": evaluate_diabetes_p3(); return 0
    if args.stage=="adult-surrogate": evaluate_adult_surrogate(); return 0
    if args.stage=="fit-p2-bank0": fit_p2_bank(0); return 0
    if args.stage=="fit-p2-rest": fit_p2_rest(); return 0
    if args.stage=="report-p2": evaluate_p2(); return 0
    raise AssertionError("unreachable stage")


if __name__=="__main__": raise SystemExit(main())
