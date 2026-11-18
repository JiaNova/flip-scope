"""Adult parsing, preprocessing and calibration primitives used by the paper."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from inflight import public_io as controlled
from inflight import flip  # Public comparator namespace reused by the experiment runners.

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = controlled.ROOT
CONFIG_REL = Path("configs/adult.json")
DATA_REL = Path("data/public/adult")
CHECKPOINT_REL = Path("runs/adult_pilot")
OUTPUT_REL = Path("reports/adult_pilot")
PURPOSE = "public_retrain_flip_pilot"
FEATURES = (
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
)
NUMERIC = frozenset({"age", "fnlwgt", "education-num", "capital-gain",
                     "capital-loss", "hours-per-week"})
CATEGORICAL = tuple(name for name in FEATURES if name not in NUMERIC)
MODEL_IDS = ("D0",)+tuple(f"INT-{i:02d}" for i in range(1, 7))+tuple(
    f"AUDIT-{i:02d}" for i in range(1, 12))
MAPPING_IDS = ("isotonic_margin", "logistic_nll_margin", "logistic_brier_margin",
               "isotonic_tweedie_score")
EXTRA_IDS = ("SCALE", "KDE")
SLOT_IDS = MODEL_IDS+MAPPING_IDS[:3]+EXTRA_IDS+MAPPING_IDS[3:]

class M3RError(ValueError):
    """Invalid fixed design, public source, reservation or result."""

def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise M3RError("nonfinite values are not serializable")
    return value

def canonical_json(value) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")

def strict_json(raw: bytes):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise M3RError(f"duplicate JSON key {key!r}")
            out[key] = value
        return out
    def reject(value):
        raise M3RError(f"nonfinite JSON token {value!r}")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _read_fixed(relative: Path) -> tuple[bytes, str]:
    path = controlled.validate_project_path(ROOT/relative, allow_missing_leaf=False)
    if path.is_symlink() or not path.is_file():
        raise M3RError(f"fixed input is not a regular file: {relative}")
    raw = path.read_bytes()
    return raw, sha256(raw)

def load_config() -> tuple[dict, str]:
    raw, digest = _read_fixed(CONFIG_REL)
    config = strict_json(raw)
    validate_config(config)
    return config, digest

def validate_config(config: Mapping) -> None:
    """Check feature identity and separation of evaluation from fitted maps."""
    if tuple(config.get("features", {}).get("column_order", ())) != FEATURES:
        raise M3RError("Adult feature order does not match saved preprocessing")
    if config.get("partition", {}).get("test_income_labels_used") is not False:
        raise M3RError("Adult test income labels must not enter the analysis")
    if config.get("score_and_targets", {}).get("audit_outputs_in_mapping_fit") is not False:
        raise M3RError("AUDIT outputs must not enter mapping fits")


def _feature_value(name: str, raw: str):
    value = raw.strip()
    if name in NUMERIC:
        if not value or value == "?":
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise M3RError(f"invalid integer feature {name}={value!r}") from exc
        return parsed
    return "__MISSING__" if value in ("", "?") else value

def _feature_group_id(features: tuple) -> str:
    typed = []
    for name, value in zip(FEATURES, features):
        if name in NUMERIC:
            typed.append(["numeric", value])
        else:
            typed.append(["categorical", value])
    return sha256(canonical_json(typed))

def _parse_adult(raw: bytes, source_name: str, *, keep_income: bool) -> dict[str, dict]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise M3RError(f"UCI {source_name} is not ASCII") from exc
    groups: dict[str, dict] = {}
    valid_labels = {"<=50K": 0, ">50K": 1}
    seen_rows = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("|"):
            continue
        if raw_line.count(",") != 14:
            raise M3RError(f"{source_name}:{line_number} has an invalid field count")
        if keep_income:
            row = next(csv.reader([raw_line]))
            if len(row) != 15:
                raise M3RError(f"{source_name}:{line_number} has {len(row)} fields; expected 15")
            raw_features = row[:14]
        else:
            # Extract only the fourteen feature spans; the final income field
            # remains an unparsed byte/string suffix and is never retained.
            delimiters = []
            cursor = 0
            for _ in range(14):
                cursor = raw_line.find(",", cursor)
                if cursor < 0:
                    raise M3RError(f"{source_name}:{line_number} is truncated")
                delimiters.append(cursor)
                cursor += 1
            raw_features = [raw_line[:delimiters[0]]]
            raw_features.extend(raw_line[delimiters[i]+1:delimiters[i+1]]
                                for i in range(13))
            if len(raw_features) != 14:
                raise AssertionError("adult.test feature parser did not return 14 fields")
        features = tuple(_feature_value(name, value) for name, value in zip(FEATURES, raw_features))
        group_id = _feature_group_id(features)
        group = groups.setdefault(group_id, {
            "group_id": group_id, "features": features, "labels": [], "row_numbers": [],
        })
        if group["features"] != features:
            raise AssertionError("feature SHA collision detected")
        if keep_income:
            label_text = row[14].strip().rstrip(".")
            if label_text not in valid_labels:
                raise M3RError(f"{source_name}:{line_number} has invalid training label")
            group["labels"].append(valid_labels[label_text])
        # The final field of adult.test is intentionally not inspected or retained.
        group["row_numbers"].append(line_number)
        seen_rows += 1
    if not seen_rows:
        raise M3RError(f"UCI {source_name} contained no records")
    return groups

def _partition_integer(salt: str, group_id: str) -> int:
    digest = hashlib.sha256((salt+"\n"+group_id).encode("ascii")).digest()
    return int.from_bytes(digest, "big", signed=False)

def _fit_preprocessor(train_groups: list[dict]) -> dict:
    if not train_groups:
        raise M3RError("train-only preprocessing has no feature groups")
    medians = {}
    for name in FEATURES:
        if name not in NUMERIC:
            continue
        index = FEATURES.index(name)
        values = sorted(float(group["features"][index]) for group in train_groups
                        if group["features"][index] is not None)
        if not values:
            raise M3RError(f"training partition has no observed values for {name}")
        mid = len(values)//2
        medians[name] = (values[mid] if len(values) % 2 else
                         (values[mid-1]+values[mid])/2.0)
    vocabularies = {}
    for name in CATEGORICAL:
        index = FEATURES.index(name)
        values = {str(group["features"][index]) for group in train_groups}
        values.update(("__MISSING__", "__UNK__"))
        vocabularies[name] = sorted(values)
    return {"schema_version": 1, "learned_from": "training feature groups only",
            "numeric_medians": medians, "categorical_vocabularies": vocabularies,
            "numeric_scaling": "none", "fit_group_count": len(train_groups)}

def _transform_features(groups: list[dict], preprocessing: Mapping) -> np.ndarray:
    medians = preprocessing["numeric_medians"]
    vocabularies = preprocessing["categorical_vocabularies"]
    category_ids = {name: {value: index for index, value in enumerate(vocabularies[name])}
                    for name in CATEGORICAL}
    rows = []
    for group in groups:
        values = []
        for name, raw in zip(FEATURES, group["features"]):
            if name in NUMERIC:
                values.append(float(medians[name] if raw is None else raw))
            else:
                token = str(raw)
                index = category_ids[name].get(token, category_ids[name]["__UNK__"])
                values.append(float(index))
        rows.append(values)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES) or not np.isfinite(matrix).all():
        raise M3RError("fixed Adult feature construction produced an invalid matrix")
    return matrix

def _group_assignment_manifest(data_groups: Mapping, test_groups: Mapping, config: Mapping):
    test_ids = set(test_groups)
    train_salt = config["partition"]["training_calibration_hash_salt"]
    subset_salt = config["partition"]["calibration_subset_salt"]
    train, calibration, overlap = [], [], []
    for group_id in sorted(data_groups):
        group = data_groups[group_id]
        if group_id in test_ids:
            overlap.append(group_id)
            continue
        value = _partition_integer(train_salt, group_id)
        if value*10 < 7*(1 << 256):
            train.append(group)
        else:
            calibration.append(group)
    if not train or not calibration:
        raise M3RError("fixed SHA partition produced an empty train or calibration partition")
    calibration.sort(key=lambda group: (_partition_integer(subset_salt, group["group_id"]),
                                        group["group_id"]))
    selected = calibration[:512]
    test = [test_groups[key] for key in sorted(test_groups)]
    if {x["group_id"] for x in train} & {x["group_id"] for x in selected}:
        raise AssertionError("train/calibration group overlap")
    if ({x["group_id"] for x in train} | {x["group_id"] for x in selected}) & set(test_ids):
        raise AssertionError("Adult test feature groups leaked into train/calibration")
    return {"train": train, "calibration": selected, "calibration_candidates": calibration,
            "evaluation": test, "overlap_removed": overlap}

def logistic_objective(theta: np.ndarray, z: np.ndarray, target: np.ndarray,
                       *, loss: str, regularization: float = 1e-6,
                       weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Fixed two-parameter soft-target NLL/Brier objective and analytic gradient."""
    z = np.asarray(z, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if z.ndim != 1 or target.shape != z.shape or theta.shape != (2,) or not z.size:
        raise M3RError("logistic objective arrays must be aligned and nonempty")
    if not np.isfinite(z).all() or not np.isfinite(target).all() or np.any((target < 0)|(target > 1)):
        raise M3RError("invalid margin or soft calibration target")
    w = np.ones_like(z) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != z.shape or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise M3RError("invalid calibration group weights")
    design = np.column_stack((np.ones_like(z), z))
    prediction = expit(design@theta)
    if loss == "nll":
        eps = np.finfo(float).eps
        value = -np.sum(w*(target*np.log(np.maximum(prediction, eps))+
                           (1-target)*np.log(np.maximum(1-prediction, eps))))
        gradient = design.T@(w*(prediction-target))
    elif loss == "brier":
        residual = prediction-target
        value = np.sum(w*residual**2)
        gradient = 2.0*design.T@(w*residual*prediction*(1-prediction))
    else:
        raise M3RError(f"unsupported fixed logistic loss {loss!r}")
    value += regularization*theta[1]**2
    gradient[1] += 2*regularization*theta[1]
    return float(value), np.asarray(gradient, dtype=np.float64)

def fit_mapping(method_id: str, x: np.ndarray, target: np.ndarray,
                weights: np.ndarray | None = None) -> dict:
    """Fit one declared calibration map; callers reserve/persist before applying."""
    x = np.asarray(x, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or target.shape != x.shape or w.shape != x.shape or not x.size:
        raise M3RError("mapping requires aligned nonempty calibration vectors")
    if not np.isfinite(x).all() or not np.isfinite(target).all() or np.any(w <= 0):
        raise M3RError("mapping requires finite data and positive group weights")
    if method_id == "isotonic_margin":
        model = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0, y_max=1)
        model.fit(x, target, sample_weight=w)
        return {"kind": "isotonic", "method_id": method_id,
                "x_thresholds": model.X_thresholds_.astype(float).tolist(),
                "y_thresholds": model.y_thresholds_.astype(float).tolist(),
                "direction": "nonincreasing"}
    if method_id == "isotonic_tweedie_score":
        model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0, y_max=1)
        model.fit(x, target, sample_weight=w)
        return {"kind": "isotonic", "method_id": method_id,
                "x_thresholds": model.X_thresholds_.astype(float).tolist(),
                "y_thresholds": model.y_thresholds_.astype(float).tolist(),
                "direction": "nondecreasing"}
    loss = {"logistic_nll_margin": "nll", "logistic_brier_margin": "brier"}.get(method_id)
    if loss is None:
        raise M3RError(f"mapping method is not in the fixed batch: {method_id}")
    result = minimize(lambda theta: logistic_objective(theta, x, target,
                                                       loss=loss, weights=w),
                      x0=np.zeros(2), jac=True, method="L-BFGS-B",
                      bounds=[(-20, 20), (-20, 20)],
                      options={"maxiter": 300, "maxfun": 6000, "ftol": 1e-12,
                               "gtol": 1e-8, "maxls": 20})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise M3RError(f"{method_id} optimizer returned nonfinite parameters")
    if not result.success:
        raise M3RError(f"{method_id} fixed L-BFGS-B fit did not converge: {result.message}")
    return {"kind": "logistic", "method_id": method_id,
            "intercept": float(result.x[0]), "slope": float(result.x[1]),
            "loss": loss, "objective": float(result.fun),
            "success": bool(result.success), "message": str(result.message),
            "nit": int(result.nit), "nfev": int(result.nfev),
            "regularization": 1e-6, "initial": [0.0, 0.0],
            "bounds": [[-20, 20], [-20, 20]], "restarts": 0}

def apply_mapping(model: Mapping, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if model["kind"] == "isotonic":
        return np.interp(x, model["x_thresholds"], model["y_thresholds"],
                         left=model["y_thresholds"][0], right=model["y_thresholds"][-1])
    return expit(float(model["intercept"])+float(model["slope"])*x)

def _dataframe(matrix: np.ndarray, preprocessing: Mapping) -> pd.DataFrame:
    frame = pd.DataFrame(np.asarray(matrix), columns=FEATURES)
    for name in CATEGORICAL:
        vocab = preprocessing["categorical_vocabularies"][name]
        categories = list(range(len(vocab)))
        frame[name] = pd.Categorical(frame[name].astype(np.int64),
                                     categories=categories, ordered=False)
    return frame
