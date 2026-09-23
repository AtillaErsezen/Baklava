"""
ML Factory — Modal training backend.

  modal deploy modal_train.py                                        # agent calls the deployed functions
  modal run modal_train.py --csv data/titanic.csv --target Survived  # smoke test WITHOUT the agent (do this first!)
"""
import modal

APP_NAME = "ml-factory"
VOLUME_NAME = "ml-factory-data"
DATA_DIR = "/data"
SEED = 42
PIPELINE_VERSION = "tracking-v1"

# (task, model) -> (module, class, default kwargs). Claude's params override the defaults.
ESTIMATORS = {
    ("classification", "logreg"): ("sklearn.linear_model", "LogisticRegression", {"max_iter": 2000}),
    ("classification", "random_forest"): ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 300, "n_jobs": -1, "random_state": SEED}),
    ("classification", "lightgbm"): ("lightgbm", "LGBMClassifier", {"random_state": SEED, "verbose": -1}),
    ("classification", "xgboost"): ("xgboost", "XGBClassifier", {"random_state": SEED, "n_jobs": -1, "tree_method": "hist"}),
    ("classification", "mlp"): ("sklearn.neural_network", "MLPClassifier", {"max_iter": 500, "early_stopping": True, "random_state": SEED}),
    ("regression", "ridge"): ("sklearn.linear_model", "Ridge", {}),
    ("regression", "random_forest"): ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 300, "n_jobs": -1, "random_state": SEED}),
    ("regression", "lightgbm"): ("lightgbm", "LGBMRegressor", {"random_state": SEED, "verbose": -1}),
    ("regression", "xgboost"): ("xgboost", "XGBRegressor", {"random_state": SEED, "n_jobs": -1, "tree_method": "hist"}),
    ("regression", "mlp"): ("sklearn.neural_network", "MLPRegressor", {"max_iter": 500, "early_stopping": True, "random_state": SEED}),
}
MODEL_MENU = {
    t: [m for (tt, m) in ESTIMATORS if tt == t] for t in ("classification", "regression")
}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgomp1")  # lightgbm needs OpenMP
    # pandas/pyarrow must match the local venv: parquet is written locally, read here
    # ponytail: ML libs unpinned, pin to whatever the first green smoke test resolves
    .uv_pip_install("pandas==3.0.6", "pyarrow==25.0.1", "scikit-learn", "lightgbm", "xgboost", "joblib")
)


# ------------------------------------------------------------------ local side (your laptop)

def load_local(path: str):
    """Read CSV/Parquet and turn date-looking string columns into real datetimes."""
    import pandas as pd

    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    for c in df.columns:
        if not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
            continue
        sample = df[c].dropna().astype(str).head(200)
        if len(sample) == 0 or sample.str.len().min() < 6:
            continue
        try:
            if pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.9:
                df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
        except Exception:
            pass
    return df


def upload_dataset(df, with_metadata=False):
    """Write df as parquet to Modal; optionally return its path and exact byte hash."""
    import os
    import hashlib
    import tempfile
    import uuid

    df = df.copy()
    for c in df.columns:  # mixed-type object columns break pyarrow
        if df[c].dtype == object:
            df[c] = df[c].where(df[c].isna(), df[c].astype(str))
    remote = f"/datasets/{uuid.uuid4().hex[:12]}.parquet"
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "data.parquet")
        df.to_parquet(p, index=False)
        with open(p, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        with vol.batch_upload(force=True) as batch:
            batch.put_file(p, remote)
    return {"path": remote, "sha256": digest} if with_metadata else remote


# ------------------------------------------------------------------ remote side (Modal containers)

def _clean(X):
    """Coerce raw feature columns into types the sklearn pipeline can handle: booleans
    become 0/1 ints, each datetime column is replaced by _year/_month/_dow parts, and any
    other non-numeric column becomes object dtype holding strings, with missing values
    kept as NaN so the imputer still sees them. Returns a copy; X is not modified."""
    import pandas as pd

    X = X.copy()
    for c in list(X.columns):
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(int)
        elif pd.api.types.is_datetime64_any_dtype(s):
            X[f"{c}_year"], X[f"{c}_month"], X[f"{c}_dow"] = s.dt.year, s.dt.month, s.dt.dayofweek
            X = X.drop(columns=c)
        elif not pd.api.types.is_numeric_dtype(s):
            X[c] = s.where(s.isna(), s.astype(str)).astype(object)
    return X


def _load_xy(spec):
    """Read the volume dataset and return (X, y, classes, data_manifest).
    Drops rows with a missing target, sorts by time_column when one is given (required
    for timeseries CV), and removes drop_columns. For classification, y is
    label-encoded and `classes` holds the original labels in encoded order; for
    regression, y is float and `classes` is None."""
    import pandas as pd
    import hashlib
    import io
    from pathlib import Path

    vol.reload()
    raw = Path(f"{DATA_DIR}{spec['dataset_path']}").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if spec.get("prepared_sha256") and digest != spec["prepared_sha256"]:
        raise ValueError("Prepared dataset hash mismatch")
    df = pd.read_parquet(io.BytesIO(raw))
    source_count = len(df)
    target = spec["target"]
    df = df.dropna(subset=[target])
    if spec.get("time_column") and spec["time_column"] in df.columns:
        df = df.sort_values(spec["time_column"]).reset_index(drop=True)
    y = df[target]
    source_features = df.drop(columns=[target, *spec.get("drop_columns", [])], errors="ignore")
    X = _clean(source_features)
    manifest = {
        "dataset_id": spec.get("dataset_id"), "source_sha256": spec.get("source_sha256"),
        "prepared_data_path": spec["dataset_path"], "prepared_sha256": digest,
        "pipeline_version": PIPELINE_VERSION, "target": target,
        "source_row_count": source_count, "usable_row_count": len(X),
        "excluded_missing_target": source_count - len(X),
        "included_source_columns": source_features.columns.tolist(), "included_columns": X.columns.tolist(),
        "dropped_columns": [c for c in spec.get("drop_columns", []) if c in df.columns and c != target],
        "derived_columns": [c for c in X.columns if c not in source_features.columns],
    }
    classes = None
    if spec["task"] == "classification":
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        classes = le.classes_.tolist()
    else:
        y = y.astype(float).values
    return X, y, classes, manifest


def _effective_config(pipe):
    import importlib.metadata
    import json

    prep = pipe.named_steps["prep"]
    numeric = prep.transformers[0][1]
    categorical = prep.transformers[1][1]
    packages = ("pandas", "pyarrow", "scikit-learn", "numpy", "lightgbm", "xgboost")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return json.loads(json.dumps({
        "pipeline_version": PIPELINE_VERSION,
        "estimator": type(pipe.named_steps["model"]).__name__,
        "params": pipe.named_steps["model"].get_params(deep=False),
        "preprocessing": {
            "numeric_impute": numeric.named_steps["impute"].strategy,
            "scale": "scale" in numeric.named_steps,
            "categorical_encoder": type(categorical.named_steps["encode"]).__name__,
            "categorical_params": categorical.named_steps["encode"].get_params(deep=False),
            "numeric_columns": prep.transformers[0][2], "categorical_columns": prep.transformers[1][2],
        }, "library_versions": versions,
    }, default=str))


def _build_pipeline(X, spec):
    """Build an unfitted preprocessing + estimator Pipeline for one candidate.
    Numeric columns are imputed (median by default) and scaled by default only for the
    scale-sensitive models (logreg, ridge, mlp). Categorical columns are imputed with
    the most frequent value, then one-hot encoded (capped at max_categories, default 20)
    or ordinal encoded. The estimator gets the ESTIMATORS defaults with the spec's
    params merged over them. Raises ValueError if the model isn't on the task's menu."""
    import importlib

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    task, model = spec["task"], spec["model"]
    if (task, model) not in ESTIMATORS:
        raise ValueError(f"'{model}' is not in the {task} menu: {MODEL_MENU[task]}")
    prep = spec.get("preprocessing") or {}

    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]

    num_steps = [("impute", SimpleImputer(strategy=prep.get("numeric_impute", "median")))]
    if prep.get("scale", model in ("logreg", "ridge", "mlp")):
        num_steps.append(("scale", StandardScaler()))
    if prep.get("categorical", "onehot") == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        enc = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=prep.get("max_categories", 20))

    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", enc)]), cat),
    ])
    mod, cls, defaults = ESTIMATORS[(task, model)]
    est = getattr(importlib.import_module(mod), cls)(**{**defaults, **(spec.get("params") or {})})
    return Pipeline([("prep", pre), ("model", est)])


def _cv(spec):
    """Pick the CV splitter: TimeSeriesSplit for cv="timeseries" (no shuffling, so
    rows must already be in time order), otherwise a shuffled, seeded
    StratifiedKFold for classification or KFold for regression."""
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

    k = spec.get("cv_folds", 5)
    if spec.get("cv") == "timeseries":
        return TimeSeriesSplit(n_splits=k)
    if spec["task"] == "classification":
        return StratifiedKFold(k, shuffle=True, random_state=SEED)
    return KFold(k, shuffle=True, random_state=SEED)


def _scoring(task, n_classes):
    """Map our metric names to sklearn scorer names. Multiclass ROC AUC uses
    one-vs-rest. Error metrics use sklearn's negated scorers; train_candidate flips
    the sign back, so reported rmse and mae are positive."""
    if task == "classification":
        return {"accuracy": "accuracy", "f1_macro": "f1_macro", "roc_auc": "roc_auc" if n_classes == 2 else "roc_auc_ovr"}
    return {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def _top_features(pipe, k=10):
    """Return the k most important features of a fitted pipeline as
    [{"feature", "importance"}], using feature_importances_ for tree models or the mean
    absolute coef_ for linear ones. Names are post-encoding (e.g. one-hot columns).
    Returns [] for models with neither (MLP) or if the lookup fails; it is only
    informational and must never fail the final fit."""
    import numpy as np

    try:
        names = pipe.named_steps["prep"].get_feature_names_out()
        est = pipe.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            imp = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            imp = np.abs(np.asarray(est.coef_, dtype=float)).reshape(-1, len(names)).mean(0)
        else:
            return []
        return [{"feature": str(names[i]), "importance": round(float(imp[i]), 5)} for i in np.argsort(imp)[::-1][:k]]
    except Exception:
        return []


@app.function(image=image, volumes={DATA_DIR: vol}, cpu=4, memory=8192, timeout=1200)
def train_candidate(spec: dict) -> dict:
    """Cross-validate ONE candidate. Never raises: errors come back so the agent can react."""
    import time
    import traceback
    import hashlib
    import math

    from sklearn.model_selection import cross_validate

    t0 = time.time()
    metadata = {}
    try:
        X, y, classes, manifest = _load_xy(spec)
        metadata["data_manifest"] = manifest
        pipe = _build_pipeline(X, spec)
        metadata["effective_config"] = _effective_config(pipe)
        splitter = _cv(spec)
        splits = list(splitter.split(X, y))
        split_hash = hashlib.sha256()
        for train, validation in splits:
            split_hash.update(train.astype("<i8").tobytes())
            split_hash.update(b"|validation|")
            split_hash.update(validation.astype("<i8").tobytes())
            split_hash.update(b"|fold|")
        metadata["evaluation_config"] = {"method": type(splitter).__name__, "n_splits": len(splits),
                        "seed": None if spec.get("cv") == "timeseries" else SEED,
                        "time_column": spec.get("time_column"), "split_sha256": split_hash.hexdigest(),
                        "fold_sizes": [{"train": len(tr), "validation": len(val)} for tr, val in splits]}
        scoring = _scoring(spec["task"], len(classes) if classes else 0)
        res = cross_validate(
            pipe, X, y, cv=splits, scoring=scoring,
            return_train_score=True, error_score="raise",
        )
        metrics = {}
        for name, scorer in scoring.items():
            sign = -1 if scorer.startswith("neg_") else 1
            val, tr = sign * res[f"test_{name}"], sign * res[f"train_{name}"]
            metrics[name] = {"mean": round(float(val.mean()), 5), "std": round(float(val.std()), 5),
                             "train_mean": round(float(tr.mean()), 5),
                             "fold_scores": [float(v) for v in val], "train_fold_scores": [float(v) for v in tr]}
            if not all(math.isfinite(float(v)) for v in [*val, *tr]):
                raise ValueError(f"Non-finite {name} scores; inspect the dataset and split configuration")
        return {"name": spec["name"], "ok": True, "metrics": metrics,
                "n_rows": len(X), "fit_seconds": round(time.time() - t0, 1), **metadata}
    except Exception as e:
        return {"name": spec.get("name"), "ok": False, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1500:], "fit_seconds": round(time.time() - t0, 1), **metadata}


@app.function(image=image, volumes={DATA_DIR: vol}, cpu=4, memory=8192, timeout=1800)
def fit_final(spec: dict, run_id: str) -> dict:
    """Fit the winner on all data, save it to the volume, return path + top features."""
    import os

    import joblib

    X, y, classes, manifest = _load_xy(spec)
    pipe = _build_pipeline(X, spec)
    pipe.fit(X, y)
    out_dir = f"{DATA_DIR}/models/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{spec['name']}.joblib"
    joblib.dump({"pipeline": pipe, "classes": classes, "spec": spec}, path)
    vol.commit()
    return {"ok": True, "model_path": path.removeprefix(DATA_DIR), "n_rows": len(X), "top_features": _top_features(pipe),
            "data_manifest": manifest, "effective_config": _effective_config(pipe),
            "evaluation_config": {"method": "full_fit", "rows": len(X)}}


@app.function(image=image, volumes={DATA_DIR: vol}, cpu=4, memory=8192, timeout=1800)
def track_training(spec: dict, run_id: str, stage: str = "candidate_cv"):
    """Stream actual worker start and completion; no Supabase credentials enter the worker."""
    from datetime import datetime, timezone
    import time

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    yield {"kind": "started", "started_at": started_at}
    try:
        if stage == "candidate_cv":
            result = train_candidate.local(spec)
        elif stage == "final_fit":
            result = fit_final.local(spec, run_id)
        else:
            raise ValueError("Unknown training stage")
    except Exception as exc:
        result = {"name": spec["name"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    result.update(started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                  fit_seconds=round(time.monotonic() - t0, 3))
    yield {"kind": "finished", "result": result}


@app.local_entrypoint()
def smoke(csv: str, target: str, task: str = "classification"):
    """Every model in the menu with defaults, 3-fold. If this works, the backend is done."""
    path = upload_dataset(load_local(csv))
    base = {"dataset_path": path, "target": target, "task": task, "cv_folds": 3}
    specs = [{**base, "name": m, "model": m} for m in MODEL_MENU[task]]
    for r in train_candidate.map(specs):
        print(r["name"], "→", {k: v["mean"] for k, v in r["metrics"].items()} if r["ok"] else r["error"])
