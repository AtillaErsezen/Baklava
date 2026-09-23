"""
ML Factory — Modal training backend.

  uv run modal deploy modal_train.py                                  # agent calls the deployed functions
  uv run modal run modal_train.py --csv data/churn.csv --target Churn # smoke test WITHOUT the agent (do this first!)
"""
import modal

APP_NAME = "ml-factory"
VOLUME_NAME = "ml-factory-data"
DATA_DIR = "/data"
SEED = 42

# (task, model) -> (module, class, default kwargs). The agent's params override the defaults.
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


def upload_dataset(df) -> str:
    """Write df as parquet to the Modal volume. Returns the path inside the volume."""
    import os
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
        with vol.batch_upload(force=True) as batch:
            batch.put_file(p, remote)
    return remote


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


def _read(path):
    """Read a parquet file from the volume (path as returned by upload_dataset)."""
    import pandas as pd

    vol.reload()
    return pd.read_parquet(f"{DATA_DIR}{path}")


def _xy(df, spec, classes=None):
    """Split a DataFrame into (X, y, classes) for one spec.
    Drops rows with a missing target, sorts by time_column when one is given (required
    for timeseries CV), and removes drop_columns. For classification, y is encoded as
    0..K-1 in `classes` order; `classes` defaults to the sorted labels of this df. Pass
    the dev set's classes when encoding the hidden set, so both share one encoding;
    rows whose label is not in `classes` are dropped. For regression, y is float and
    `classes` is None."""
    import pandas as pd

    target = spec["target"]
    df = df.dropna(subset=[target])
    if spec.get("time_column") and spec["time_column"] in df.columns:
        df = df.sort_values(spec["time_column"], kind="stable").reset_index(drop=True)
    X = _clean(df.drop(columns=[target, *spec.get("drop_columns", [])], errors="ignore"))
    if spec["task"] != "classification":
        return X, df[target].astype(float).to_numpy(), None
    labels = df[target].astype(str)
    classes = classes or sorted(labels.unique().tolist())
    codes = pd.Categorical(labels, categories=classes).codes
    keep = codes >= 0
    return X[keep].reset_index(drop=True), codes[keep].astype(int), classes


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
    rows must already be in time order; cv_repeats is ignored), otherwise a shuffled,
    seeded (Repeated)StratifiedKFold for classification or (Repeated)KFold for
    regression. The fixed seed gives every candidate the same folds, so per-fold
    scores are paired across candidates."""
    from sklearn.model_selection import (KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold,
                                         TimeSeriesSplit)

    k, r = spec.get("cv_folds", 5), spec.get("cv_repeats", 1)
    if spec.get("cv") == "timeseries":
        return TimeSeriesSplit(n_splits=k)
    strat = spec["task"] == "classification"
    if r > 1:
        return (RepeatedStratifiedKFold if strat else RepeatedKFold)(n_splits=k, n_repeats=r, random_state=SEED)
    return (StratifiedKFold if strat else KFold)(k, shuffle=True, random_state=SEED)


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


def _failure(spec, e):
    """Error result in the shape the agent expects, with a trimmed traceback."""
    import traceback

    return {"name": spec.get("name"), "ok": False, "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:]}


def _train(df, spec):
    """Cross-validate one candidate on df. Per metric: mean, std, train_mean, and the
    per-fold test scores (`folds`, same fold order for every candidate, so they can
    be compared pairwise). Error metrics are reported positive. Never raises."""
    import time

    from sklearn.model_selection import cross_validate

    t0 = time.time()
    try:
        X, y, classes = _xy(df, spec)
        scoring = _scoring(spec["task"], len(classes) if classes else 0)
        res = cross_validate(
            _build_pipeline(X, spec), X, y, cv=_cv(spec), scoring=scoring,
            return_train_score=True, error_score="raise",
        )
        metrics = {}
        for name, scorer in scoring.items():
            sign = -1 if scorer.startswith("neg_") else 1
            val, tr = sign * res[f"test_{name}"], sign * res[f"train_{name}"]
            metrics[name] = {"mean": round(float(val.mean()), 5), "std": round(float(val.std()), 5),
                             "train_mean": round(float(tr.mean()), 5), "folds": [round(float(v), 6) for v in val]}
        return {"name": spec["name"], "ok": True, "metrics": metrics,
                "n_rows": len(X), "fit_seconds": round(time.time() - t0, 1)}
    except Exception as e:
        return _failure(spec, e)


def _predict_holdout(dev, hidden, spec):
    """Fit on dev, predict the hidden set once. Returns the hidden labels `y` (encoded
    with the dev classes), `pred` (positive-class probability for binary, a probability
    row per sample for multiclass, values for regression), fit seconds and predict ms
    per 1k rows. Never raises."""
    import time

    import numpy as np

    try:
        X, y, classes = _xy(dev, spec)
        Xh, yh, _ = _xy(hidden, spec, classes)
        pipe = _build_pipeline(X, spec)
        t0 = time.perf_counter()
        pipe.fit(X, y)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if spec["task"] == "classification":
            pred = pipe.predict_proba(Xh)
            pred = pred[:, 1] if pred.shape[1] == 2 else pred
        else:
            pred = pipe.predict(Xh)
        ms_per_1k = (time.perf_counter() - t0) * 1000 * 1000 / max(len(Xh), 1)
        return {"name": spec["name"], "ok": True, "y": np.asarray(yh).tolist(),
                "pred": np.round(np.asarray(pred, dtype=float), 6).tolist(), "classes": classes,
                "fit_seconds": round(fit_s, 2), "predict_ms_per_1k": round(ms_per_1k, 2)}
    except Exception as e:
        return _failure(spec, e)


@app.function(image=image, volumes={DATA_DIR: vol}, cpu=4, memory=8192, timeout=1200)
def train_candidate(spec: dict) -> dict:
    """Cross-validate ONE candidate. Never raises: errors come back so the agent can react."""
    try:
        df = _read(spec["dataset_path"])
    except Exception as e:
        return _failure(spec, e)
    return _train(df, spec)


@app.function(image=image, volumes={DATA_DIR: vol}, cpu=4, memory=8192, timeout=1200)
def predict_holdout(spec: dict, hidden_path: str) -> dict:
    """Fit ONE candidate on the dev set (spec's dataset_path) and predict the hidden set."""
    try:
        dev, hidden = _read(spec["dataset_path"]), _read(hidden_path)
    except Exception as e:
        return _failure(spec, e)
    return _predict_holdout(dev, hidden, spec)


@app.function(image=image, volumes={DATA_DIR: vol}, cpu=4, memory=8192, timeout=1800)
def fit_final(spec: dict, run_id: str) -> dict:
    """Fit the winner on all data, save it to the volume, return path + top features."""
    import os

    import joblib

    X, y, classes = _xy(_read(spec["dataset_path"]), spec)
    pipe = _build_pipeline(X, spec)
    pipe.fit(X, y)
    out_dir = f"{DATA_DIR}/models/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{spec['name']}.joblib"
    joblib.dump({"pipeline": pipe, "classes": classes, "spec": spec}, path)
    vol.commit()
    return {"model_path": path.removeprefix(DATA_DIR), "n_rows": len(X), "top_features": _top_features(pipe)}


@app.local_entrypoint()
def smoke(csv: str, target: str, task: str = "classification"):
    """Every model in the menu with defaults, 3-fold. If this works, the backend is done."""
    path = upload_dataset(load_local(csv))
    base = {"dataset_path": path, "target": target, "task": task, "cv_folds": 3}
    specs = [{**base, "name": m, "model": m} for m in MODEL_MENU[task]]
    for r in train_candidate.map(specs):
        print(r["name"], "→", {k: v["mean"] for k, v in r["metrics"].items()} if r["ok"] else r["error"])
