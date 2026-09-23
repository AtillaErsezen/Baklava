"""
ML Factory: Modal training backend.

  modal deploy modal_train.py                                        # agent calls the deployed functions
  modal run modal_train.py --csv data/titanic.csv --target Survived  # smoke test WITHOUT the agent (do this first!)
"""
import os

import modal

APP_NAME = "ml-factory"
VOLUME_NAME = "ml-factory-data"
DATA_DIR = "/data"
SEED = 42

# ML_FACTORY_GPU=0 deploys without GPU functions (e.g. no payment method on the workspace): tabicl leaves
# the menu and the *_gpu functions run on the CPU image so their names still resolve.
GPU_ENABLED = os.environ.get("ML_FACTORY_GPU", "1") == "1"

# (task, model) -> (module, class, default kwargs). Claude's params override the defaults.
ESTIMATORS = {
    ("classification", "logreg"): ("sklearn.linear_model", "LogisticRegression", {"max_iter": 2000}),
    ("classification", "random_forest"): ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 300, "n_jobs": -1, "random_state": SEED}),
    ("classification", "lightgbm"): ("lightgbm", "LGBMClassifier", {"random_state": SEED, "verbose": -1}),
    ("classification", "xgboost"): ("xgboost", "XGBClassifier", {"random_state": SEED, "n_jobs": -1, "tree_method": "hist"}),
    ("classification", "mlp"): ("sklearn.neural_network", "MLPClassifier", {"max_iter": 500, "early_stopping": True, "random_state": SEED}),
    ("classification", "catboost"): ("catboost", "CatBoostClassifier", {"random_seed": SEED, "verbose": 0, "thread_count": -1}),
    ("classification", "tabicl"): ("tabicl", "TabICLClassifier", {}),
    ("regression", "ridge"): ("sklearn.linear_model", "Ridge", {}),
    ("regression", "random_forest"): ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 300, "n_jobs": -1, "random_state": SEED}),
    ("regression", "lightgbm"): ("lightgbm", "LGBMRegressor", {"random_state": SEED, "verbose": -1}),
    ("regression", "xgboost"): ("xgboost", "XGBRegressor", {"random_state": SEED, "n_jobs": -1, "tree_method": "hist"}),
    ("regression", "mlp"): ("sklearn.neural_network", "MLPRegressor", {"max_iter": 500, "early_stopping": True, "random_state": SEED}),
    ("regression", "catboost"): ("catboost", "CatBoostRegressor", {"random_seed": SEED, "verbose": 0, "thread_count": -1}),
    ("regression", "tabicl"): ("tabicl", "TabICLRegressor", {}),
}
# Models that only run on the GPU image. train_candidate and predict_holdout forward these to their _gpu twins,
# but the agent should call train_candidate_gpu / predict_holdout_gpu directly to skip the idle CPU hop.
GPU_MODELS = {"tabicl"}
SCALED_TARGET_MODELS = {"mlp"}  # regression target standardized inside the pipeline
MODEL_MENU = {
    t: [m for (tt, m) in ESTIMATORS if tt == t] for t in ("classification", "regression")
}
if not GPU_ENABLED:
    ESTIMATORS = {k: v for k, v in ESTIMATORS.items() if k[1] != "tabicl"}
    MODEL_MENU = {t: [x for x in ms if x != "tabicl"] for t, ms in MODEL_MENU.items()}

NAME_RE = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}"  # no leading "-" or "."  # candidate and dataset names become file names
MAX_ROUNDS_PARAM = 5000  # cap on n_estimators / iterations / max_iter (resource guard)
_ITER_KEYS = ("n_estimators", "iterations", "max_iter")
SIZE_CAPS = {"num_leaves": 4096, "max_leaves": 4096, "max_depth": 64, "depth": 16, "max_bin": 1024,
             "batch_size": 4096, "max_ctr_complexity": 8, "one_hot_max_size": 255}
# Constructor kwargs the LLM, the search space or stored memory may set, per model. Anything else
# (e.g. LightGBM machines/tree_learner, CatBoost train_dir, callbacks, file paths) is rejected.
ALLOWED_PARAMS = {
    "logreg": {"C", "class_weight", "penalty", "solver", "l1_ratio", "max_iter", "fit_intercept"},
    "ridge": {"alpha", "fit_intercept"},
    "random_forest": {"n_estimators", "max_depth", "max_features", "min_samples_leaf", "min_samples_split",
                      "class_weight", "bootstrap", "criterion", "max_samples"},
    "lightgbm": {"learning_rate", "num_leaves", "min_child_samples", "colsample_bytree", "subsample",
                 "subsample_freq", "reg_lambda", "reg_alpha", "n_estimators", "max_depth", "class_weight",
                 "extra_trees", "scale_pos_weight", "min_split_gain", "max_bin"},
    "xgboost": {"learning_rate", "max_depth", "min_child_weight", "colsample_bytree", "colsample_bylevel",
                "colsample_bynode", "subsample", "reg_lambda", "reg_alpha", "n_estimators", "grow_policy",
                "max_leaves", "gamma", "scale_pos_weight", "max_bin"},
    "catboost": {"depth", "learning_rate", "l2_leaf_reg", "iterations", "random_strength", "auto_class_weights",
                 "boosting_type", "bootstrap_type", "colsample_bylevel", "grow_policy", "leaf_estimation_iterations",
                 "max_bin", "max_ctr_complexity", "model_size_reg", "one_hot_max_size", "subsample"},
    "mlp": {"hidden_layer_sizes", "alpha", "learning_rate_init", "activation", "batch_size", "max_iter",
            "early_stopping"},
    "tabicl": {"n_estimators"},
}


def check_name(name) -> str:
    """Reject names that could escape a directory when used in a file path."""
    import re

    if not isinstance(name, str) or not re.fullmatch(NAME_RE, name) or name in (".", ".."):
        raise ValueError(f"bad name {name!r}: 1-64 of letters, digits, _ . -, not starting with '.'")
    return name


def check_params(model: str, params: dict) -> dict:
    """Allowlist constructor kwargs per model and bound iteration counts."""
    bad = sorted(set(params) - ALLOWED_PARAMS.get(model, set()))
    if bad:
        raise ValueError(f"params {bad} are not allowed for {model}; allowed: {sorted(ALLOWED_PARAMS.get(model, ()))}")
    for k in _ITER_KEYS:
        v = params.get(k)
        if v is not None and (not isinstance(v, (int, float)) or v > MAX_ROUNDS_PARAM):
            raise ValueError(f"{k}={v!r} exceeds the cap of {MAX_ROUNDS_PARAM}")
    for k, cap in SIZE_CAPS.items():  # memory guard: an OOM kill would skip _train's error handling
        v = params.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= cap):
            raise ValueError(f"{k}={v!r} must be a number in (0, {cap}]")
    h = params.get("hidden_layer_sizes")
    if h is not None and (not isinstance(h, (list, tuple)) or not 1 <= len(h) <= 4
                          or not all(isinstance(w, int) and 0 < w <= 1024 for w in h)):
        raise ValueError(f"hidden_layer_sizes={h!r} must be 1-4 layers of 1-1024 units")
    return params


app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgomp1")  # lightgbm needs OpenMP
    # pandas/pyarrow must match the local venv: parquet is written locally, read here
    # ponytail: ML libs unpinned, pin to whatever the first green smoke test resolves
    .uv_pip_install("pandas==3.0.6", "pyarrow==25.0.1", "scikit-learn", "lightgbm", "xgboost", "catboost", "joblib")
    .env({"ML_FACTORY_GPU": "1" if GPU_ENABLED else "0"})  # same menu inside containers
)
# ponytail: tabicl pulls its checkpoint from Hugging Face on every cold start, cache it on the volume if that hurts
gpu_image = image.uv_pip_install("torch", "tabicl")
GPU_KW = dict(image=gpu_image, gpu="L4", memory=16384) if GPU_ENABLED else dict(image=image, memory=8192)


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
    import uuid

    return upload_frame(df, uuid.uuid4().hex[:12])


def upload_frame(df, name: str) -> str:
    """Write df as /datasets/<name>.parquet on the Modal volume (e.g. a holdout split for
    predict_holdout). Overwrites an existing file of that name. Returns the volume path."""
    import os
    import re
    import tempfile

    check_name(name)
    df = df.copy()
    for c in df.columns:  # mixed-type object columns break pyarrow
        if df[c].dtype == object:
            df[c] = df[c].where(df[c].isna(), df[c].astype(str))
    remote = f"/datasets/{name}.parquet"
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


def _subsample_index(y, task, n, seed=SEED, n_bins=10):
    """Pick exactly n row positions of y (sorted ascending, so time order survives),
    stratified by class for classification or by n_bins quantile bins of y for regression.
    NESTED: every row gets a fixed priority from default_rng(seed), so for one seed the
    n=500 sample is a subset of the n=1500 one and fidelity levels share rows. Priority
    is (rank of the row's random key within its stratum + 0.5) / stratum size, which
    interleaves strata proportionally; taking the n lowest priorities keeps every
    stratum within one row of its proportional share. Returns all rows if n >= len(y)."""
    import numpy as np

    y = np.asarray(y)
    N = len(y)
    if n >= N:
        return np.arange(N)
    key = np.random.default_rng(seed).random(N)
    if task == "classification":
        strata = np.unique(y.astype(str), return_inverse=True)[1]
    else:
        strata = np.argsort(np.argsort(y, kind="stable"), kind="stable") * min(n_bins, N) // N
    prio = np.empty(N)
    for s in np.unique(strata):
        rows = np.flatnonzero(strata == s)
        prio[rows[np.argsort(key[rows])]] = (np.arange(len(rows)) + 0.5) / len(rows)
    return np.sort(np.lexsort((key, prio))[:n])


def _load_xy(spec):
    """Read the spec's dataset from the volume and split it into (X, y, classes).
    Drops rows with a missing target, sorts by time_column when one is given (required
    for timeseries CV), and removes drop_columns. If spec["train_rows"] is below the row
    count, keeps a nested stratified subsample of that many rows (see _subsample_index,
    seeded by spec["subsample_seed"], default SEED). For classification, y is
    label-encoded and `classes` holds the original labels in encoded order; for
    regression, y is float and `classes` is None."""
    import pandas as pd

    vol.reload()
    df = pd.read_parquet(f"{DATA_DIR}{spec['dataset_path']}")
    target = spec["target"]
    df = df.dropna(subset=[target])
    if spec.get("time_column") and spec["time_column"] in df.columns:
        df = df.sort_values(spec["time_column"]).reset_index(drop=True)
    if spec.get("train_rows") and spec["train_rows"] < len(df):
        idx = _subsample_index(df[target].values, spec["task"], int(spec["train_rows"]), spec.get("subsample_seed", SEED))
        df = df.iloc[idx].reset_index(drop=True)
    y = df[target]
    X = _clean(df.drop(columns=[target, *spec.get("drop_columns", [])], errors="ignore"))
    classes = None
    if spec["task"] == "classification":
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        classes = le.classes_.tolist()
    else:
        y = y.astype(float).values
    return X, y, classes


def _build_pipeline(X, spec):
    """Build an unfitted preprocessing + estimator Pipeline for one candidate.
    Numeric columns are imputed (median by default) and scaled by default only for the
    scale-sensitive models (logreg, ridge, mlp). Categorical columns are imputed with
    the most frequent value, then one-hot encoded (capped at max_categories, default 20)
    or ordinal encoded (the default for catboost and tabicl). The estimator gets the ESTIMATORS defaults with the spec's
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
    if prep.get("categorical", "ordinal" if model in ("catboost", "tabicl") else "onehot") == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        enc = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=prep.get("max_categories", 20))

    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", enc)]), cat),
    ])
    mod, cls, defaults = ESTIMATORS[(task, model)]
    check_name(spec.get("name", "candidate"))
    params = check_params(model, dict(spec.get("params") or {}))
    est = getattr(importlib.import_module(mod), cls)(**{**defaults, **params})
    if task == "regression" and model in SCALED_TARGET_MODELS:  # MLP diverges on raw targets (e.g. prices)
        from sklearn.compose import TransformedTargetRegressor

        est = TransformedTargetRegressor(regressor=est, transformer=StandardScaler())
    return Pipeline([("prep", pre), ("model", est)])


class PurgedKFold:
    """K-fold over contiguous blocks in row order (rows must already be time sorted), with
    no shuffling. For each test block, `gap` rows right before it and `embargo` rows right
    after it are purged from training so labels that overlap in time cannot leak.
    embargo=None means 1% of the rows. Works as a cv= argument to cross_validate."""

    def __init__(self, n_splits=5, gap=0, embargo=None):
        self.n_splits, self.gap, self.embargo = n_splits, gap, embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        import numpy as np

        n = len(X)
        emb = int(0.01 * n) if self.embargo is None else self.embargo
        idx = np.arange(n)
        for test in np.array_split(idx, self.n_splits):
            a, b = test[0], test[-1] + 1
            yield idx[(idx < a - self.gap) | (idx >= b + emb)], test


def _cv(spec):
    """Pick the CV splitter from spec["cv"]:
      "walk_forward": TimeSeriesSplit with spec["gap"] rows (default 0) between train and test;
      "timeseries":   alias of walk_forward with gap 0;
      "purged":       PurgedKFold with spec["gap"] and spec["embargo"] (default 1% of rows);
      anything else:  shuffled, seeded StratifiedKFold (classification) or KFold (regression),
                      or their Repeated* versions when spec["repeats"] > 1 (e.g. 10x10 CV).
    The time-ordered schemes do not shuffle, so rows must already be sorted (time_column)."""
    from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit

    k, cv = spec.get("cv_folds", 5), spec.get("cv")
    if cv in ("timeseries", "walk_forward"):
        return TimeSeriesSplit(n_splits=k, gap=spec.get("gap", 0) if cv == "walk_forward" else 0)
    if cv == "purged":
        return PurgedKFold(k, spec.get("gap", 0), spec.get("embargo"))
    clf = spec["task"] == "classification"
    if spec.get("repeats", 1) > 1:
        return (RepeatedStratifiedKFold if clf else RepeatedKFold)(n_splits=k, n_repeats=spec["repeats"], random_state=SEED)
    if clf:
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


def _predict_ms(pipe, X, n=50):
    """Median wall time in ms of pipe.predict on one row, over the first n rows of X."""
    import statistics
    import time

    times = []
    for i in range(min(n, len(X))):
        row = X.iloc[[i]]
        t = time.perf_counter()
        pipe.predict(row)
        times.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(times), 3)


def _augmented_folds(n_base, n_extra, splits):
    """(train_idx, test_idx) pairs over the concatenation [base rows, extra rows]: each base
    split keeps its test rows and adds every extra row (positions n_base..n_base+n_extra-1)
    to its train rows, so extra rows are fit on but never scored."""
    import numpy as np

    extra = np.arange(n_base, n_base + n_extra)
    return [(np.concatenate([np.asarray(tr), extra]), np.asarray(te)) for tr, te in splits]


def _load_extra(spec, X, classes):
    """Read spec["extra_train_path"] (same columns as the dataset, target included) and prepare
    it like _load_xy: target dropna, drop_columns, _clean, columns aligned to X. train_rows
    never applies here. For classification the labels are encoded with the base `classes`, and
    rows whose label the base never has are dropped, since they could never be scored.
    Returns (X_extra, y_extra)."""
    import pandas as pd

    target = spec["target"]
    df = pd.read_parquet(f"{DATA_DIR}{spec['extra_train_path']}").dropna(subset=[target])
    if classes is not None:
        code = df[target].astype(str).map({c: i for i, c in enumerate(classes)})
        df, y = df[code.notna()], code[code.notna()].astype(int).values
    else:
        y = df[target].astype(float).values
    return _clean(df.drop(columns=[target, *spec.get("drop_columns", [])], errors="ignore")).reindex(columns=X.columns), y


def _train(spec):
    """Body of train_candidate / train_candidate_gpu. Never raises.
    Per metric: mean, std, train_mean, plus folds (per-fold validation scores, sign
    corrected). Top level: fold_fit_seconds (per-fold fit time) and, only when
    spec["measure_latency"], predict_ms (median single-row predict latency of the
    last fold's fitted pipeline). With spec["extra_train_path"], those rows join every
    training fold but no test fold (see _augmented_folds), so the scores stay on the base
    rows and pair fold for fold with the same spec without it; adds "extra_rows"."""
    import time
    import traceback

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_validate

    t0 = time.time()
    try:
        X, y, classes = _load_xy(spec)
        scoring = _scoring(spec["task"], len(classes) if classes else 0)
        latency = bool(spec.get("measure_latency"))
        cv, n_base = _cv(spec), len(X)
        if spec.get("extra_train_path"):
            Xe, ye = _load_extra(spec, X, classes)
            cv = _augmented_folds(n_base, len(Xe), cv.split(X, y))
            X, y = pd.concat([X, Xe], ignore_index=True), np.concatenate([y, ye])
        res = cross_validate(
            _build_pipeline(X, spec), X, y, cv=cv, scoring=scoring,
            return_train_score=True, error_score="raise", return_estimator=latency,
        )
        metrics = {}
        for name, scorer in scoring.items():
            sign = -1 if scorer.startswith("neg_") else 1
            val, tr = sign * res[f"test_{name}"], sign * res[f"train_{name}"]
            metrics[name] = {"mean": round(float(val.mean()), 5), "std": round(float(val.std()), 5),
                             "train_mean": round(float(tr.mean()), 5), "folds": [round(float(v), 5) for v in val]}
        out = {"name": spec["name"], "ok": True, "metrics": metrics,
               "n_rows": n_base, "fit_seconds": round(time.time() - t0, 1),
               "fold_fit_seconds": [round(float(t), 3) for t in res["fit_time"]]}
        if spec.get("extra_train_path"):
            out["extra_rows"] = len(X) - n_base
        if latency:
            out["predict_ms"] = _predict_ms(res["estimator"][-1], X)
        return out
    except Exception as e:
        return {"name": spec.get("name"), "ok": False, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1500:]}


@app.function(image=image, volumes={DATA_DIR: vol}, cpu=4, memory=8192, timeout=1200)
def train_candidate(spec: dict) -> dict:
    """Cross-validate ONE candidate. Never raises: errors come back so the agent can react.
    GPU_MODELS specs are forwarded to train_candidate_gpu (call that directly to skip this hop)."""
    if spec.get("model") in GPU_MODELS:
        return train_candidate_gpu.remote(spec)
    return _train(spec)


@app.function(volumes={DATA_DIR: vol}, cpu=4, timeout=1800, **GPU_KW)
def train_candidate_gpu(spec: dict) -> dict:
    """Same as train_candidate on an L4 GPU with tabicl + torch installed. Route tabicl specs here."""
    return _train(spec)


@app.function(image=image, volumes={DATA_DIR: vol}, cpu=4, memory=8192, timeout=1800)
def fit_final(spec: dict, run_id: str) -> dict:
    """Fit the winner on all data, save it to the volume, return path + top features."""
    return _fit_final(spec, run_id)


@app.function(volumes={DATA_DIR: vol}, cpu=4, timeout=1800, **GPU_KW)
def fit_final_gpu(spec: dict, run_id: str) -> dict:
    """fit_final for foundation models (tabicl) that need the GPU image."""
    return _fit_final(spec, run_id)


def _fit_final(spec, run_id):
    """Body of fit_final / fit_final_gpu."""
    import os

    import joblib

    X, y, classes = _load_xy(spec)
    pipe = _build_pipeline(X, spec)
    pipe.fit(X, y)
    out_dir = f"{DATA_DIR}/models/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{spec['name']}.joblib"
    joblib.dump({"pipeline": pipe, "classes": classes, "spec": spec}, path)
    vol.commit()
    return {"model_path": path.removeprefix(DATA_DIR), "n_rows": len(X), "top_features": _top_features(pipe)}


def _predict_holdout(spec, holdout_path):
    """Fit on the spec's dataset (train_rows respected) and predict the holdout parquet at
    holdout_path on the volume. The holdout goes through the same target dropna,
    drop_columns and _clean as training, and its columns are aligned to the training ones.
    y_true uses the training label encoding (-1 for a label never seen in training)."""
    import pandas as pd

    X, y, classes = _load_xy(spec)
    pipe = _build_pipeline(X, spec)
    pipe.fit(X, y)
    target = spec["target"]
    hd = pd.read_parquet(f"{DATA_DIR}{holdout_path}").dropna(subset=[target])
    Xh = _clean(hd.drop(columns=[target, *spec.get("drop_columns", [])], errors="ignore")).reindex(columns=X.columns)
    pred = pipe.predict(Xh)
    proba = None
    if classes is not None:
        code = {c: i for i, c in enumerate(classes)}
        y_true = [code.get(v, -1) for v in hd[target].astype(str)]
        pred = [int(p) for p in pred]
        if len(classes) == 2:
            proba = [round(float(p), 6) for p in pipe.predict_proba(Xh)[:, 1]]
    else:
        y_true = hd[target].astype(float).tolist()
        pred = [float(p) for p in pred]
    return {"name": spec["name"], "y_true": y_true, "pred": pred, "proba": proba, "classes": classes}


@app.function(image=image, volumes={DATA_DIR: vol}, cpu=4, memory=8192, timeout=1800)
def predict_holdout(spec: dict, holdout_path: str) -> dict:
    """Fit the spec, predict a holdout already on the volume (see upload_frame). Returns
    {"name", "y_true", "pred", "proba" (positive-class prob for binary, else None), "classes"};
    for classification y_true and pred are codes into classes. GPU_MODELS go to predict_holdout_gpu."""
    if spec.get("model") in GPU_MODELS:
        return predict_holdout_gpu.remote(spec, holdout_path)
    return _predict_holdout(spec, holdout_path)


@app.function(volumes={DATA_DIR: vol}, cpu=4, timeout=1800, **GPU_KW)
def predict_holdout_gpu(spec: dict, holdout_path: str) -> dict:
    """predict_holdout on the GPU image, for tabicl."""
    return _predict_holdout(spec, holdout_path)


@app.local_entrypoint()
def smoke(csv: str, target: str, task: str = "classification"):
    """Every model in the menu with defaults, 3-fold. If this works, the backend is done."""
    path = upload_dataset(load_local(csv))
    base = {"dataset_path": path, "target": target, "task": task, "cv_folds": 3}
    specs = [{**base, "name": m, "model": m} for m in MODEL_MENU[task]]
    for r in train_candidate.map(specs):
        print(r["name"], "→", {k: v["mean"] for k, v in r["metrics"].items()} if r["ok"] else r["error"])
