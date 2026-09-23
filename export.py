"""Export a recommended candidate spec as artifacts the end user owns:
a standalone training script, a model card, and params.json."""
import json
import os
import pprint
from string import Template

import html
import re

from modal_train import ESTIMATORS, SCALED_TARGET_MODELS, SEED, check_name

SCALED_MODELS = ("logreg", "ridge", "mlp")
DEFAULT_METRIC = {"classification": "roc_auc", "regression": "rmse"}
SKLEARN_PKG = "scikit-learn"
PIP_NAMES = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}

SCRIPT = Template('''\
"""Train your own $cls model ($task), exported from candidate "$name".

Purpose: $purpose

Model: $module.$cls
Params:
$params_doc

Install:
    pip install $pip

Run:
    python train_$name.py data.csv --target $target [--out model.joblib] [--cv $cv]

Prints the cross-validated $metric, fits on all rows and saves
{"pipeline", "classes", "spec"} with joblib.
"""
import argparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler,
)
from $module import $cls

SEED = $seed
TASK = $task_r
TARGET = $target_r
DROP_COLUMNS = $drop_r
TIME_COLUMN = $time_r
CV_MODE = $cv_mode_r
PRIMARY_METRIC = $metric_r
PARAMS = $params_r
SPEC = $spec_r
SCORERS = {
    "accuracy": "accuracy", "f1_macro": "f1_macro", "roc_auc": "roc_auc",
    "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def load(path: str) -> pd.DataFrame:
    """Read CSV or Parquet; parse date-looking text columns as datetimes."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    for c in df.columns:
        if not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
            continue
        sample = df[c].dropna().astype(str).head(200)
        if len(sample) == 0 or sample.str.len().min() < 6:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.9:
                df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
        except (ValueError, TypeError):
            pass
    return df


def clean(X: pd.DataFrame) -> pd.DataFrame:
    """Bool to int, datetime to year/month/dow, other text to object."""
    X = X.copy()
    for c in list(X.columns):
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(int)
        elif pd.api.types.is_datetime64_any_dtype(s):
            X[f"{c}_year"] = s.dt.year
            X[f"{c}_month"] = s.dt.month
            X[f"{c}_dow"] = s.dt.dayofweek
            X = X.drop(columns=c)
        elif not pd.api.types.is_numeric_dtype(s):
            X[c] = s.where(s.isna(), s.astype(str)).astype(object)
    return X


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Preprocessing plus the recommended estimator, unfitted."""
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    num_steps = [("impute", SimpleImputer(strategy=$num_impute_r))]
$scale_line    enc = $encoder
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", enc),
        ]), cat),
    ])
    return Pipeline([("prep", pre), ("model", $model_expr)])


def load_xy(path: str, target: str) -> tuple:
    """Return (X, y, classes) with the target label encoded if needed."""
    df = load(path).dropna(subset=[target])
    if TIME_COLUMN and TIME_COLUMN in df.columns:
        df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    X = clean(df.drop(columns=[target, *DROP_COLUMNS], errors="ignore"))
    if TASK == "classification":
        le = LabelEncoder()
        y = le.fit_transform(df[target].astype(str))
        return X, y, le.classes_.tolist()
    return X, df[target].astype(float).values, None


def splitter(k: int):
    """Same CV scheme the agent used to score this candidate."""
    if CV_MODE == "timeseries":
        return TimeSeriesSplit(n_splits=k)
    if TASK == "classification":
        return StratifiedKFold(k, shuffle=True, random_state=SEED)
    return KFold(k, shuffle=True, random_state=SEED)


def main() -> None:
    """Cross-validate, fit on all rows, save the model."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="CSV or Parquet file")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--out", default="model.joblib")
    ap.add_argument("--cv", type=int, default=$cv)
    args = ap.parse_args()

    X, y, classes = load_xy(args.data, args.target)
    scorer = SCORERS[PRIMARY_METRIC]
    if scorer == "roc_auc" and classes and len(classes) > 2:
        scorer = "roc_auc_ovr"
    if args.cv > 1:
        scores = cross_val_score(
            build_pipeline(X), X, y, cv=splitter(args.cv), scoring=scorer,
        )
        if scorer.startswith("neg_"):
            scores = -scores
        print(f"{PRIMARY_METRIC}: mean {scores.mean():.5f} "
              f"std {scores.std():.5f} ({args.cv}-fold, {len(X)} rows)")
    pipe = build_pipeline(X).fit(X, y)
    joblib.dump({"pipeline": pipe, "classes": classes, "spec": SPEC}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
''')


def _no_dashes(text: str) -> str:
    """Replace em and en dashes with plain hyphens."""
    return text.replace("\u2014", "-").replace("\u2013", "-")


def _estimator(spec: dict) -> tuple[str, str, dict]:
    """(module, class, merged params) for the spec; ValueError if unknown."""
    key = (spec["task"], spec["model"])
    if key not in ESTIMATORS:
        raise ValueError(f"unknown model {key}")
    mod, cls, defaults = ESTIMATORS[key]
    return mod, cls, {**defaults, **(spec.get("params") or {})}


def _metric(spec: dict) -> str:
    return spec.get("primary_metric") or DEFAULT_METRIC[spec["task"]]


def render_train_script(spec: dict, purpose: str | None = None) -> str:
    """Render a standalone training script for one candidate spec."""
    mod, cls, params = _estimator(spec)
    prep = spec.get("preprocessing") or {}
    scale = prep.get("scale", spec["model"] in SCALED_MODELS)
    if prep.get("categorical", "onehot") == "ordinal":
        encoder = ('OrdinalEncoder(handle_unknown="use_encoded_value", '
                   'unknown_value=-1)')
    else:
        encoder = ('OneHotEncoder(handle_unknown="infrequent_if_exist", '
                   f'max_categories={int(prep.get("max_categories", 20))})')
    lib = mod.split(".")[0]
    pip = " ".join(p for p in ("pandas", SKLEARN_PKG, PIP_NAMES.get(lib)) if p)
    src = SCRIPT.substitute(
        name=spec["name"], task=_doc(spec["task"]), module=mod, cls=cls, pip=pip,
        purpose=_doc(purpose or "not stated"), target=_doc(spec["target"]), cv=int(spec.get("cv_folds", 5)),
        metric=_doc(_metric(spec)), seed=SEED,
        params_doc="\n".join(_doc(f"    {k} = {v!r}") for k, v in sorted(params.items())),
        task_r=repr(spec["task"]), target_r=repr(spec["target"]),
        drop_r=repr(list(spec.get("drop_columns") or [])),
        time_r=repr(spec.get("time_column")), cv_mode_r=repr(spec.get("cv")),
        metric_r=repr(_metric(spec)), params_r=repr(params),
        spec_r=pprint.pformat(spec, indent=4, width=70),
        num_impute_r=repr(prep.get("numeric_impute", "median")),
        scale_line='    num_steps.append(("scale", StandardScaler()))\n' if scale else "",
        encoder=encoder,
        model_expr=(f"TransformedTargetRegressor(regressor={cls}(**PARAMS), transformer=StandardScaler())"
                    if spec["task"] == "regression" and spec["model"] in SCALED_TARGET_MODELS else f"{cls}(**PARAMS)"),
    )
    return _no_dashes(src)


def _fmt(v: object) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _flatten(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Nested dict to sorted (dotted key, value) rows."""
    rows = []
    for k, v in sorted(d.items()):
        key = f"{prefix}{k}"
        rows += _flatten(v, key + ".") if isinstance(v, dict) else [(key, v)]
    return rows


def _table(rows: list[tuple[str, object]], head: tuple[str, str]) -> list[str]:
    return [f"| {head[0]} | {head[1]} |", "|---|---|",
            *(f"| {_plain(k)} | {_plain(_fmt(v)).replace('|', '/')} |" for k, v in rows)]


def _doc(text) -> str:
    """Untrusted text inside the generated script's docstring: no double quotes and no
    backslashes, so nothing can close the docstring and inject code."""
    return str(text).replace("\\", "/").replace('"', "'")


def _plain(text) -> str:
    """Untrusted text (column names, findings) as one escaped line: no headings, no HTML."""
    return html.escape(re.sub(r"\s+", " ", str(text)).strip()[:300], quote=False)


def render_model_card(spec: dict, metrics: dict, purpose: str | None,
                      caveats: list[str], extra: dict | None) -> str:
    """Render a markdown model card for the exported candidate."""
    mod, cls, params = _estimator(spec)
    extra = extra or {}
    name, metric = spec["name"], _metric(spec)
    out = [f"# Model card: {name}", "", "## Purpose", "", _plain(purpose or "Not stated."), "",
           "## Model", "", f"`{mod}.{cls}` for {_plain(spec['task'])}, target "
           f"`{_plain(spec['target']).replace('`', "'")}`, primary metric `{_plain(metric)}`.", "",
           "### Hyperparameters", "", *_table(sorted(params.items()), ("param", "value")),
           "", "## Metrics", ""]
    if "cv_mean" in metrics:
        line = f"CV {metric}: {_fmt(metrics['cv_mean'])}"
        if "cv_std" in metrics:
            line += f" +- {_fmt(metrics['cv_std'])}"
        if "ci_low" in metrics and "ci_high" in metrics:
            line += f" (CI {_fmt(metrics['ci_low'])} to {_fmt(metrics['ci_high'])})"
        out += [line, ""]
    out += [*_table(_flatten(metrics), ("metric", "value")), ""]
    if "calibration" in extra:
        cal = extra["calibration"]
        out += ["## Calibration", "", *(_table(_flatten(cal), ("stat", "value"))
                                        if isinstance(cal, dict) else [str(cal)]), ""]
    if "threshold" in extra:
        out += ["## Operating threshold", "",
                f"Predict positive when probability >= {_fmt(extra['threshold'])}.", ""]
    if "complexity" in extra:
        out += ["## Complexity (Big-O)", "", str(extra["complexity"]), ""]
    out += ["## Caveats", "", *([f"- {_plain(c)}" for c in caveats] or ["- None recorded."]), "",
            "## How to retrain", "",
            f"    python train_{name}.py data.csv --target {spec['target']} "
            f"--out model.joblib --cv {int(spec.get('cv_folds', 5))}", "",
            "The script prints the cross-validated metric, then fits on all rows.", "",
            "## Checks to rerun on new data", "",
            "- Drift: compare feature distributions and the target rate against the "
            "training data (e.g. PSI or KS per column).",
            "- Leakage: confirm no column is recorded after the outcome; drop any "
            "feature that alone scores near-perfectly.",
            f"- Drop columns used here: {', '.join(spec.get('drop_columns') or []) or 'none'}.",
            "- Re-check the metric and calibration on a recent holdout before deploying.",
            ""]
    return _no_dashes("\n".join(out))


def export_bundle(spec: dict, metrics: dict, out_dir: str, purpose: str | None = None,
                  caveats: tuple | list = (), extra: dict | None = None) -> dict[str, str]:
    """Write script, params.json and MODEL_CARD.md into out_dir; return paths."""
    check_name(spec["name"])
    os.makedirs(out_dir, exist_ok=True)
    paths = {"script": os.path.join(out_dir, f"train_{spec['name']}.py"),
             "params": os.path.join(out_dir, "params.json"),
             "card": os.path.join(out_dir, "MODEL_CARD.md")}
    body = {
        "script": render_train_script(spec, purpose),
        "params": json.dumps({"spec": spec, "metrics": metrics}, indent=2,
                             sort_keys=True, default=str) + "\n",
        "card": render_model_card(spec, metrics, purpose, list(caveats), extra),
    }
    for key, path in paths.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(body[key])
    return paths
