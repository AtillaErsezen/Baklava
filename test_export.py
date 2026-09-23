"""Tests for export.py: generated training script, model card, bundle."""
import os
import re
import subprocess
import sys
import tempfile

from export import export_bundle, render_model_card, render_train_script
from modal_train import ESTIMATORS

ROOT = os.path.dirname(os.path.abspath(__file__))
CHURN_SPEC = {
    "name": "churn_logreg", "model": "logreg", "task": "classification",
    "target": "Churn", "params": {"C": 0.5},
    "drop_columns": ["customer_id", "refund_issued"],
}
RIDGE_SPEC = {
    "name": "houses_ridge", "model": "ridge", "task": "regression",
    "target": "SalePrice", "params": {"alpha": 2.0},
}
DASHES = ("\u2014", "\u2013")


def _run_script(spec: dict, csv: str, target: str, tmp: str) -> tuple[str, str]:
    """Render, write and run a script; return (stdout, joblib path)."""
    path = os.path.join(tmp, f"train_{spec['name']}.py")
    with open(path, "w") as f:
        f.write(render_train_script(spec, purpose="demo purpose"))
    out = os.path.join(tmp, "m.joblib")
    res = subprocess.run(
        ["uv", "run", "python", path, os.path.join(ROOT, csv), "--target", target,
         "--out", out, "--cv", "3"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout, out


def _score(stdout: str, metric: str) -> float:
    m = re.search(rf"{metric}: mean (-?[0-9.]+)", stdout)
    assert m, stdout
    return float(m.group(1))


def test_script_compiles_for_every_model() -> None:
    for task, model in ESTIMATORS:
        spec = {"name": f"{task}_{model}", "model": model, "task": task,
                "target": "y", "params": {"random_state": 1},
                "preprocessing": {"categorical": "ordinal", "scale": True}}
        src = render_train_script(spec, purpose="Predict \u2014 things")
        compile(src, f"train_{model}.py", "exec")
        assert not any(d in src for d in DASHES)


def test_script_inlines_params_and_import() -> None:
    src = render_train_script(CHURN_SPEC, purpose="flag churners")
    assert "from sklearn.linear_model import LogisticRegression" in src
    assert "'C': 0.5" in src and "'max_iter': 2000" in src
    assert "flag churners" in src and "max_categories=20" in src


def test_logreg_script_runs_and_matches_sklearn() -> None:
    import joblib
    import pandas as pd  # noqa: F401
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    from modal_train import _build_pipeline, _clean, load_local

    with tempfile.TemporaryDirectory() as tmp:
        stdout, out = _run_script(CHURN_SPEC, "data/churn.csv", "Churn", tmp)
        score = _score(stdout, "roc_auc")
        assert 0.6 < score < 0.99, stdout
        bundle = joblib.load(out)
        assert set(bundle) == {"pipeline", "classes", "spec"}
        assert bundle["classes"] == ["No", "Yes"]

    df = load_local(os.path.join(ROOT, "data/churn.csv")).dropna(subset=["Churn"])
    X = _clean(df.drop(columns=["Churn", *CHURN_SPEC["drop_columns"]]))
    y = LabelEncoder().fit_transform(df["Churn"].astype(str))
    ref = cross_val_score(_build_pipeline(X, CHURN_SPEC), X, y, scoring="roc_auc",
                          cv=StratifiedKFold(3, shuffle=True, random_state=42)).mean()
    assert abs(score - ref) < 0.02, (score, ref)


def test_ridge_script_runs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stdout, out = _run_script(RIDGE_SPEC, "data/houses.csv", "SalePrice", tmp)
        assert _score(stdout, "rmse") > 0
        assert os.path.exists(out)


def test_model_card() -> None:
    metrics = {"cv_mean": 0.84, "cv_std": 0.01, "ci_low": 0.82, "ci_high": 0.86,
               "hidden_score": 0.83}
    extra = {"calibration": {"brier": 0.12}, "threshold": 0.35,
             "complexity": "O(n d) per epoch"}
    card = render_model_card(CHURN_SPEC, metrics, "Flag churners \u2013 early",
                             ["small sample"], extra)
    assert not any(d in card for d in DASHES)
    for s in ("LogisticRegression", "| C | 0.5 |", "max_iter", "0.84", "0.35",
              "O(n d)", "small sample", "brier", "retrain", "drift", "leakage"):
        assert s.lower() in card.lower(), s


def test_export_bundle_writes_three_files() -> None:
    import json

    with tempfile.TemporaryDirectory() as tmp:
        paths = export_bundle(CHURN_SPEC, {"cv_mean": 0.8}, tmp, purpose="p")
        assert len(paths) == 3 and all(os.path.exists(p) for p in paths.values())
        assert os.path.basename(paths["script"]) == "train_churn_logreg.py"
        with open(paths["params"]) as f:
            data = json.load(f)
        assert data["spec"]["name"] == "churn_logreg" and data["metrics"]["cv_mean"] == 0.8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    sys.exit(0)
