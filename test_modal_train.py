import numpy as np
import pandas as pd
from sklearn.model_selection import (KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold,
                                     TimeSeriesSplit, cross_validate)

import modal_train as mt


def _clf_y(n=5000, seed=0):
    return (np.random.default_rng(seed).random(n) < 0.12).astype(int)


def test_subsample_exact_unique():
    y = _clf_y()
    for n in (1, 37, 500, 4999):
        idx = mt._subsample_index(y, "classification", n, seed=7)
        assert len(idx) == n and len(np.unique(idx)) == n
        assert np.all(np.diff(idx) > 0) and idx.min() >= 0 and idx.max() < len(y)
    assert np.array_equal(mt._subsample_index(y, "classification", 10**6, 7), np.arange(len(y)))


def test_subsample_preserves_class_ratio():
    y = _clf_y()
    for n in (200, 500, 1500):
        idx = mt._subsample_index(y, "classification", n, seed=3)
        assert abs(y[idx].mean() - y.mean()) < 0.01, (n, y[idx].mean(), y.mean())
    y3 = np.random.default_rng(1).choice(["a", "b", "c"], size=3000, p=[0.6, 0.3, 0.1])
    idx = mt._subsample_index(y3, "classification", 400, seed=0)
    for c in "abc":
        assert abs((y3[idx] == c).mean() - (y3 == c).mean()) < 0.01


def test_subsample_nested():
    y = _clf_y()
    for task, yy in (("classification", y), ("regression", np.random.default_rng(2).lognormal(size=5000))):
        small, big = (set(mt._subsample_index(yy, task, n, seed=11)) for n in (500, 1500))
        assert small <= big, task
    a, b = (set(mt._subsample_index(y, "classification", 500, seed=s)) for s in (1, 2))
    assert a != b


def test_subsample_regression_quantiles():
    y = np.random.default_rng(4).lognormal(size=4000)
    idx = mt._subsample_index(y, "regression", 400, seed=0)
    qs = np.quantile(y, np.linspace(0.1, 0.9, 9))
    counts = np.histogram(y[idx], bins=[-np.inf, *qs, np.inf])[0]
    assert counts.min() >= 39 and counts.max() <= 41, counts


def test_purged_kfold():
    n, k, gap, emb = 1000, 5, 7, 13
    splits = list(mt.PurgedKFold(k, gap, emb).split(np.zeros(n)))
    assert len(splits) == k
    seen = np.concatenate([te for _, te in splits])
    assert np.array_equal(np.sort(seen), np.arange(n))
    for tr, te in splits:
        assert not set(tr) & set(te)
        a, b = te[0], te[-1] + 1
        assert np.array_equal(te, np.arange(a, b))
        assert not np.any((tr >= a - gap) & (tr < b + emb))
        assert len(tr) == n - (b - a) - min(gap, a) - min(emb, n - b)
    tr, te = list(mt.PurgedKFold(4).split(np.zeros(1000)))[1]
    assert not np.any((tr >= te[-1] + 1) & (tr < te[-1] + 11)) and te[-1] + 11 in tr


def test_cv_types():
    c = {"task": "classification"}
    r = {"task": "regression"}
    assert isinstance(mt._cv(c), StratifiedKFold) and isinstance(mt._cv(r), KFold)
    rs = mt._cv({**c, "cv_folds": 10, "repeats": 10})
    assert isinstance(rs, RepeatedStratifiedKFold) and rs.get_n_splits() == 100
    assert isinstance(mt._cv({**r, "repeats": 3}), RepeatedKFold)
    wf = mt._cv({**r, "cv": "walk_forward", "gap": 5})
    assert isinstance(wf, TimeSeriesSplit) and wf.gap == 5
    ts = mt._cv({**r, "cv": "timeseries", "gap": 5})
    assert isinstance(ts, TimeSeriesSplit) and ts.gap == 0
    pk = mt._cv({**r, "cv": "purged", "cv_folds": 4, "gap": 2, "embargo": 3})
    assert isinstance(pk, mt.PurgedKFold) and (pk.n_splits, pk.gap, pk.embargo) == (4, 2, 3)


def test_estimators_menu():
    for t in ("classification", "regression"):
        assert (t, "catboost") in mt.ESTIMATORS and "catboost" in mt.MODEL_MENU[t]
    assert mt.ESTIMATORS[("classification", "tabicl")][1] == "TabICLClassifier"
    assert mt.ESTIMATORS[("regression", "tabicl")][1] == "TabICLRegressor"
    assert "tabicl" in mt.GPU_MODELS


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                         "cat": rng.choice(list("abc"), size=n).astype(object)})


def test_build_pipeline_sklearn_models():
    X = mt._clean(_frame())
    rng = np.random.default_rng(0)
    for task, model, y in (("classification", "logreg", rng.integers(0, 2, len(X))),
                           ("classification", "random_forest", rng.integers(0, 3, len(X))),
                           ("regression", "ridge", rng.normal(size=len(X))),
                           ("regression", "random_forest", rng.normal(size=len(X)))):
        pipe = mt._build_pipeline(X, {"task": task, "model": model, "params": {"n_estimators": 10} if model == "random_forest" else {}})
        assert len(pipe.fit(X, y).predict(X)) == len(X)


def test_default_encoding_for_catboost():
    import sklearn.preprocessing as skp

    X = mt._clean(_frame())
    encoder = lambda pipe: pipe.named_steps["prep"].transformers[1][1].named_steps["encode"]
    saved = mt.ESTIMATORS[("classification", "catboost")]
    try:  # catboost is not installed locally, so stand a sklearn estimator in for it
        mt.ESTIMATORS[("classification", "catboost")] = ("sklearn.linear_model", "LogisticRegression", {})
        assert isinstance(encoder(mt._build_pipeline(X, {"task": "classification", "model": "catboost"})), skp.OrdinalEncoder)
        assert isinstance(encoder(mt._build_pipeline(X, {"task": "classification", "model": "logreg"})), skp.OneHotEncoder)
    finally:
        mt.ESTIMATORS[("classification", "catboost")] = saved


def test_purged_works_with_cross_validate_and_latency():
    X = mt._clean(_frame(400))
    y = np.random.default_rng(0).normal(size=len(X))
    spec = {"task": "regression", "model": "ridge", "cv": "purged", "cv_folds": 4, "gap": 3}
    res = cross_validate(mt._build_pipeline(X, spec), X, y, cv=mt._cv(spec), scoring="r2", return_estimator=True)
    assert len(res["test_score"]) == 4
    ms = mt._predict_ms(res["estimator"][-1], X)
    assert ms > 0


def _local_xy(n=300):
    df = _frame(n)
    df["y"] = np.where(df["x1"] + np.random.default_rng(5).normal(scale=0.5, size=n) > 0, "yes", "no")
    return df


def test_train_result_keys_backward_compatible():
    df = _local_xy()
    saved = mt._load_xy
    mt._load_xy = lambda spec: (mt._clean(df.drop(columns="y")), (df["y"] == "yes").astype(int).values, ["no", "yes"])
    try:
        r = mt._train({"name": "lr", "task": "classification", "model": "logreg", "cv_folds": 3, "measure_latency": True})
        assert r["ok"], r
        for k in ("name", "ok", "metrics", "fit_seconds", "n_rows", "fold_fit_seconds", "predict_ms"):
            assert k in r, k
        m = r["metrics"]["roc_auc"]
        assert {"mean", "std", "train_mean", "folds"} <= set(m) and len(m["folds"]) == 3
        assert abs(np.mean(m["folds"]) - m["mean"]) < 1e-4
        r = mt._train({"name": "lr", "task": "classification", "model": "logreg", "cv_folds": 3})
        assert "predict_ms" not in r and len(r["fold_fit_seconds"]) == 3
        assert mt._train({"name": "bad", "task": "classification", "model": "nope"})["ok"] is False
    finally:
        mt._load_xy = saved


def test_predict_holdout_local(tmp_path=None):
    import os
    import tempfile

    tmp = str(tmp_path or tempfile.mkdtemp())
    train, hold = _local_xy(300), _local_xy(80).assign(x1=lambda d: d["x1"] * -1)
    os.makedirs(f"{tmp}/datasets")
    hold.to_parquet(f"{tmp}/datasets/h.parquet", index=False)
    saved_xy, saved_dir = mt._load_xy, mt.DATA_DIR
    mt._load_xy = lambda spec: (mt._clean(train.drop(columns="y")), (train["y"] == "yes").astype(int).values, ["no", "yes"])
    mt.DATA_DIR = tmp
    try:
        r = mt._predict_holdout({"name": "lr", "task": "classification", "model": "logreg", "target": "y"}, "/datasets/h.parquet")
    finally:
        mt._load_xy, mt.DATA_DIR = saved_xy, saved_dir
    assert r["classes"] == ["no", "yes"] and len(r["y_true"]) == len(r["pred"]) == len(r["proba"]) == 80
    assert r["y_true"] == [int(v == "yes") for v in hold["y"]]
    assert set(r["pred"]) <= {0, 1} and all(0 <= p <= 1 for p in r["proba"])


def test_augmented_folds_keep_extra_rows_out_of_test():
    n_base, n_extra = 50, 17
    folds = mt._augmented_folds(n_base, n_extra, KFold(5, shuffle=True, random_state=0).split(np.zeros(n_base)))
    extra = set(range(n_base, n_base + n_extra))
    assert len(folds) == 5
    assert np.array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(n_base))  # each base row once
    for tr, te in folds:
        assert extra <= set(tr) and not extra & set(te) and not set(tr) & set(te)
        assert te.max() < n_base


def test_train_with_extra_rows_local(tmp_path=None):
    import os
    import tempfile

    tmp = str(tmp_path or tempfile.mkdtemp())
    base = _local_xy(300)
    extra = _local_xy(120).assign(y=lambda d: d["y"].where(d.index % 10 != 0, "maybe"))  # "maybe": extra only
    os.makedirs(f"{tmp}/datasets")
    extra.to_parquet(f"{tmp}/datasets/extra.parquet", index=False)
    saved_xy, saved_dir = mt._load_xy, mt.DATA_DIR
    mt._load_xy = lambda spec: (mt._clean(base.drop(columns="y")), (base["y"] == "yes").astype(int).values, ["no", "yes"])
    mt.DATA_DIR = tmp
    spec = {"name": "lr", "task": "classification", "model": "logreg", "target": "y", "cv_folds": 3}
    try:
        plain = mt._train(spec)
        aug = mt._train({**spec, "extra_train_path": "/datasets/extra.parquet"})
    finally:
        mt._load_xy, mt.DATA_DIR = saved_xy, saved_dir
    assert plain["ok"] and aug["ok"], aug
    assert set(aug) == set(plain) | {"extra_rows"} and "extra_rows" not in plain
    assert aug["extra_rows"] == 108 and aug["n_rows"] == plain["n_rows"] == 300
    assert len(aug["fold_fit_seconds"]) == 3
    for name, m in aug["metrics"].items():
        assert set(m) == set(plain["metrics"][name]) and len(m["folds"]) == 3
        assert abs(np.mean(m["folds"]) - m["mean"]) < 1e-4
    assert aug["metrics"]["roc_auc"]["folds"] != plain["metrics"]["roc_auc"]["folds"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
