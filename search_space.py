"""Candidate pipeline generator: zero-shot portfolio + Sobol draws, ranked by a zero-cost prior.

Specs match modal_train.py: {"name", "model", "params", "preprocessing"}, plus "task", "family"
and "prior" here. Params use each library's native sklearn-API names.
"""
import json
import math

from scipy.stats import qmc

FAMILIES = {
    "classification": {"logreg", "random_forest", "lightgbm", "xgboost", "catboost", "mlp", "tabicl"},
    "regression": {"ridge", "random_forest", "lightgbm", "xgboost", "catboost", "mlp", "tabicl"},
}
GBDT = ("lightgbm", "xgboost", "catboost")
LINEAR = ("logreg", "ridge")
PREFIX = {"logreg": "lr", "ridge": "ridge", "random_forest": "rf", "lightgbm": "lgbm", "xgboost": "xgb",
          "catboost": "cat", "mlp": "mlp", "tabicl": "tabicl"}
MIN_SHARE = 0.05  # every allowed family keeps at least 5% of the prior mass (diversity)
PORTFOLIO_BONUS = 1.25  # benchmark-selected configs outrank random draws of the same family

TREE_PREP = {"categorical": ("choice", ["ordinal", "onehot"]), "max_categories": ("choice", [10, 20, 50])}
SCALED_PREP = {"scale": ("fixed", True), "numeric_impute": ("choice", ["median", "mean"]),
               "max_categories": ("choice", [10, 20, 50])}

# Search boxes: (kind, lo, hi) with kind in log | lin | int | logint, or ("choice", [...]) / ("fixed", v).
BOXES = {
    "lightgbm": {"learning_rate": ("log", 0.01, 0.3), "num_leaves": ("logint", 8, 256),
                 "min_child_samples": ("int", 5, 100), "colsample_bytree": ("lin", 0.4, 1.0),
                 "subsample": ("lin", 0.5, 1.0), "subsample_freq": ("fixed", 1),
                 "reg_lambda": ("log", 1e-3, 10.0), "n_estimators": ("int", 100, 1500)},
    "xgboost": {"learning_rate": ("log", 0.01, 0.3), "max_depth": ("int", 3, 10),
                "min_child_weight": ("log", 0.1, 20.0), "colsample_bytree": ("lin", 0.4, 1.0),
                "subsample": ("lin", 0.5, 1.0), "reg_lambda": ("log", 1e-3, 10.0),
                "reg_alpha": ("log", 1e-3, 5.0), "n_estimators": ("int", 100, 1500)},
    "catboost": {"depth": ("int", 4, 10), "learning_rate": ("log", 0.01, 0.3),
                 "l2_leaf_reg": ("log", 0.5, 30.0), "iterations": ("int", 200, 2000),
                 "random_strength": ("lin", 0.0, 5.0)},
    "random_forest": {"n_estimators": ("int", 100, 800),
                      "max_depth": ("choice", [None, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40]),
                      "max_features": ("choice", ["sqrt", "log2", 0.3, 0.5, 0.7, 0.85, 1.0]),
                      "min_samples_leaf": ("logint", 1, 20)},
    "logreg": {"C": ("log", 1e-3, 1e3), "class_weight": ("choice", [None, "balanced"])},
    "ridge": {"alpha": ("log", 1e-3, 1e3)},
    "mlp": {"hidden_layer_sizes": ("choice", [[64], [128], [256], [64, 64], [128, 64], [256, 128],
                                              [128, 128, 64]]),
            "alpha": ("log", 1e-6, 1e-1), "learning_rate_init": ("log", 1e-4, 1e-2)},
    "tabicl": {"n_estimators": ("choice", [1, 2, 4, 8, 16, 32])},
}
PREP_BOXES = {"lightgbm": TREE_PREP, "xgboost": TREE_PREP, "catboost": TREE_PREP, "random_forest": TREE_PREP,
              "logreg": SCALED_PREP, "ridge": SCALED_PREP, "mlp": SCALED_PREP,
              "tabicl": {"categorical": ("fixed", "ordinal")}}  # TabICL normalises features itself
INTERPRETABLE_RF = {**BOXES["random_forest"], "max_depth": ("choice", [2, 3, 4, 5, 6])}

COMPLEXITY = {  # n rows, p features, T trees, L leaves, h hidden width; order = 1-SE simplicity rank
    "logreg": {"train": "O(n p iters)", "predict": "O(p)", "order": 0},
    "ridge": {"train": "O(n p^2 + p^3)", "predict": "O(p)", "order": 0},
    "random_forest": {"train": "O(T m n log n), m = max_features", "predict": "O(T depth)", "order": 1},
    "lightgbm": {"train": "O(T n p) histogram splits", "predict": "O(T log L)", "order": 2},
    "xgboost": {"train": "O(T n p) histogram splits", "predict": "O(T depth)", "order": 2},
    "catboost": {"train": "O(T n p) oblivious trees", "predict": "O(T depth)", "order": 2},
    "mlp": {"train": "O(epochs n sum h_i h_i+1)", "predict": "O(sum h_i h_i+1)", "order": 3},
    "tabicl": {"train": "O(1), no gradient fit (in-context)",
               "predict": "O(E (n_ctx + n_test) p^2 + E n_test n_ctx) attention over the training context",
               "order": 4},
}


def _boost_rounds(lr: float) -> int:
    # ponytail: AutoGluon trains these with early stopping; our Modal fit has none, so rounds are
    # pinned to ~40 / learning_rate (capped). Add eval-set early stopping to modal_train to lift this.
    return int(min(3000, max(100, round(40 / lr))))


# Source: https://raw.githubusercontent.com/autogluon/autogluon/master/tabular/src/autogluon/tabular/
#   configs/zeroshot/zeroshot_portfolio_2025.py, hyperparameter_portfolio_zeroshot_2025_small
# (TabArena-derived, tuned for <=10k rows / <=500 features). GBM/XGB/CAT translated to native kwargs;
# ag_args dropped; categorical-handling keys (cat_l2, cat_smooth, max_cat_to_onehot, min_data_per_group,
# enable_categorical) dropped because our pipeline encodes categoricals before the model.
_LGBM = [  # AG GBM: bagging_fraction->subsample, bagging_freq->subsample_freq, feature_fraction->
    # colsample_bytree, lambda_l1->reg_alpha, lambda_l2->reg_lambda, min_data_in_leaf->min_child_samples
    {"subsample": 0.9625293420216, "subsample_freq": 1, "extra_trees": False, "colsample_bytree": 0.6189215809382,
     "reg_alpha": 0.1641757352921, "reg_lambda": 0.6937755557881, "learning_rate": 0.0154031028561,
     "min_child_samples": 1, "num_leaves": 68},
    {"subsample": 0.7218730663234, "subsample_freq": 1, "extra_trees": False, "colsample_bytree": 0.4557131604374,
     "reg_alpha": 0.5219704038237, "reg_lambda": 0.1070959487853, "learning_rate": 0.0055891584996,
     "min_child_samples": 50, "num_leaves": 30},
    {"subsample": 0.775784726514, "subsample_freq": 1, "extra_trees": True, "colsample_bytree": 0.7732354787904,
     "reg_alpha": 0.2211002452568, "reg_lambda": 1.1318405980187, "learning_rate": 0.0090151778542,
     "min_child_samples": 4, "num_leaves": 2},
]
_XGB = [
    {"colsample_bylevel": 0.9213705632288, "colsample_bynode": 0.6443385965381, "grow_policy": "lossguide",
     "learning_rate": 0.0068171645251, "max_depth": 6, "max_leaves": 10, "min_child_weight": 0.0507304250576,
     "reg_alpha": 4.2446346389037, "reg_lambda": 1.4800570021253, "subsample": 0.9656290596647},
    {"colsample_bylevel": 0.6377491713202, "colsample_bynode": 0.9237625621103, "grow_policy": "lossguide",
     "learning_rate": 0.0112462621131, "max_depth": 10, "max_leaves": 35, "min_child_weight": 0.1403464856034,
     "reg_alpha": 3.4960653958503, "reg_lambda": 1.3062320805235, "subsample": 0.6948898835178},
]
_CAT = [
    {},  # CatBoost defaults
    {"boosting_type": "Plain", "bootstrap_type": "Bernoulli", "colsample_bylevel": 0.8771035272558, "depth": 7,
     "grow_policy": "SymmetricTree", "l2_leaf_reg": 2.0107286863021, "leaf_estimation_iterations": 2,
     "learning_rate": 0.0058424016622, "max_bin": 254, "max_ctr_complexity": 4, "model_size_reg": 0.1307400355809,
     "one_hot_max_size": 23, "subsample": 0.809527841437},
    {"boosting_type": "Plain", "bootstrap_type": "Bernoulli", "colsample_bylevel": 0.8994502668431, "depth": 6,
     "grow_policy": "Depthwise", "l2_leaf_reg": 1.8187025215896, "leaf_estimation_iterations": 7,
     "learning_rate": 0.005177304142, "max_bin": 254, "max_ctr_complexity": 4, "model_size_reg": 0.5247386875068,
     "one_hot_max_size": 53, "subsample": 0.8705228845742},
    {"boosting_type": "Plain", "bootstrap_type": "Bernoulli", "colsample_bylevel": 0.8597809376276, "depth": 8,
     "grow_policy": "Depthwise", "l2_leaf_reg": 0.3628261923976, "leaf_estimation_iterations": 5,
     "learning_rate": 0.016851077771, "max_bin": 254, "max_ctr_complexity": 4, "model_size_reg": 0.1253820547902,
     "one_hot_max_size": 20, "subsample": 0.8120271122061},
    {"boosting_type": "Plain", "bootstrap_type": "Bernoulli", "colsample_bylevel": 0.8959275863514, "depth": 4,
     "grow_policy": "SymmetricTree", "l2_leaf_reg": 0.0026915894253, "leaf_estimation_iterations": 12,
     "learning_rate": 0.0475233791203, "max_bin": 254, "max_ctr_complexity": 5, "model_size_reg": 0.1633175256924,
     "one_hot_max_size": 11, "subsample": 0.798554178926},
]
_ORD = {"categorical": "ordinal"}
PORTFOLIO_2025 = (
    [{"name": f"lgbm_p{i:02d}", "model": "lightgbm", "preprocessing": _ORD,
      "params": {**p, "n_estimators": _boost_rounds(p["learning_rate"])}} for i, p in enumerate(_LGBM)]
    + [{"name": f"xgb_p{i:02d}", "model": "xgboost", "preprocessing": _ORD,
        "params": {**p, "n_estimators": _boost_rounds(p["learning_rate"])}} for i, p in enumerate(_XGB)]
    + [{"name": f"cat_p{i:02d}", "model": "catboost", "preprocessing": _ORD,
        "params": {**p, "iterations": _boost_rounds(p["learning_rate"])} if p else {}} for i, p in enumerate(_CAT)]
    + [{"name": "tabicl_default", "model": "tabicl", "preprocessing": _ORD, "params": {}}]  # AG TABICL: defaults
)


def _sig(x: float) -> float:
    return float(f"{x:.4g}")


def _decode(spec: tuple, u: float):
    kind = spec[0]
    if kind == "fixed":
        return spec[1]
    if kind == "choice":
        return spec[1][min(int(u * len(spec[1])), len(spec[1]) - 1)]
    lo, hi = spec[1], spec[2]
    if kind == "int":
        return int(min(hi, lo + math.floor(u * (hi - lo + 1))))
    if kind in ("log", "logint"):
        x = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        return int(round(x)) if kind == "logint" else _sig(x)
    return _sig(lo + u * (hi - lo))


def sobol_configs(family: str, task: str, n: int, seed: int = 42, box: dict | None = None) -> list[dict]:
    """n scrambled-Sobol draws over the family's search box (params + preprocessing)."""
    if family not in FAMILIES[task]:
        raise ValueError(f"{family} is not a {task} family")
    if n <= 0:
        return []
    box, prep = box or BOXES[family], PREP_BOXES[family]
    dims = list(box.items()) + [(f"prep.{k}", v) for k, v in prep.items()]
    pts = qmc.Sobol(len(dims), scramble=True, seed=seed).random_base2(max(0, math.ceil(math.log2(n))))[:n]
    out = []
    for i, row in enumerate(pts):
        vals = {k: _decode(s, u) for (k, s), u in zip(dims, row)}
        out.append({"name": f"{PREFIX[family]}_s{i:04d}", "model": family, "task": task,
                    "params": {k: v for k, v in vals.items() if not k.startswith("prep.")},
                    "preprocessing": {k[5:]: v for k, v in vals.items() if k.startswith("prep.")}})
    return out


def _normalize(scores: dict) -> dict:
    """Softmax, then mix with a uniform floor so every family keeps >= MIN_SHARE."""
    top = max(scores.values())
    e = {f: math.exp(s - top) for f, s in scores.items()}
    z, k = sum(e.values()), len(e)
    return {f: MIN_SHARE + (1 - MIN_SHARE * k) * v / z for f, v in e.items()}


def family_prior(task: str, meta: dict) -> dict:
    """Family weights (sum 1) from benchmark rules of thumb; missing meta keys are neutral."""
    s = {"logreg": 0.0, "ridge": 0.0, "random_forest": 0.4, "lightgbm": 1.0, "xgboost": 0.8,
         "catboost": 1.0, "mlp": 0.0, "tabicl": 0.5}
    s = {f: v for f, v in s.items() if f in FAMILIES[task]}
    n, d = meta.get("n_rows"), meta.get("n_features")

    def add(fams, x):
        for f in fams:
            if f in s:
                s[f] += x

    # TabArena 2025: in-context foundation models (TabICL, TabPFN) lead on small tables,
    # GBDTs stay on par in the mid range and lead on large or wide data.
    if n is not None and (n > 100_000 or (d or 0) > 500):
        add(["tabicl"], -2.0)
        add(GBDT, 0.5)
    elif n is not None and n <= 10_000 and (d is None or d <= 100):
        add(["tabicl"], 2.0)
    elif n is not None and n <= 100_000:
        add(["tabicl"], 0.6)
    # High-cardinality categoricals: CatBoost's ordered target statistics handle them natively.
    if meta.get("max_cardinality", 0) > 50:
        add(["catboost"], 0.7)
    if meta.get("frac_categorical", 0) > 0.3:
        add(["catboost"], 0.3)
    # Grinsztajn 2022: trees are robust to skewed/heavy-tailed features and uninformative
    # features, while MLPs degrade on both.
    if meta.get("max_abs_skew", 0) > 2:
        add(GBDT, 0.3)
        add(["mlp"], -0.5)
        add(LINEAR, -0.3)
    if meta.get("frac_uninformative", 0) > 0.3:
        add(GBDT, 0.3)
        add(["mlp"], -0.5)
    # McElfresh 2023: GBDTs win when the target is irregular; when a linear model is almost as
    # good (small nonlinearity_gap) and data is scarce per feature, simple baselines are competitive.
    gap = meta.get("nonlinearity_gap")
    if gap is not None and gap > 0.05:
        add(GBDT, 0.5)
        add(LINEAR, -1.0)
        add(["mlp"], -0.3)
    if gap is not None and gap < 0.02 and meta.get("n_over_p", math.inf) < 20:
        add(LINEAR, 1.5)
    return _normalize(s)


def _trees(c: dict) -> int:
    p = c["params"]
    return p.get("n_estimators") or p.get("iterations") or (1000 if c["model"] == "catboost" else 0)


def plausibility(c: dict, n_rows: int) -> float:
    """Within-family factor in (0, 1]: penalise capacity far beyond what n_rows supports."""
    p, f = c["params"], 1.0
    cap = max(n_rows / 20, 2)
    if p.get("num_leaves", 0) > cap:
        f *= (cap / p["num_leaves"]) ** 0.5
    depth = p.get("depth") or (p.get("max_depth") if c["model"] == "xgboost" else None)
    if depth and 2 ** depth > max(n_rows / 10, 4):
        f *= (max(n_rows / 10, 4) / 2 ** depth) ** 0.25
    if p.get("min_child_samples", 0) > max(n_rows / 50, 5):
        f *= (max(n_rows / 50, 5) / p["min_child_samples"]) ** 0.5
    if c["model"] in GBDT and p.get("learning_rate") and _trees(c):
        prod = p["learning_rate"] * _trees(c)  # total shrinkage budget, sweet spot ~20-60
        f *= math.exp(-0.5 * abs(math.log(prod / min(60, max(20, prod)))))
    return f


def _allowed(task: str, purpose: dict, n_rows: int) -> set:
    fams = set(FAMILIES[task])
    if purpose.get("interpretable"):
        fams &= {"logreg", "ridge", "random_forest"}
    ms = purpose.get("max_predict_ms")
    if ms is not None and ms < 100:
        fams.discard("tabicl")  # attends over the whole training context per prediction batch
    if ms is not None and ms < 10:
        fams.discard("mlp")
    mins = purpose.get("max_train_minutes")
    if mins is not None and mins < 5 and n_rows > 10_000:
        fams.discard("tabicl")
    return fams


def _keep(c: dict, purpose: dict) -> bool:
    # ponytail: fixed tree-count thresholds, swap for a measured latency/cost model if one appears
    if purpose.get("interpretable") and c["model"] == "random_forest" and (c["params"].get("max_depth") or 99) > 6:
        return False
    if purpose.get("max_predict_ms") is not None and purpose["max_predict_ms"] < 10 and _trees(c) > 300:
        return False
    if purpose.get("max_train_minutes") is not None and purpose["max_train_minutes"] < 5 and _trees(c) > 1000:
        return False
    return True


def _weighted(c: dict, task: str, purpose: dict) -> dict:
    if not (purpose.get("class_weighting") and task == "classification"):
        return c
    if c["model"] in ("logreg", "random_forest", "lightgbm"):
        return {**c, "params": {**c["params"], "class_weight": "balanced"}}
    if c["model"] == "catboost":
        return {**c, "params": {**c["params"], "auto_class_weights": "Balanced"}}
    return c


def _key(c: dict) -> str:
    return json.dumps([c["model"], c["params"], c["preprocessing"]], sort_keys=True)


def rank_space(configs: list[dict]) -> list[dict]:
    """Sort by prior, highest first."""
    return sorted(configs, key=lambda c: c["prior"], reverse=True)


def build_space(task: str, n_rows: int, n_features: int, meta: dict, purpose: dict,
                budget: int = 3000, seed: int = 42) -> list[dict]:
    """Portfolio + Sobol draws split across allowed families by prior; deduplicated and ranked."""
    meta = {"n_over_p": n_rows / max(n_features, 1), **meta, "n_rows": n_rows, "n_features": n_features}
    fams = _allowed(task, purpose, n_rows)
    prior = family_prior(task, meta)
    w = _normalize({f: math.log(prior[f]) for f in fams})  # renormalise over the allowed subset

    def finish(c: dict, bonus: float = 1.0) -> dict:
        c = _weighted({**c, "task": task}, task, purpose)
        return {**c, "family": c["model"], "prior": w[c["model"]] * plausibility(c, n_rows) * bonus}

    port = [finish(c, PORTFOLIO_BONUS) for c in PORTFOLIO_2025 if c["model"] in fams]
    port = [c for c in port if _keep(c, purpose)]
    seen, target = {_key(c) for c in port}, max(0, budget - len(port))
    boxes = {f: INTERPRETABLE_RF if f == "random_forest" and purpose.get("interpretable") else BOXES[f] for f in fams}
    counts = {f: math.ceil(w[f] * target) for f in fams}
    pool: dict = {}
    for _ in range(4):  # small discrete boxes (tabicl) saturate; hand their deficit to the others
        for f in fams:
            for c in sobol_configs(f, task, counts[f], seed, boxes[f]):
                c = finish(c)
                if _keep(c, purpose) and _key(c) not in seen:
                    pool.setdefault(_key(c), c)
        deficit = target - len(pool)
        if deficit <= 0:
            break
        grow = {f: w[f] for f in fams if any(b[0] not in ("choice", "fixed") for b in boxes[f].values())}
        for f in grow:
            counts[f] += math.ceil(deficit * 1.2 * grow[f] / sum(grow.values()))
    return rank_space(port + rank_space(list(pool.values()))[:target])
