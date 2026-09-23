"""
Statistical dataset diagnostics for the ML Factory agent.

The LLM never sees raw data, only compact findings. Add a check by writing a decorated function:

  @diagnostic("my_check", "quality", "one-line description")
  def my_check(ctx: Context, columns=None) -> dict:
      return _out(value, severity, "one short sentence")

Severity: 0 fine, 1 note, 2 should act, 3 must act.
"""
import math
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Callable
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MAX_SAMPLE = 10_000
MODEL_ROWS = 2_000     # landmarkers
SCORE_ROWS = 3_000     # single-feature scores, knn, drift
MAX_FINDINGS = 15


@dataclass(frozen=True)
class Check:
    name: str
    group: str
    description: str
    fn: Callable


DIAGNOSTICS: dict[str, Check] = {}


def diagnostic(name: str, group: str, description: str):
    """Register a check into DIAGNOSTICS."""
    def wrap(fn):
        DIAGNOSTICS[name] = Check(name, group, description, fn)
        return fn
    return wrap


@dataclass(frozen=True)
class Context:
    df: pd.DataFrame
    target: str
    task: str
    sample: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    seed: int = 42
    time_column: str | None = None

    @property
    def is_cls(self) -> bool:
        return self.task == "classification"

    @cached_property
    def yc(self) -> np.ndarray:
        """Target as class codes (classification) or floats (regression)."""
        return pd.factorize(self.y)[0] if self.is_cls else self.y.astype(float).to_numpy()

    @cached_property
    def enc(self) -> pd.DataFrame:
        """Model-ready features: bool to int, categoricals ordinal, numeric median-imputed, datetimes dropped."""
        out = {}
        for c in self.X.columns:
            s = self.X[c]
            if _is_dt(s):
                continue
            if pd.api.types.is_bool_dtype(s):
                out[c] = s.astype(float).fillna(0)
            elif pd.api.types.is_numeric_dtype(s):
                out[c] = s.astype(float).fillna(s.median() if s.notna().any() else 0)
            else:
                out[c] = pd.Series(pd.factorize(s)[0], index=s.index).astype(float)
        return pd.DataFrame(out, index=self.X.index)

    @cached_property
    def single_scores(self) -> dict[str, float]:
        """3-fold CV score of a depth-3 tree on each feature alone."""
        idx = _rows(self, SCORE_ROWS)
        tree = DecisionTreeClassifier if self.is_cls else DecisionTreeRegressor
        return {c: _cv(tree(max_depth=3, random_state=self.seed), self.enc.loc[idx, [c]], self.yc[idx], self)
                for c in self.enc.columns}


def make_context(df: pd.DataFrame, target: str, task: str, time_column: str | None = None, seed: int = 42) -> Context:
    """Build a Context with a seeded, stratified, row-order-preserving subsample."""
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in columns")
    if task not in ("classification", "regression"):
        raise ValueError("task must be classification or regression")
    labelled = df[df[target].notna()]
    sample = labelled
    if len(labelled) > MAX_SAMPLE:
        y = labelled[target]
        strat = y if task == "classification" and y.value_counts().min() >= 2 else None
        keep, _ = train_test_split(labelled.index, train_size=MAX_SAMPLE, stratify=strat, random_state=seed)
        sample = labelled.loc[sorted(keep)]
    sample = sample.reset_index(drop=True)
    return Context(df, target, task, sample, sample.drop(columns=[target]), sample[target], seed, time_column)


# ---------- helpers ----------

def _out(value, severity: int, finding: str) -> dict:
    return {"value": value, "severity": int(severity), "finding": finding}


def _sig(x):
    """Round to 3 significant figures, recursively, JSON-safe."""
    if isinstance(x, dict):
        return {str(k): _sig(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sig(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(x) else float(f"{float(x):.3g}")
    return x


def _is_dt(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def _is_cat(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or _is_dt(s))


def _cols(ctx: Context, columns=None) -> list[str]:
    return [c for c in ctx.X.columns if columns is None or c in columns]


def _num(ctx: Context, columns=None) -> list[str]:
    """Numeric, non-bool feature columns."""
    return [c for c in _cols(ctx, columns) if pd.api.types.is_numeric_dtype(ctx.X[c])
            and not pd.api.types.is_bool_dtype(ctx.X[c])]


def _cat(ctx: Context, columns=None) -> list[str]:
    return [c for c in _cols(ctx, columns) if _is_cat(ctx.X[c])]


def _encn(ctx: Context, columns=None) -> pd.DataFrame:
    """Encoded matrix restricted to columns, non-constant only."""
    e = ctx.enc[[c for c in _cols(ctx, columns) if c in ctx.enc]]
    return e.loc[:, e.std() > 0]


def _rows(ctx: Context, n: int) -> np.ndarray:
    """Seeded subset of sample positions, row order preserved."""
    m = len(ctx.sample)
    if m <= n:
        return np.arange(m)
    return np.sort(np.random.default_rng(ctx.seed).choice(m, n, replace=False))


def _binary(ctx: Context) -> bool:
    return ctx.is_cls and len(np.unique(ctx.yc)) == 2


def _cv(model, X, y, ctx: Context) -> float:
    """3-fold CV score: ROC AUC (binary), accuracy (multiclass), R2 (regression)."""
    if ctx.is_cls:
        cv, scoring = StratifiedKFold(3, shuffle=True, random_state=ctx.seed), "roc_auc" if _binary(ctx) else "accuracy"
    else:
        cv, scoring = KFold(3, shuffle=True, random_state=ctx.seed), "r2"
    return float(cross_val_score(model, X, y, cv=cv, scoring=scoring).mean())


def _top(d: dict, k: int = 5, reverse: bool = True) -> dict:
    return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=reverse)[:k])


def _names(d, k: int = 3) -> str:
    return ", ".join(list(d)[:k])


def _id_cols(ctx: Context) -> list[str]:
    n = len(ctx.df)
    return [c for c in ctx.df.columns if c != ctx.target
            and (pd.api.types.is_integer_dtype(ctx.df[c]) or _is_cat(ctx.df[c]))
            and ctx.df[c].nunique() / max(n, 1) > 0.95]


def _halves(ctx: Context, columns=None) -> tuple[pd.DataFrame, np.ndarray]:
    """Encoded features without ids/time columns, plus first-half/second-half labels in row order."""
    drop = set(_id_cols(ctx)) | {ctx.time_column}
    e = _encn(ctx, columns)
    e = e[[c for c in e.columns if c not in drop]]
    return e, (np.arange(len(e)) >= len(e) // 2).astype(int)


def _ols(ctx: Context):
    import statsmodels.api as sm
    idx = _rows(ctx, 5000)
    e = _encn(ctx).iloc[idx]
    Z = sm.add_constant(StandardScaler().fit_transform(e), has_constant="add")
    return sm.OLS(ctx.yc[idx], Z).fit()


# ---------- quality ----------

@diagnostic("missing_pattern", "quality", "missing rate per column, overall rate, and number of distinct missingness patterns")
def missing_pattern(ctx: Context, columns=None) -> dict:
    m = ctx.X[_cols(ctx, columns)].isna()
    rates = {c: float(r) for c, r in m.mean().items() if r > 0}
    worst = max(rates.values(), default=0.0)
    v = {"overall": float(m.to_numpy().mean()) if m.size else 0.0, "top": _top(rates),
         "n_patterns": int(m.drop_duplicates().shape[0]), "cols_over_50pct": sum(r > 0.5 for r in rates.values())}
    sev = 3 if worst > 0.9 else 2 if worst > 0.5 else 1 if worst > 0.05 else 0
    if not rates:
        return _out(v, 0, "No missing values.")
    return _out(v, sev, f"{len(rates)} columns have missing values, worst {_names(v['top'], 1)} at {worst:.0%}.")


@diagnostic("missing_informative", "quality", "mutual information between each column's missing indicator and the target")
def missing_informative(ctx: Context, columns=None) -> dict:
    cols = [c for c in _cols(ctx, columns) if ctx.X[c].isna().any()]
    if not cols:
        return _out({}, 0, "No missing values to be informative.")
    M = ctx.X[cols].isna().astype(int).to_numpy()
    fn = mutual_info_classif if ctx.is_cls else mutual_info_regression
    mi = dict(zip(cols, fn(M, ctx.yc, discrete_features=True, random_state=ctx.seed)))
    v, best = _top(mi), max(mi.values())
    sev = 2 if best > 0.05 else 1 if best > 0.01 else 0
    msg = f"Missingness in {_names(v, 1)} carries target signal; add a missing indicator." if sev else \
        "Missingness looks unrelated to the target."
    return _out(v, sev, msg)


@diagnostic("duplicates_exact", "quality", "fraction of exact duplicate rows and duplicate features with conflicting labels")
def duplicates_exact(ctx: Context, columns=None) -> dict:
    cols = _cols(ctx, columns)
    dup = float(ctx.df.duplicated().mean())
    g = ctx.sample.groupby(cols, dropna=False)[ctx.target].nunique() if cols else pd.Series(dtype=float)
    conflict = int((g > 1).sum()) if ctx.is_cls else 0
    sev = 2 if dup > 0.05 else 1 if dup > 0.01 or conflict else 0
    return _out({"dup_rate": dup, "conflicting_groups": conflict}, sev,
                f"{dup:.1%} exact duplicate rows, {conflict} feature groups with conflicting labels.")


@diagnostic("constant", "quality", "constant or quasi-constant columns (top value > 99%)")
def constant(ctx: Context, columns=None) -> dict:
    const, quasi = [], []
    for c in _cols(ctx, columns):
        s = ctx.df[c]
        if s.nunique(dropna=False) <= 1:
            const.append(c)
        elif s.value_counts(normalize=True, dropna=False).iloc[0] > 0.99:
            quasi.append(c)
    sev = 2 if const else 1 if quasi else 0
    msg = f"Drop constant {const[:5]}; quasi-constant {quasi[:5]}." if sev else "No constant columns."
    return _out({"constant": const, "quasi_constant": quasi}, sev, msg)


@diagnostic("id_like", "quality", "integer or string columns with unique ratio > 0.95 (row identifiers)")
def id_like(ctx: Context, columns=None) -> dict:
    ids = [c for c in _id_cols(ctx) if columns is None or c in columns]
    return _out(ids, 2 if ids else 0, f"Drop id-like columns {ids[:5]}." if ids else "No id-like columns.")


@diagnostic("high_cardinality", "quality", "categorical columns with > 50 levels (excluding id-like)")
def high_cardinality(ctx: Context, columns=None) -> dict:
    ids = set(_id_cols(ctx))
    card = {c: int(ctx.df[c].nunique()) for c in _cat(ctx, columns) if c not in ids}
    high = _top({c: k for c, k in card.items() if k > 50})
    msg = f"High-cardinality categoricals {_names(high)}; use target or frequency encoding." if high else \
        "No high-cardinality categoricals."
    return _out(high, 1 if high else 0, msg)


@diagnostic("rare_levels", "quality", "categorical levels under 1% frequency and the share of rows they cover")
def rare_levels(ctx: Context, columns=None) -> dict:
    ids, v = set(_id_cols(ctx)), {}
    for c in _cat(ctx, columns):
        if c in ids:
            continue
        f = ctx.df[c].value_counts(normalize=True)
        rare = f[f < 0.01]
        if len(rare):
            v[c] = {"n_rare": int(len(rare)), "row_share": float(rare.sum())}
    sev = 1 if any(x["row_share"] > 0.05 for x in v.values()) else 0
    msg = f"Rare levels in {_names(v)}; group them into 'other'." if v else "No rare categorical levels."
    return _out(dict(list(v.items())[:5]), sev, msg)


@diagnostic("mixed_types", "quality", "string columns that are partly numeric (numbers stored as text or dirty values)")
def mixed_types(ctx: Context, columns=None) -> dict:
    v = {}
    for c in _cat(ctx, columns):
        s = ctx.X[c].dropna().astype(str)
        if len(s):
            r = float(pd.to_numeric(s, errors="coerce").notna().mean())
            if 0.05 < r < 1:
                v[c] = r
    sev = 2 if any(r > 0.8 for r in v.values()) else 1 if v else 0
    msg = f"Mixed numeric/text values in {_names(v)}; clean and cast." if v else "No mixed-type columns."
    return _out(_top(v), sev, msg)


@diagnostic("mad_outliers", "quality", "fraction of values with robust (MAD) z-score > 3.5 per numeric column")
def mad_outliers(ctx: Context, columns=None) -> dict:
    v = {}
    for c in _num(ctx, columns):
        x = ctx.X[c].dropna().astype(float)
        mad = float(np.median(np.abs(x - x.median()))) if len(x) else 0.0
        if mad > 0:
            v[c] = float((np.abs(0.6745 * (x - x.median()) / mad) > 3.5).mean())
    v = {c: r for c, r in v.items() if r > 0}
    worst = max(v.values(), default=0.0)
    sev = 2 if worst > 0.1 else 1 if worst > 0.02 else 0
    msg = f"Outliers in {_names(_top(v))} (worst {worst:.1%}); prefer robust scaling or trees." if sev else \
        "Few robust outliers."
    return _out(_top(v), sev, msg)


def _dt_share(s: pd.Series) -> float:
    s = s.dropna().astype(str).head(200)
    if not len(s) or pd.to_numeric(s, errors="coerce").notna().mean() > 0.5:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(pd.to_datetime(s, errors="coerce", format="mixed").notna().mean())


@diagnostic("datetime_like", "quality", "string columns that parse as dates, plus native datetime columns")
def datetime_like(ctx: Context, columns=None) -> dict:
    native = [c for c in _cols(ctx, columns) if _is_dt(ctx.X[c])]
    strs = [c for c in _cat(ctx, columns) if _dt_share(ctx.X[c]) > 0.9]
    sev = 1 if strs or native else 0
    msg = f"Date columns {(native + strs)[:5]}; extract year/month/dayofweek, never feed raw." if sev else \
        "No date-like columns."
    return _out({"native": native, "strings": strs}, sev, msg)


@diagnostic("text_like", "quality", "string columns with long free text (mean length > 30 chars)")
def text_like(ctx: Context, columns=None) -> dict:
    v = {c: float(ctx.X[c].dropna().astype(str).str.len().mean()) for c in _cat(ctx, columns)
         if ctx.X[c].notna().any()}
    v = {c: m for c, m in v.items() if m > 30}
    msg = f"Free text in {_names(v)}; drop or vectorize (tf-idf)." if v else "No free-text columns."
    return _out(_top(v), 1 if v else 0, msg)


# ---------- target ----------

@diagnostic("target_missing", "target", "rows with a missing target (excluded from all checks)")
def target_missing(ctx: Context, columns=None) -> dict:
    r = float(ctx.df[ctx.target].isna().mean())
    sev = 2 if r > 0.05 else 1 if r > 0 else 0
    return _out(r, sev, f"{r:.1%} of rows have no target; drop them before training." if r else "Target is complete.")


@diagnostic("imbalance", "target", "class ratio (majority/minority) and normalized entropy, classification only")
def imbalance(ctx: Context, columns=None) -> dict:
    if not ctx.is_cls:
        return _out(None, 0, "Not applicable to regression.")
    p = ctx.y.value_counts(normalize=True)
    ratio = float(p.iloc[0] / p.iloc[-1])
    ent = float(stats.entropy(p) / np.log(len(p))) if len(p) > 1 else 0.0
    sev = 3 if ratio >= 50 else 2 if ratio >= 5 else 1 if ratio >= 2 else 0
    v = {"ratio": ratio, "norm_entropy": ent, "minority": str(p.index[-1]), "minority_share": float(p.iloc[-1])}
    msg = f"Imbalanced {ratio:.1f}:1; use class_weight and roc_auc or f1_macro." if sev else "Classes are balanced."
    return _out(v, sev, msg)


@diagnostic("target_skew", "target", "skew, kurtosis, and Yeo-Johnson lambda of the target, regression only")
def target_skew(ctx: Context, columns=None) -> dict:
    if ctx.is_cls:
        return _out(None, 0, "Not applicable to classification.")
    y = ctx.yc
    sk, ku = float(stats.skew(y)), float(stats.kurtosis(y))
    lam = float(stats.yeojohnson_normmax(y))
    sev = 2 if abs(sk) > 2 else 1 if abs(sk) > 1 else 0
    hint = "log1p" if y.min() >= 0 else "Yeo-Johnson"
    msg = f"Target skew {sk:.2f}; try a {hint} transform (lambda {lam:.2f})." if sev else "Target roughly symmetric."
    return _out({"skew": sk, "kurtosis": ku, "yj_lambda": lam}, sev, msg)


@diagnostic("label_noise_knn", "target", "5-NN label disagreement rate on scaled features vs the majority baseline")
def label_noise_knn(ctx: Context, columns=None) -> dict:
    if not ctx.is_cls:
        return _out(None, 0, "Not applicable to regression.")
    idx = _rows(ctx, SCORE_ROWS)
    Z, y = StandardScaler().fit_transform(_encn(ctx, columns).iloc[idx]), ctx.yc[idx]
    nb = NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    rate = float((y[nb] != y[:, None]).mean())
    base = float(1 - np.sum(np.bincount(y) ** 2) / len(y) ** 2)
    rel = rate / base if base > 0 else 0.0
    sev = 2 if rel > 0.9 else 1 if rel > 0.7 else 0
    msg = f"Neighbors disagree on {rate:.0%} of labels (chance {base:.0%}); noisy labels or weak features." if sev \
        else f"Neighbors disagree on {rate:.0%} of labels vs chance {base:.0%}."
    return _out({"rate": rate, "chance": base}, sev, msg)


@diagnostic("heteroscedasticity_bp", "target", "Breusch-Pagan test on OLS residuals, regression only")
def heteroscedasticity_bp(ctx: Context, columns=None) -> dict:
    if ctx.is_cls:
        return _out(None, 0, "Not applicable to classification.")
    from statsmodels.stats.diagnostic import het_breuschpagan
    res = _ols(ctx)
    lm, p, _, _ = het_breuschpagan(res.resid, res.model.exog)
    sev = 1 if p < 0.01 else 0
    msg = "Residual variance depends on features; consider a log target or robust loss." if sev else \
        "No strong heteroscedasticity."
    return _out({"lm": float(lm), "p": float(p)}, sev, msg)


# ---------- signal ----------

@diagnostic("mutual_info", "signal", "mutual information of each feature with the target (normalized by H(y) for classification); top 5")
def mutual_info(ctx: Context, columns=None) -> dict:
    idx = _rows(ctx, 5000)
    e = _encn(ctx, columns).iloc[idx]
    disc = np.array([_is_cat(ctx.X[c]) or pd.api.types.is_bool_dtype(ctx.X[c]) for c in e.columns], dtype=bool)
    fn = mutual_info_classif if ctx.is_cls else mutual_info_regression
    mi = fn(e.to_numpy(), ctx.yc[idx], discrete_features=disc, random_state=ctx.seed)
    if ctx.is_cls:
        mi = mi / max(stats.entropy(np.bincount(ctx.yc[idx])), 1e-12)
    v = _top(dict(zip(e.columns, map(float, mi))))
    best = max(v.values(), default=0.0)
    sev = 2 if best < 0.01 else 0
    return _out(v, sev, "Almost no feature carries target information." if sev else f"Most informative: {_names(v)}.")


@diagnostic("spearman_pearson_gap", "signal", "max |spearman| - |pearson| with the target (monotone nonlinearity)")
def spearman_pearson_gap(ctx: Context, columns=None) -> dict:
    y = pd.Series(ctx.yc, index=ctx.X.index)
    gaps = {}
    for c in _num(ctx, columns):
        x = ctx.enc[c]
        if x.std() > 0:
            gaps[c] = abs(x.corr(y, method="spearman")) - abs(x.corr(y))
    v = _top({c: g for c, g in gaps.items() if np.isfinite(g)}, 3)
    best = max(v.values(), default=0.0)
    sev = 1 if best > 0.1 else 0
    msg = f"{_names(v, 1)} relates to the target nonlinearly; trees or transforms help." if sev else \
        "Rank and linear correlations agree."
    return _out(v, sev, msg)


@diagnostic("single_feature_score", "signal", "per-feature 3-fold CV of a depth-3 tree on that feature alone (AUC or R2)")
def single_feature_score(ctx: Context, columns=None) -> dict:
    s = {c: v for c, v in ctx.single_scores.items() if columns is None or c in columns}
    v = _top(s)
    return _out(v, 0, f"Best single features: {_names(v)}.")


@diagnostic("leakage", "signal", "any single feature scoring >= 0.98 alone (AUC or R2) is probable target leakage")
def leakage(ctx: Context, columns=None) -> dict:
    s = {c: v for c, v in ctx.single_scores.items() if columns is None or c in columns}
    leaks = _top({c: v for c, v in s.items() if v >= 0.98})
    sus = _top({c: v for c, v in s.items() if 0.9 <= v < 0.98})
    if leaks:
        return _out(leaks, 3, f"Probable leakage: {_names(leaks)} predict the target alone; drop them.")
    if sus:
        return _out(sus, 2, f"Suspiciously strong single features: {_names(sus)}; inspect for leakage.")
    return _out({}, 0, "No single feature predicts the target suspiciously well.")


# ---------- structure ----------

def _onehot_p(ctx: Context) -> int:
    return int(sum(max(ctx.X[c].nunique(), 1) - 1 if _is_cat(ctx.X[c]) else 1 for c in ctx.X.columns))


@diagnostic("n_over_p", "structure", "rows per (one-hot estimated) feature")
def n_over_p(ctx: Context, columns=None) -> dict:
    p = max(_onehot_p(ctx), 1)
    r = len(ctx.df) / p
    sev = 3 if r < 1 else 2 if r < 10 else 0
    msg = f"Only {r:.1f} rows per feature; regularize strongly and prefer simple models." if sev else \
        f"{r:.0f} rows per feature."
    return _out({"n": len(ctx.df), "p_onehot": p, "ratio": r}, sev, msg)


@diagnostic("vif_condition", "structure", "condition number of the standardized numeric matrix and max VIF")
def vif_condition(ctx: Context, columns=None) -> dict:
    e = _encn(ctx, _num(ctx, columns))
    if e.shape[1] < 2:
        return _out(None, 0, "Fewer than two numeric features.")
    Z = StandardScaler().fit_transform(e)
    cond = float(np.linalg.cond(Z))
    vif = dict(zip(e.columns, np.diag(np.linalg.pinv(np.corrcoef(Z, rowvar=False)))))
    worst = max(vif.values())
    sev = 2 if worst > 100 else 1 if worst > 10 else 0
    msg = f"Multicollinearity (max VIF {worst:.0f} in {_names(_top(vif), 1)}); use Ridge or drop one." if sev else \
        "No serious multicollinearity."
    return _out({"condition": cond, "max_vif": worst, "top_vif": _top(vif, 3)}, sev, msg)


@diagnostic("pca_intrinsic_dim", "structure", "PCA components needed for 95% variance, as a fraction of p")
def pca_intrinsic_dim(ctx: Context, columns=None) -> dict:
    e = _encn(ctx, columns)
    if e.shape[1] < 2:
        return _out(None, 0, "Fewer than two features.")
    var = np.cumsum(PCA(random_state=ctx.seed).fit(StandardScaler().fit_transform(e)).explained_variance_ratio_)
    k = int(np.searchsorted(var, 0.95) + 1)
    ratio = k / e.shape[1]
    sev = 1 if ratio < 0.3 and e.shape[1] >= 5 else 0
    msg = f"{k} of {e.shape[1]} components hold 95% variance; features are redundant." if sev else \
        f"{k} of {e.shape[1]} components hold 95% variance."
    return _out({"k95": k, "ratio": ratio}, sev, msg)


@diagnostic("redundant_pairs", "structure", "feature pairs with |spearman| > 0.95")
def redundant_pairs(ctx: Context, columns=None) -> dict:
    e = _encn(ctx, columns)
    c = e.corr(method="spearman").abs().to_numpy()
    iu = np.triu_indices_from(c, 1)
    pairs = [(e.columns[i], e.columns[j], float(c[i, j])) for i, j in zip(*iu) if c[i, j] > 0.95]
    pairs = sorted(pairs, key=lambda t: -t[2])[:5]
    msg = f"{len(pairs)} near-duplicate pairs, e.g. {pairs[0][0]} ~ {pairs[0][1]}; drop one of each." if pairs else \
        "No redundant feature pairs."
    return _out([list(p) for p in pairs], 1 if pairs else 0, msg)


@diagnostic("sparsity", "structure", "fraction of zeros in the one-hot-expanded design matrix (estimate)")
def sparsity(ctx: Context, columns=None) -> dict:
    n, zeros, cells = len(ctx.X), 0, 0
    for c in _cols(ctx, columns):
        s = ctx.X[c]
        if _is_cat(s):
            k = max(s.nunique(), 1)
            zeros, cells = zeros + n * (k - 1), cells + n * k
        elif not _is_dt(s):
            zeros, cells = zeros + int((s.fillna(1) == 0).sum()), cells + n
    r = zeros / cells if cells else 0.0
    sev = 1 if r > 0.8 else 0
    return _out(r, sev, f"Design matrix is {r:.0%} zeros; prefer sparse-friendly models." if sev else f"{r:.0%} zeros.")


# ---------- complexity ----------

@diagnostic("landmarkers", "complexity", "3-fold CV of linear, stump, depth-3 tree, 1-NN, naive Bayes; nonlinearity_gap = tree3 - linear")
def landmarkers(ctx: Context, columns=None) -> dict:
    idx = _rows(ctx, MODEL_ROWS)
    X, y = _encn(ctx, columns).iloc[idx], ctx.yc[idx]
    s = ctx.seed
    if ctx.is_cls:
        models = {"linear": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
                  "stump": DecisionTreeClassifier(max_depth=1, random_state=s),
                  "tree3": DecisionTreeClassifier(max_depth=3, random_state=s),
                  "nn1": make_pipeline(StandardScaler(), KNeighborsClassifier(1)),
                  "nb": GaussianNB()}
    else:
        models = {"linear": make_pipeline(StandardScaler(), Ridge()),
                  "stump": DecisionTreeRegressor(max_depth=1, random_state=s),
                  "tree3": DecisionTreeRegressor(max_depth=3, random_state=s),
                  "nn1": make_pipeline(StandardScaler(), KNeighborsRegressor(1))}
    v = {k: _cv(m, X, y, ctx) for k, m in models.items()}
    v["nonlinearity_gap"] = v["tree3"] - v["linear"]
    sev = 1 if v["nonlinearity_gap"] > 0.05 else 0
    msg = f"Depth-3 tree beats linear by {v['nonlinearity_gap']:.2f}; nonlinear models favored." if sev else \
        f"Linear baseline is competitive (gap {v['nonlinearity_gap']:.2f})."
    return _out(v, sev, msg)


@diagnostic("reset_test", "complexity", "Ramsey RESET test for missing nonlinearity in OLS, regression only")
def reset_test(ctx: Context, columns=None) -> dict:
    if ctx.is_cls:
        return _out(None, 0, "Not applicable to classification.")
    from statsmodels.stats.diagnostic import linear_reset
    r = linear_reset(_ols(ctx), power=2, use_f=True)
    p = float(r.pvalue)
    sev = 1 if p < 0.01 else 0
    return _out({"f": float(r.statistic), "p": p}, sev,
                "OLS misses nonlinear structure (RESET)." if sev else "RESET finds no missing nonlinearity.")


@diagnostic("fisher_f1", "complexity", "max Fisher discriminant ratio over features, classification only")
def fisher_f1(ctx: Context, columns=None) -> dict:
    if not ctx.is_cls:
        return _out(None, 0, "Not applicable to regression.")
    e, y = _encn(ctx, columns), ctx.yc
    g = e.groupby(y)
    n, mu, var = g.size().to_numpy()[:, None], g.mean(), g.var(ddof=0).fillna(0)
    num = (n * (mu - e.mean()) ** 2).sum()
    den = (n * var).sum().replace(0, np.nan)
    f = (num / den).dropna()
    if f.empty:
        return _out(None, 0, "No usable features.")
    best = float(f.max())
    sev = 1 if best < 0.05 else 0
    msg = f"No feature separates classes linearly (max F1 {best:.3f}); expect a hard problem." if sev else \
        f"Best separating feature {f.idxmax()} (F1 {best:.2f})."
    return _out({"max": best, "feature": str(f.idxmax()), "complexity": 1 / (1 + best)}, sev, msg)


@diagnostic("class_overlap", "complexity", "per-feature fraction of rows inside the class overlap region (F3); min over features")
def class_overlap(ctx: Context, columns=None) -> dict:
    if not ctx.is_cls:
        return _out(None, 0, "Not applicable to regression.")
    e, y = _encn(ctx, columns), ctx.yc
    g = e.groupby(y)
    lo, hi = g.min().max(), g.max().min()
    frac = {c: float(((e[c] >= lo[c]) & (e[c] <= hi[c])).mean()) for c in e.columns}
    if not frac:
        return _out(None, 0, "No usable features.")
    best = min(frac.values())
    # ponytail: informational only; F3 saturates on discrete features, label_noise_knn carries hardness severity
    msg = f"Least-overlapping feature {min(frac, key=frac.get)} ({best:.0%} of rows in the class overlap region)."
    return _out({"min_overlap": best, "top": _top(frac, 3, reverse=False)}, 0, msg)


# ---------- time / drift ----------

@diagnostic("time_order", "time", "datetime columns (native or parsed) and whether rows are sorted by them")
def time_order(ctx: Context, columns=None) -> dict:
    cands = [c for c in ctx.df.columns if c != ctx.target and (columns is None or c in columns)
             and (c == ctx.time_column or _is_dt(ctx.df[c]) or (_is_cat(ctx.df[c]) and _dt_share(ctx.df[c]) > 0.9))]
    v = {}
    for c in cands:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = ctx.df[c] if not _is_cat(ctx.df[c]) else pd.to_datetime(ctx.df[c], errors="coerce", format="mixed")
        t = t.dropna()
        v[c] = "increasing" if t.is_monotonic_increasing else "decreasing" if t.is_monotonic_decreasing else "unsorted"
    sorted_cols = [c for c, o in v.items() if o != "unsorted"]
    if sorted_cols:
        return _out(v, 2, f"Rows are ordered by {sorted_cols[0]}; use timeseries CV to avoid look-ahead.")
    if v:
        return _out(v, 1, f"Time column {list(v)[0]} present but rows unsorted; sort before timeseries CV.")
    return _out(v, 0, "No time column detected.")


@diagnostic("autocorr_target", "time", "lag-1 autocorrelation of the target in row order")
def autocorr_target(ctx: Context, columns=None) -> dict:
    y = ctx.df[ctx.target].dropna()
    y = pd.Series(pd.factorize(y)[0] if ctx.is_cls else y.astype(float).to_numpy())
    r = float(y.autocorr(1)) if y.std() > 0 else 0.0
    r = 0.0 if not np.isfinite(r) else r
    thr = max(0.1, 3 / np.sqrt(len(y)))
    sev = 2 if r > 0.3 else 1 if r > thr else 0
    msg = f"Target autocorrelation {r:.2f} in row order; rows are not exchangeable, use timeseries or group CV." if sev \
        else "No row-order autocorrelation in the target."
    return _out(r, sev, msg)


@diagnostic("adversarial_drift", "time", "classifier AUC separating first vs second half of rows (ids excluded); top features")
def adversarial_drift(ctx: Context, columns=None) -> dict:
    e, h = _halves(ctx, columns)
    if e.shape[1] == 0:
        return _out(None, 0, "No usable features.")
    idx = _rows(ctx, SCORE_ROWS)
    X, y = e.iloc[idx], h[idx]
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, n_jobs=-1, random_state=ctx.seed)
    p = cross_val_predict(rf, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=ctx.seed), method="predict_proba")
    auc = float(roc_auc_score(y, p[:, 1]))
    top = _top(dict(zip(X.columns, rf.fit(X, y).feature_importances_)), 3)
    sev = 3 if auc > 0.8 else 2 if auc > 0.7 else 1 if auc > 0.6 else 0
    msg = f"Early and late rows differ (AUC {auc:.2f}), driven by {_names(top)}; use timeseries CV." if sev else \
        f"No drift between halves (AUC {auc:.2f})."
    return _out({"auc": auc, "top": top}, sev, msg)


@diagnostic("ks_drift_per_feature", "time", "KS test first vs second half per numeric feature, Holm-corrected count")
def ks_drift_per_feature(ctx: Context, columns=None) -> dict:
    from statsmodels.stats.multitest import multipletests
    e, h = _halves(ctx, _num(ctx, columns))
    if e.shape[1] == 0:
        return _out(None, 0, "No numeric features.")
    ks = {c: stats.ks_2samp(e[c][h == 0], e[c][h == 1]) for c in e.columns}
    rej = multipletests([r.pvalue for r in ks.values()], alpha=0.05, method="holm")[0]
    sig = {c: float(r.statistic) for (c, r), b in zip(ks.items(), rej) if b}
    sev = 2 if len(sig) > len(ks) / 2 else 1 if sig else 0
    msg = f"{len(sig)} of {len(ks)} features shift between halves: {_names(_top(sig))}." if sig else \
        "No feature distribution shifts between halves."
    return _out({"n_sig": len(sig), "n": len(ks), "top": _top(sig, 3)}, sev, msg)


# ---------- runner ----------

def run_check(name: str, ctx: Context, columns=None) -> dict:
    """Run one check; never raises."""
    if name not in DIAGNOSTICS:
        return {"severity": 0, "error": f"unknown check {name!r}; valid: {sorted(DIAGNOSTICS)}"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _sig(DIAGNOSTICS[name].fn(ctx, columns))
    except Exception as e:
        return {"severity": 0, "error": f"{type(e).__name__}: {e}"[:300]}


def _meta(ctx: Context, checks: dict) -> dict[str, float]:
    def val(name, *keys):
        v = checks.get(name, {}).get("value")
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None
        return v if isinstance(v, (int, float)) else None
    cats = _cat(ctx)
    skews = [abs(stats.skew(ctx.enc[c])) for c in _num(ctx) if ctx.enc[c].std() > 0]
    m = {"n_rows": len(ctx.df), "n_features": ctx.X.shape[1], "n_over_p": val("n_over_p", "ratio"),
         "frac_categorical": len(cats) / max(ctx.X.shape[1], 1),
         "max_cardinality": max((ctx.X[c].nunique() for c in cats), default=0),
         "missing_rate": val("missing_pattern", "overall"), "max_abs_skew": max(skews, default=0.0),
         "imbalance_ratio": val("imbalance", "ratio"), "pca_dim_ratio": val("pca_intrinsic_dim", "ratio"),
         "drift_auc": val("adversarial_drift", "auc"), "target_autocorr": val("autocorr_target")}
    for k in ("linear", "stump", "tree3", "nn1", "nb", "nonlinearity_gap"):
        m[k if k == "nonlinearity_gap" else f"lm_{k}"] = val("landmarkers", k)
    return {k: float(v) for k, v in _sig(m).items() if v is not None}


def run_all(ctx: Context) -> dict:
    """Run every check; return top findings, a meta-feature vector, and all results."""
    checks = {name: run_check(name, ctx) for name in DIAGNOSTICS}
    ranked = sorted(((n, r) for n, r in checks.items() if r.get("severity", 0) > 0 and "finding" in r),
                    key=lambda nr: -nr[1]["severity"])
    findings = [{"check": n, "severity": r["severity"], "finding": r["finding"]} for n, r in ranked[:MAX_FINDINGS]]
    return {"findings": findings, "meta": _meta(ctx, checks), "checks": checks}


def catalog() -> list[str]:
    """One line per check, for the system prompt."""
    return [f"{c.name}: {c.description}" for c in DIAGNOSTICS.values()]
