"""Turn the user's stated purpose into search settings: metric, interpretability, latency, weighting."""
import json
import re

from report import json_call

# Mirrors factory_tools.TASK_METRICS; copied so this module stays free of the Modal import chain.
TASK_METRICS = {"classification": ("roc_auc", "f1_macro", "accuracy"), "regression": ("rmse", "mae", "r2")}
FAST_MS = 50
IMBALANCED = 3.0
PURPOSE_SCHEMA = {
    "type": "object",
    "properties": {"primary_metric": {"type": "string", "enum": [m for ms in TASK_METRICS.values() for m in ms]},
                   "interpretable": {"type": "boolean"},
                   "max_predict_ms": {"type": ["number", "null"]},
                   "class_weighting": {"type": "boolean"},
                   "operating_point": {"type": ["string", "null"]},
                   "notes": {"type": "string"}},
    "required": ["primary_metric", "interpretable", "max_predict_ms", "class_weighting", "operating_point", "notes"],
    "additionalProperties": False,
}
SYSTEM = ("Map a machine learning purpose to search settings. primary_metric must be one of allowed_metrics. "
          "interpretable: true if people must read or justify the model. max_predict_ms: a latency cap only if the "
          "purpose implies one, else null. class_weighting: true for imbalanced classes. operating_point: 'high recall',"
          " 'high precision' or null. notes: one plain sentence, no em dashes.")
_EXPLICIT = {"classification": ((r"\b(auc|roc)\b", "roc_auc"), (r"\baccuracy\b", "accuracy"), (r"\bf1\b", "f1_macro")),
             "regression": ((r"\brmse\b", "rmse"), (r"\bmae\b", "mae"), (r"\b(r2|r-squared|r squared)\b", "r2"))}


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def rule_spec(purpose: str | None, task: str, meta: dict) -> dict:
    """Deterministic keyword rules; the fallback for any LLM failure and the scripted-mode path."""
    p, notes = (purpose or "").lower(), []
    interpretable = _has(r"\b(explain|interpretab|regulator|transparen)", p)
    fast = _has(r"\b(fast|real[- ]?time|latency|milliseconds?|ms)\b", p)
    explicit = next((m for pat, m in _EXPLICIT[task] if _has(pat, p)), None)
    spec = {"primary_metric": explicit, "interpretable": interpretable, "max_predict_ms": FAST_MS if fast else None,
            "class_weighting": False, "operating_point": None}
    if task == "classification":
        if _has(r"\b(recall|catch\w*|miss(es|ed|ing)?|early)\b", p):
            spec["operating_point"], spec["primary_metric"] = "high recall", explicit or "f1_macro"
        elif _has(r"\bprecision\b|false alarm", p):
            spec["operating_point"] = "high precision"
        ratio = meta.get("imbalance_ratio")
        if ratio is not None and ratio >= IMBALANCED:
            spec["class_weighting"], spec["primary_metric"] = True, spec["primary_metric"] or "f1_macro"
            notes.append(f"class weighting for imbalance ratio {ratio:.3g}")
        spec["primary_metric"] = spec["primary_metric"] or "roc_auc"
    else:
        spec["primary_metric"] = explicit or ("mae" if _has(r"percentage|relative", p) else "rmse")
    notes = (["interpretable model wanted"] if interpretable else []) + (
        [f"predict under {FAST_MS} ms"] if fast else []) + ([spec["operating_point"]] if spec["operating_point"] else []) + notes
    spec["notes"] = f"Rules: {'; '.join(notes)}; metric {spec['primary_metric']}." if notes else \
        f"Rules: no keyword matched; default metric {spec['primary_metric']} for {task}."
    return spec


def _valid(key: str, v, task: str) -> bool:
    """Per-field check of LLM output against what the harness accepts."""
    if key == "primary_metric":
        return v in TASK_METRICS[task]
    if key in ("interpretable", "class_weighting"):
        return isinstance(v, bool)
    if key == "max_predict_ms":
        return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0)
    if key == "operating_point":
        return v is None or (isinstance(v, str) and len(v) <= 40)
    return isinstance(v, str)


def purpose_spec(client, purpose: str | None, task: str, meta: dict) -> dict:
    """LLM reading of the purpose, one constrained call with no tools; invalid fields fall back to rule_spec."""
    base = rule_spec(purpose, task, meta)
    if not purpose:
        return base
    user = json.dumps({"purpose": purpose, "task": task, "allowed_metrics": TASK_METRICS[task], "meta": meta},
                      default=str)
    out = json_call(client, SYSTEM, user, "purpose_spec", PURPOSE_SCHEMA, 300) or {}
    spec = {k: out[k] if k in out and _valid(k, out[k], task) else base[k] for k in base}
    spec["notes"] = spec["notes"][:300].replace(chr(0x2014), "-").replace(chr(0x2013), "-")
    return spec
