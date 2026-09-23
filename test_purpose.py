"""Tests for purpose.py: keyword rules, constrained LLM call, per-field fallback. No network."""
import json
from types import SimpleNamespace

from purpose import PURPOSE_SCHEMA, purpose_spec, rule_spec

KEYS = {"primary_metric", "interpretable", "max_predict_ms", "class_weighting", "operating_point", "notes"}


def test_rule_regulator_wants_interpretable():
    s = rule_spec("Explain loan decisions to the regulator", "classification", {})
    assert set(s) == KEYS and s["interpretable"] and s["primary_metric"] == "roc_auc" and s["max_predict_ms"] is None


def test_rule_realtime_catch_fraud():
    s = rule_spec("Real-time scoring in milliseconds; catch every fraud case", "classification", {})
    assert s["max_predict_ms"] == 50 and s["primary_metric"] == "f1_macro" and s["operating_point"] == "high recall"


def test_rule_false_alarms_means_precision():
    s = rule_spec("Flag defects but keep false alarms rare", "classification", {})
    assert s["operating_point"] == "high precision" and s["primary_metric"] == "roc_auc"


def test_rule_imbalance_sets_weighting_and_f1():
    s = rule_spec(None, "classification", {"imbalance_ratio": 3.9})
    assert s["class_weighting"] and s["primary_metric"] == "f1_macro"
    assert rule_spec("maximize AUC", "classification", {"imbalance_ratio": 5})["primary_metric"] == "roc_auc"


def test_rule_regression_metrics():
    assert rule_spec("Forecast weekly demand", "regression", {})["primary_metric"] == "rmse"
    s = rule_spec("Price homes; relative error matters", "regression", {"imbalance_ratio": 9})
    assert s["primary_metric"] == "mae" and not s["class_weighting"] and s["operating_point"] is None


def test_notes_have_no_dashes():
    s = rule_spec("Explain, fast, catch misses", "classification", {"imbalance_ratio": 4})
    assert chr(0x2014) not in s["notes"] and chr(0x2013) not in s["notes"] and s["notes"]


# ---- purpose_spec
def fake_client(payload: dict | None = None, error: Exception | None = None):
    calls = []

    def create(**kw):
        calls.append(kw)
        if error:
            raise error
        msg = SimpleNamespace(content=json.dumps(payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    ledger = SimpleNamespace(add=lambda u: calls.append("ledger"))
    comp = SimpleNamespace(create=create)
    return SimpleNamespace(model="m", ledger=ledger, client=SimpleNamespace(chat=SimpleNamespace(completions=comp))), calls


def test_schema_is_strict():
    assert set(PURPOSE_SCHEMA["required"]) == KEYS and PURPOSE_SCHEMA["additionalProperties"] is False


def test_purpose_spec_uses_valid_llm_output_without_tools():
    llm = {"primary_metric": "accuracy", "interpretable": True, "max_predict_ms": 20, "class_weighting": False,
           "operating_point": None, "notes": "Balanced classes; accuracy is fine."}
    client, calls = fake_client(llm)
    assert purpose_spec(client, "Explain it, 20 ms budget", "classification", {}) == llm
    assert "tools" not in calls[0] and calls[0]["response_format"]["json_schema"]["strict"] is True
    assert "ledger" in calls


def test_purpose_spec_invalid_metric_falls_back_per_field():
    llm = {"primary_metric": "rmse", "interpretable": "yes", "max_predict_ms": -3, "class_weighting": True,
           "operating_point": "high recall", "notes": "x"}
    client, _ = fake_client(llm)
    s = purpose_spec(client, "catch churners early", "classification", {})
    rule = rule_spec("catch churners early", "classification", {})
    assert s["primary_metric"] == rule["primary_metric"] == "f1_macro"
    assert s["interpretable"] == rule["interpretable"] and s["max_predict_ms"] == rule["max_predict_ms"]
    assert s["class_weighting"] is True  # valid LLM field kept


def test_purpose_spec_falls_back_on_error_scripted_and_no_purpose():
    rule = rule_spec("fast", "regression", {})
    raising, _ = fake_client(error=RuntimeError("down"))
    assert purpose_spec(raising, "fast", "regression", {}) == rule
    assert purpose_spec(SimpleNamespace(), "fast", "regression", {}) == rule  # ScriptedClient: no .client
    quiet, calls = fake_client({})
    assert purpose_spec(quiet, None, "regression", {}) == rule_spec(None, "regression", {}) and calls == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
