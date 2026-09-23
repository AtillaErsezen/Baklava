"""Offline checks for scripts/probe_models.py (fake client, no network). Run: uv run python test_probe.py"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import probe_models as pm  # noqa: E402


def _msg(name: str | None, args) -> SimpleNamespace:
    if name is None:
        return SimpleNamespace(content="I would inspect it.", tool_calls=None)
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(content="", tool_calls=[SimpleNamespace(function=SimpleNamespace(name=name, arguments=raw))])


GOOD_RUN = {"task": "classification", "primary_metric": "roc_auc",
            "candidates": [{"name": "lr", "model": "logreg"}, {"name": "gb", "model": "lightgbm", "params": {}}]}


def test_score_call_cases():
    assert pm.score_call(_msg("inspect_column", {"column": "age"}), "inspect_column") == \
        {"tool": "inspect_column", "valid": True, "correct": True, "error": None}
    wrong = pm.score_call(_msg("write_report", {"markdown": "x", "spoken_summary": "y"}), "inspect_column")
    assert wrong["valid"] and not wrong["correct"]
    bad_json = pm.score_call(_msg("inspect_column", "{oops"), "inspect_column")
    assert not bad_json["valid"] and bad_json["correct"]
    nested = {**GOOD_RUN, "candidates": [{"name": "lr"}]}  # missing model inside the array
    assert not pm.score_call(_msg("run_experiments", nested), "run_experiments")["valid"]
    bad_enum = {**GOOD_RUN, "task": "clustering"}
    assert not pm.score_call(_msg("run_experiments", bad_enum), "run_experiments")["valid"]
    assert pm.score_call(_msg("run_experiments", GOOD_RUN), "run_experiments")["valid"]
    none = pm.score_call(_msg(None, None), "inspect_column")
    assert none == {"tool": None, "valid": False, "correct": False, "error": "no tool call"}
    unknown = pm.score_call(_msg("delete_all", {}), "inspect_column")
    assert not unknown["valid"] and not unknown["correct"]


def test_summarize_counts_and_cost():
    rows = [{"valid": True, "correct": True, "latency_s": 1.0, "prompt_tokens": 100, "completion_tokens": 10},
            {"valid": True, "correct": False, "latency_s": 2.0, "prompt_tokens": 100, "completion_tokens": 10},
            {"valid": False, "correct": True, "latency_s": 3.0, "prompt_tokens": 100, "completion_tokens": 10},
            {"valid": False, "correct": False, "latency_s": None, "prompt_tokens": 0, "completion_tokens": 0,
             "api_error": "boom"}]
    s = pm.summarize(rows, price_in=1.0, price_out=2.0)
    assert (s["n"], s["valid_rate"], s["correct_rate"], s["ok_rate"]) == (4, 0.5, 0.5, 0.25)
    assert s["mean_latency_s"] == 2.0 and s["api_errors"] == 1
    assert (s["prompt_tokens"], s["completion_tokens"]) == (300, 30)
    assert abs(s["est_usd"] - (300 * 1.0 + 30 * 2.0) / 1e6) < 1e-12


class FakeClient:
    """Mimics openai.OpenAI: chat.completions.create replays scripted messages."""

    def __init__(self, messages):
        self.messages, self.calls = list(messages), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        assert kwargs["tools"] and kwargs["model"] == "fake/model"
        msg = self.messages[self.calls % len(self.messages)]
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=50, completion_tokens=5))


def test_probe_model_with_fake_client(monkeypatch=None):
    prompts = [("look at age", "inspect_column"), ("run it", "run_experiments"), ("save", "write_report")]
    fake = FakeClient([_msg("inspect_column", {"column": "age"}),      # valid, correct
                       _msg("inspect_column", {"column": "x"}),        # valid, wrong tool
                       _msg("write_report", {"markdown": "# r"})])     # invalid (missing spoken_summary)
    s = pm.probe_model(fake, "fake/model", prompts)
    assert fake.calls == 3
    assert (s["valid_rate"], s["correct_rate"], s["ok_rate"]) == (round(2 / 3, 4), round(2 / 3, 4), round(1 / 3, 4))
    assert s["prompt_tokens"] == 150 and s["model"] == "fake/model"


def test_rank_orders_by_ok_rate_then_latency():
    a = {"model": "a", "ok_rate": 0.9, "mean_latency_s": 3.0}
    b = {"model": "b", "ok_rate": 0.9, "mean_latency_s": 1.0}
    c = {"model": "c", "ok_rate": 1.0, "mean_latency_s": 9.0}
    assert [r["model"] for r in pm.rank([a, b, c])] == ["c", "b", "a"]


def test_prompts_cycle_to_n():
    ps = pm.build_prompts(20)
    assert len(ps) == 20 and {t for _, t in ps} == {"inspect_column", "run_experiments", "write_report"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
