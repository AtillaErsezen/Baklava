"""inspect_column is k-anonymous: its output feeds the LLM, whose narration lands in a public event feed, so a
raw value may appear only if at least K_MIN rows share it and it does not look like personal data.
Offline: Modal stubbed. Run: uv run python test_privacy.py"""
import functools
import json

import numpy as np
import pandas as pd

import agent
import factory_tools

factory_tools.upload_dataset = lambda df: "/datasets/fake.parquet"

FIRST = ["Alice", "Bruno", "Chiara", "Dmitri", "Elif", "Farah", "Goran", "Hanna", "Ibrahim", "Jana",
         "Kenji", "Lotte", "Mateo", "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Sven", "Tamar"]
LAST = ["Moreau", "Novak", "Okafor", "Petrov", "Quispe", "Rahman", "Silva", "Tanaka", "Umarov", "Vidal",
        "Weber", "Xu", "Yilmaz", "Zamora", "Achterberg"]
NAMES = [f"{f} {l}" for f in FIRST for l in LAST]  # 300 unique full names, one per row
N = len(NAMES)
EMAILS = [f"user{i}@example.com" for i in range(N // 10)]  # 30 addresses, each on 10 rows


class FakeFn:
    def map(self, specs):
        return []

    def remote(self, *args):
        return {}


@functools.cache
def make_run(path="runs/_privacy.csv"):
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "name": NAMES,
        "email": [EMAILS[i // 10] for i in range(N)],
        "plan": ["basic", "plus", "pro"] * (N // 3),
        "city": ["Amsterdam"] * 150 + ["Utrecht"] * 145 + ["Delft"] * 3 + ["Leiden"] * 2,
        "spend": rng.gamma(2, 50, N).round(2),
        "tickets": [i % 5 for i in range(N - 1)] + [17],
        "y": rng.integers(0, 2, N),
    }).to_csv(path, index=False)
    agent.modal.Function.from_name = lambda app, name: FakeFn()
    return agent.FactoryRun(path, "y", provider="scripted")


def inspect(column):
    return make_run().tool_inspect_column({"column": column})


def test_unique_names_never_appear_raw():
    out = inspect("name")
    dumped = json.dumps(out)
    assert not [n for n in NAMES if n in dumped]
    assert out["sample"][:3] == ["<rare value #1>", "<rare value #2>", "<rare value #3>"]
    assert out["masked"] == {"rare": N, "pii_like": 0} and out["privacy_note"]


def test_frequent_emails_are_still_masked():
    out = inspect("email")
    dumped = json.dumps(out)
    assert not [e for e in EMAILS if e in dumped] and "@" not in dumped
    assert all(v.startswith("<pii-like value #") for v in out["sample"])
    assert set(out["value_counts"].values()) == {10}
    assert out["masked"] == {"rare": 0, "pii_like": len(EMAILS)}


def test_common_levels_are_shown_raw_with_target_rates():
    out = inspect("plan")
    assert set(out["sample"]) == {"basic", "plus", "pro"}
    assert out["value_counts"] == {"basic": 100, "plus": 100, "pro": 100}
    assert set(out["target_rate_by_value"]) == {"basic", "plus", "pro"}
    assert out["masked"] == {"rare": 0, "pii_like": 0}


def test_rare_levels_are_pooled_in_target_rates():
    run = make_run()
    out = inspect("city")
    dumped = json.dumps(out)
    assert "Delft" not in dumped and "Leiden" not in dumped
    assert out["masked"] == {"rare": 2, "pii_like": 0}
    assert set(out["target_rate_by_value"]) == {"Amsterdam", "Utrecht", "<rare values>"}
    rare_y = run.df.loc[run.df["city"].isin(["Delft", "Leiden"]), "y"].astype(str)
    assert out["target_rate_by_value"]["<rare values>"] == {k: round(float((rare_y == k).mean()), 3) for k in ("0", "1")}


def test_numeric_column_returns_a_summary_not_raw_values():
    spend = make_run().df["spend"]
    out = inspect("spend")
    assert "sample" not in out and "value_counts" not in out and "target_rate_by_value" not in out
    assert set(out["summary"]) == {"min", "p25", "median", "p75", "max", "n_unique"}
    assert out["summary"]["n_unique"] == spend.nunique() and out["summary"]["max"] == round(float(spend.max()), 4)


def test_low_cardinality_numeric_keeps_value_counts_under_k_min():
    out = inspect("tickets")
    assert "sample" not in out and "summary" in out
    assert out["value_counts"] == {**{str(i): 60 for i in range(4)}, "4": 59, "<rare value #1>": 1}
    assert out["masked"] == {"rare": 1, "pii_like": 0}


def test_every_column_output_is_json_serializable():
    run = make_run()
    for c in run.df.columns:
        out = inspect(c)
        assert json.loads(json.dumps(out)) == out and "masked" in out and "privacy_note" in out
    assert "error" in inspect("nope")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
