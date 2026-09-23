"""Checks for providers.py. Run: uv run python test_providers.py (or pytest)."""
import json
from types import SimpleNamespace

from providers import Ledger, ScriptedClient, parse_arguments, to_openai_tools

TOOLS = [{"name": "inspect_column", "description": "One column.",
          "input_schema": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}}]


def test_to_openai_tools_wraps_schema():
    out = to_openai_tools(TOOLS)
    assert out == [{"type": "function", "function": {
        "name": "inspect_column", "description": "One column.", "parameters": TOOLS[0]["input_schema"]}}]


def test_parse_arguments_good_and_bad():
    assert parse_arguments('{"column": "x"}', TOOLS[0]["input_schema"]) == ({"column": "x"}, None)
    args, err = parse_arguments("{not json", TOOLS[0]["input_schema"])
    assert args is None and "valid JSON" in err
    args, err = parse_arguments("{}", TOOLS[0]["input_schema"])
    assert args is None and "column" in err and "required" in err
    assert parse_arguments("", {"type": "object", "properties": {}}) == ({}, None)


def test_ledger_totals_and_cost():
    led = Ledger(price_in=0.2, price_out=0.6)
    led.add(SimpleNamespace(prompt_tokens=1000, completion_tokens=500))
    led.add(None)  # providers may omit usage
    t = led.totals()
    assert t["calls"] == 2 and t["input_tokens"] == 1000 and t["output_tokens"] == 500
    assert abs(t["usd"] - (1000 * 0.2 + 500 * 0.6) / 1e6) < 1e-12


def test_scripted_client_openai_shape():
    profile = {"target": {"suggested_task": "classification"},
               "columns": [{"name": "id", "flags": ["id_like"]}, {"name": "x", "flags": []}]}
    client = ScriptedClient(profile, menu={"classification": ["logreg"], "regression": ["ridge"]})
    msg, finish = client.chat([{"role": "user", "content": "go"}], tools=[])
    assert finish == "tool_calls" and msg.tool_calls[0].function.name == "diag_summary"
    msg, _ = client.chat([{"role": "tool", "content": json.dumps(profile)}], tools=[])
    args = json.loads(msg.tool_calls[0].function.arguments)
    assert msg.tool_calls[0].function.name == "run_search" and args["drop_columns"] == ["id"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
