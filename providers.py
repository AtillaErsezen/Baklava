"""
LLM providers for the agent loop, all speaking the OpenAI chat-completions shape.

  NebiusClient    open-weight models on Nebius Token Factory (OpenAI-compatible API)
  ScriptedClient  fixed tool sequence, no API key; tests the pipeline end to end
"""
import json
import os
from types import SimpleNamespace

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# USD per 1M tokens (input, output); used only for the ledger. Override with AGENT_PRICE_IN / AGENT_PRICE_OUT.
DEFAULT_PRICES = (0.20, 0.60)


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert our tool specs ({name, description, input_schema}) to OpenAI function tools."""
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


def parse_arguments(raw: str | None, schema: dict) -> tuple[dict | None, str | None]:
    """Parse a tool call's JSON arguments and check required keys. Returns (args, None) or
    (None, error). Errors are written for the model: what is wrong and how to fix it."""
    if not raw or not raw.strip():
        args = {}
    else:
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as e:
            return None, f"arguments are not valid JSON ({e.msg} at char {e.pos}). Send one JSON object."
    if not isinstance(args, dict):
        return None, "arguments must be a JSON object, e.g. {\"column\": \"age\"}."
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        props = list(schema.get("properties", {}))
        return None, f"missing required field(s) {missing}. required: {schema['required']}; allowed: {props}."
    return args, None


class Ledger:
    """Token and cost accounting across every LLM call in a run."""

    def __init__(self, price_in: float, price_out: float):
        self.price_in, self.price_out = price_in, price_out
        self.calls = self.input_tokens = self.output_tokens = 0

    def add(self, usage) -> None:
        self.calls += 1
        if usage is not None:
            self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.output_tokens += getattr(usage, "completion_tokens", 0) or 0

    def totals(self) -> dict:
        usd = (self.input_tokens * self.price_in + self.output_tokens * self.price_out) / 1e6
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": round(usd, 6)}


class NebiusClient:
    """Chat completions against Nebius Token Factory. Tool calls are never streamed
    (Qwen tool-call parsing is unreliable when streaming)."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        key = os.getenv("NEBIUS_API_KEY")
        if not key:
            raise SystemExit("NEBIUS_API_KEY is not set. Put it in .env and run: uv run --env-file .env agent.py ...")
        self.model = model or os.getenv("AGENT_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=key)
        self.ledger = Ledger(float(os.getenv("AGENT_PRICE_IN", DEFAULT_PRICES[0])),
                             float(os.getenv("AGENT_PRICE_OUT", DEFAULT_PRICES[1])))

    def chat(self, messages: list[dict], tools: list[dict], max_tokens: int = 2048):
        """One completion. Returns (message, finish_reason)."""
        kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0,
                                                   max_tokens=max_tokens, **kwargs)
        self.ledger.add(resp.usage)
        choice = resp.choices[0]
        return choice.message, choice.finish_reason


def _call(i: int, name: str, args: dict):
    return SimpleNamespace(id=f"dry{i}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class ScriptedClient:
    """Stand-in for an LLM: replays diag_summary -> run_search (flagged columns dropped) ->
    confirm_and_test -> finalize the 1-SE pick -> report -> stop. Everything else (Modal,
    statistics, events, export) runs for real, so the pipeline can be tested without a key."""

    def __init__(self, profile: dict, menu: dict):
        self.profile, self.menu, self.step = profile, menu, 0
        self.ledger = Ledger(0.0, 0.0)

    def chat(self, messages: list[dict], tools: list[dict], max_tokens: int = 2048):
        self.step += 1
        self.ledger.add(None)
        last = json.loads(messages[-1]["content"]) if messages[-1]["role"] == "tool" else None
        task = self.profile["target"]["suggested_task"]
        if self.step == 1:
            name, args = "diag_summary", {}
        elif self.step == 2:
            bad = {"id_like", "possible_leakage", "constant"}
            name, args = "run_search", {
                "rationale": "Scripted run: drop flagged columns, race the prior-ranked space.",
                "task": task, "primary_metric": "roc_auc" if task == "classification" else "rmse",
                "drop_columns": [c["name"] for c in self.profile["columns"] if bad & set(c.get("flags", []))]}
        elif self.step == 3:
            name, args = "confirm_and_test", {}
        elif self.step == 4:
            name, args = "finalize_model", {"candidate_name": last["recommendation"]["pick"]}
        elif self.step == 5:
            name, args = "write_report", {"why": "Scripted run: the 1-SE pick from the confirmation step.",
                                          "caveats": ["Scripted run, no LLM judgement."],
                                          "spoken_summary": "Scripted run complete."}
        else:
            return SimpleNamespace(content="", tool_calls=None), "stop"
        return SimpleNamespace(content=f"(scripted) calling {name}", tool_calls=[_call(self.step, name, args)]), "tool_calls"
