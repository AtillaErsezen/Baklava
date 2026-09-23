"""
Pick the Nebius model by measurement: send canned prompts that each need one tool call and score
schema validity, tool choice, latency, tokens, and estimated cost.

  uv run --env-file .env scripts/probe_models.py
  uv run --env-file .env scripts/probe_models.py --models Qwen/Qwen3-235B-A22B-Instruct-2507 --n 10
"""
import argparse
import difflib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from providers import DEFAULT_PRICES, NebiusClient, parse_arguments, to_openai_tools  # noqa: E402

DEFAULT_MODELS = ["Qwen/Qwen3-235B-A22B-Instruct-2507", "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro",
                  "moonshotai/Kimi-K2.6", "zai-org/GLM-5.3", "openai/gpt-oss-120b"]
# USD per 1M tokens (input, output). Unlisted models fall back to providers.DEFAULT_PRICES.
PRICES = {"Qwen/Qwen3-235B-A22B-Instruct-2507": (0.20, 0.60), "deepseek-ai/DeepSeek-V4-Flash": (0.14, 0.28)}
MAX_TOKENS = 512
MAX_API_ERRORS = 3

TOOLS = [
    {"name": "inspect_column", "description": "Sample values and target rate for one column.",
     "input_schema": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "run_experiments", "description": "Cross-validate a list of candidate models.",
     "input_schema": {"type": "object", "properties": {
         "task": {"type": "string", "enum": ["classification", "regression"]},
         "primary_metric": {"type": "string", "enum": ["roc_auc", "f1_macro", "accuracy", "rmse", "mae", "r2"]},
         "candidates": {"type": "array", "items": {"type": "object", "properties": {
             "name": {"type": "string"},
             "model": {"type": "string", "enum": ["logreg", "ridge", "random_forest", "lightgbm", "xgboost"]},
             "params": {"type": "object"}}, "required": ["name", "model"]}}},
         "required": ["task", "primary_metric", "candidates"]}},
    {"name": "write_report", "description": "Save the final report. Call once, at the end.",
     "input_schema": {"type": "object", "properties": {"markdown": {"type": "string"},
                                                       "spoken_summary": {"type": "string"}},
                      "required": ["markdown", "spoken_summary"]}},
]
SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}
OPENAI_TOOLS = to_openai_tools(TOOLS)
SYSTEM = "You operate an ML pipeline. Answer every request with exactly one tool call and no other text."

BASE_PROMPTS = [
    ("Show me sample values of the column 'tenure'.", "inspect_column"),
    ("The column refund_issued looks too good. Check it for leakage.", "inspect_column"),
    ("Is 'customer_id' just an identifier? Look at it.", "inspect_column"),
    ("Cross-validate a logistic regression baseline and a LightGBM with 300 trees. Binary target, use roc_auc.",
     "run_experiments"),
    ("Regression on house prices: compare ridge and a random forest with max_depth 8, metric rmse.",
     "run_experiments"),
    ("The classes are imbalanced. Try xgboost with scale_pos_weight 5 and lightgbm with class_weight balanced, "
     "scored by f1_macro.", "run_experiments"),
    ("Run three random forests with 100, 300 and 500 trees on the classification task, metric accuracy.",
     "run_experiments"),
    ("We are done. Save the report '# Result\\nLightGBM wins at 0.91 AUC.' with the spoken summary "
     "'LightGBM is the best model.'", "write_report"),
    ("Write the final report: markdown '# Summary\\nRidge is enough.' and a one-line spoken summary.",
     "write_report"),
    ("Finish the run: report that no model beat the baseline, and give a short summary for text to speech.",
     "write_report"),
]


def build_prompts(n: int) -> list[tuple[str, str]]:
    """n (prompt, expected_tool) pairs, cycling the base set."""
    return [BASE_PROMPTS[i % len(BASE_PROMPTS)] for i in range(n)]


def _schema_error(value, schema: dict, where: str = "args") -> str | None:
    """First nested type, enum, or required-key violation, or None."""
    if "enum" in schema and value not in schema["enum"]:
        return f"{where}: {value!r} not in {schema['enum']}"
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return f"{where}: expected object"
        missing = [k for k in schema.get("required", []) if k not in value]
        if missing:
            return f"{where}: missing {missing}"
        for k, sub in schema.get("properties", {}).items():
            if k in value and (err := _schema_error(value[k], sub, f"{where}.{k}")):
                return err
    elif kind == "array":
        if not isinstance(value, list):
            return f"{where}: expected array"
        for i, item in enumerate(value):
            if err := _schema_error(item, schema.get("items", {}), f"{where}[{i}]"):
                return err
    elif kind == "string" and not isinstance(value, str):
        return f"{where}: expected string"
    return None


def score_call(message, expected: str) -> dict:
    """Score one assistant message: tool used, schema-valid arguments, correct tool chosen."""
    calls = getattr(message, "tool_calls", None) or []
    if not calls:
        return {"tool": None, "valid": False, "correct": False, "error": "no tool call"}
    name, raw = calls[0].function.name, calls[0].function.arguments
    if name not in SCHEMAS:
        return {"tool": name, "valid": False, "correct": False, "error": f"unknown tool {name}"}
    args, err = parse_arguments(raw, SCHEMAS[name])
    if err is None:
        err = _schema_error(args, SCHEMAS[name])
    return {"tool": name, "valid": err is None, "correct": name == expected, "error": err}


def summarize(rows: list[dict], price_in: float, price_out: float) -> dict:
    """Aggregate per-prompt rows into rates, mean latency, tokens, and estimated USD."""
    n = len(rows)
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    tin = sum(r.get("prompt_tokens", 0) for r in rows)
    tout = sum(r.get("completion_tokens", 0) for r in rows)
    rate = lambda pred: round(sum(1 for r in rows if pred(r)) / n, 4) if n else 0.0  # noqa: E731
    return {"n": n, "valid_rate": rate(lambda r: r["valid"]), "correct_rate": rate(lambda r: r["correct"]),
            "ok_rate": rate(lambda r: r["valid"] and r["correct"]),
            "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
            "api_errors": sum(1 for r in rows if r.get("api_error")),
            "prompt_tokens": tin, "completion_tokens": tout,
            "est_usd": (tin * price_in + tout * price_out) / 1e6}


def probe_model(client, model: str, prompts: list[tuple[str, str]]) -> dict:
    """Run every prompt against one model through an OpenAI-shaped client; return the summary plus rows."""
    rows, api_errors = [], 0
    for prompt, expected in prompts:
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0, max_tokens=MAX_TOKENS, tools=OPENAI_TOOLS, tool_choice="auto",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        except Exception as e:
            api_errors += 1
            rows.append({"expected": expected, "tool": None, "valid": False, "correct": False, "latency_s": None,
                         "api_error": f"{type(e).__name__}: {e}"[:300]})
            if api_errors >= MAX_API_ERRORS:
                print(f"  {model}: {api_errors} API errors, skipping the rest")
                break
            continue
        row = {"expected": expected, **score_call(resp.choices[0].message, expected),
               "latency_s": round(time.perf_counter() - t0, 3),
               "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
               "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0}
        rows.append(row)
    price_in, price_out = PRICES.get(model, DEFAULT_PRICES)
    return {"model": model, **summarize(rows, price_in, price_out),
            "price_known": model in PRICES, "rows": rows}


def rank(summaries: list[dict]) -> list[dict]:
    """Best first: highest valid-and-correct rate, then lowest mean latency."""
    return sorted(summaries, key=lambda s: (-s["ok_rate"], s["mean_latency_s"] or float("inf")))


def check_available(client, models: list[str]) -> list[str]:
    """Print which requested ids the endpoint lists, with close matches for missing ones.
    Returns the ids to probe (all of them if the listing fails)."""
    try:
        available = [m.id for m in client.models.list()]
    except Exception as e:
        print(f"GET /v1/models failed ({type(e).__name__}: {e}); probing every requested id")
        return models
    keep = []
    for m in models:
        if m in available:
            print(f"  found    {m}")
            keep.append(m)
        else:
            close = difflib.get_close_matches(m, available, n=3, cutoff=0.4)
            print(f"  MISSING  {m}  close matches: {close or 'none'}")
    return keep


def print_table(ranked: list[dict]) -> None:
    head = f"{'#':>2}  {'model':<40} {'ok':>6} {'valid':>6} {'tool':>6} {'lat_s':>6} {'tok_in':>7} {'tok_out':>7} {'usd':>9}"
    print(head)
    print("-" * len(head))
    for i, s in enumerate(ranked, 1):
        lat = f"{s['mean_latency_s']:.2f}" if s["mean_latency_s"] is not None else "n/a"
        usd = f"{s['est_usd']:.5f}" + ("" if s["price_known"] else "*")
        print(f"{i:>2}  {s['model']:<40} {s['ok_rate']:>6.2f} {s['valid_rate']:>6.2f} {s['correct_rate']:>6.2f} "
              f"{lat:>6} {s['prompt_tokens']:>7} {s['completion_tokens']:>7} {usd:>9}")
    print("ok = valid and correct tool; * = default price, not model-specific")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=20, help="prompts per model")
    args = ap.parse_args()

    client = NebiusClient().client
    print("Checking model ids:")
    models = check_available(client, args.models)
    prompts = build_prompts(args.n)
    summaries = []
    for m in models:
        print(f"Probing {m} ({len(prompts)} prompts)...")
        summaries.append(probe_model(client, m, prompts))
    ranked = rank(summaries)
    print()
    print_table(ranked)

    os.makedirs("runs", exist_ok=True)
    out = f"runs/probe_{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"requested": args.models, "probed": models, "n_prompts": len(prompts), "ranked": ranked},
                  f, indent=2, default=str)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
