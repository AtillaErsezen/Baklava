"""Final report: every number comes from harness state; LLM prose is claim-checked before it lands."""
import html
import json
import re

REMOVED = "[number removed: not in tool results]"
WHITELIST = {1.0, 2.0}  # "1-SE rule", "one of 2": too common to demand a source
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(%?)")
_NEXT_WORD = re.compile(r"\s*([A-Za-z_]+)")


def _numbers(obj, key: str = "", out: dict | None = None) -> dict[str, set[float]]:
    """Map each state key to the numbers directly under it (scalars, scalar lists, list lengths); '' holds all."""
    out = {"": set()} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _numbers(v, str(k).lower(), out)
    elif isinstance(obj, (list, tuple)):
        vals = {float(len(obj))} | {float(v) for v in obj if _is_num(v)}
        out.setdefault(key, set()).update(vals)
        out[""].update(vals)
        for v in obj:
            if isinstance(v, (dict, list, tuple)):
                _numbers(v, key, out)
    elif _is_num(obj):
        out.setdefault(key, set()).add(float(obj))
        out[""].add(float(obj))
    return out


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _expand(vals: set[float], percent: bool) -> set[float]:
    """Values plus their 2/3/4 decimal roundings; percent prose also matches 100x a fraction."""
    base = {abs(v) for v in vals}
    if percent:
        base |= {v * 100 for v in base if v <= 1}
    return base | {round(v, d) for v in base for d in (0, 1, 2, 3, 4)}


def _close(x: float, cands: set[float]) -> bool:
    return any(abs(x - c) <= max(0.001, 0.005 * max(abs(x), abs(c))) for c in cands)


def _keyed(word: str, index: dict[str, set[float]]) -> set[float] | None:
    """Numbers under state keys named like the word after the number ("9 features" -> dataset.features)."""
    w = word.lower()
    forms = {w, w[:-1]} if w.endswith("s") and len(w) > 3 else {w}
    hits = [v for k, v in index.items() if k and any(k == f or k.endswith("_" + f) for f in forms)]
    return set().union(*hits) if hits else None


def claim_check(text: str, state: dict) -> tuple[str, list[str]]:
    """Replace every number in LLM prose that no state value supports; return (clean_text, removed)."""
    index, removed = _numbers(state), []

    def fix(m: re.Match) -> str:
        x, pct = float(m.group(1).replace(",", "")), bool(m.group(2))
        nxt = _NEXT_WORD.match(text, m.end())
        pool = (nxt and _keyed(nxt.group(1), index)) or index[""]
        if (x in WHITELIST and not pct) or _close(x, _expand(pool, pct)):
            return m.group(0)
        removed.append(m.group(0))
        return REMOVED

    return _NUM.sub(fix, text), removed


PARAM_PRIORITY = ("n_estimators", "learning_rate", "max_depth", "num_leaves", "C", "alpha", "hidden_layer_sizes")


def _nd(text: str) -> str:
    """No em or en dashes, ever."""
    return text.replace(chr(0x2014), "-").replace(chr(0x2013), "-")


def _line(v, n: int = 600) -> str:
    """Untrusted text as one escaped line: no headings, no HTML, no dashes."""
    return _nd(html.escape(re.sub(r"\s+", " ", str(v)).strip()[:n], quote=False))


def _cell(v) -> str:
    return _line(_n(v) if _is_num(v) else v, 200).replace("|", "\\|")


def _n(v) -> str:
    """Numbers as the harness stored them: ints verbatim, floats to 4 significant digits."""
    if v is None:
        return "n/a"
    if isinstance(v, float) and abs(v) < 1000:
        return f"{v:.4g}"
    return str(round(v)) if isinstance(v, float) else str(v)


def _params(p: dict | None) -> str:
    """At most 4 params, the usual important ones first, floats shortened."""
    p = p or {}
    keys = [k for k in PARAM_PRIORITY if k in p] + [k for k in p if k not in PARAM_PRIORITY]
    return ", ".join(f"{k[:18]}={f'{p[k]:.3g}' if isinstance(p[k], float) else p[k]}" for k in keys[:4])


def _dataset(ds: dict, split: dict) -> list[str]:
    rows = [f"- {label}: {_line(ds[k])}" for k, label in (("name", "Name"), ("rows", "Rows"), ("features", "Features"),
            ("target", "Target"), ("task", "Task"), ("purpose", "Purpose")) if ds.get(k) is not None]
    if split:
        rows.append(f"- Split: dev {_n(split.get('dev'))}, search validation {_n(split.get('search_val'))}, hidden "
                    f"(locked until the final test) {_n(split.get('hidden_locked'))}"
                    + (f"; {_line(split['method'])}" if split.get("method") else ""))
    return ["## Dataset card", "", *rows, ""]


def _handled(f: dict, dropped: list) -> str | None:
    """The drop that answers a finding, if any."""
    hit = [c for c in dropped if str(c) in str(f.get("finding", ""))]
    return f"dropped {', '.join(map(str, hit))}" if hit else None


def _issues(findings: list, dropped: list) -> list[str]:
    out = ["## Issues found and fixes", "", "| Check | Severity | Finding | Fix |", "|---|---|---|---|"]
    out += [f"| {_cell(f.get('check'))} | {_cell(f.get('severity'))} | {_cell(f.get('finding'))} | "
            f"{_cell(_handled(f, dropped) or 'none recorded')} |" for f in findings]
    if dropped:
        out.append(f"\nDropped columns: {_line(', '.join(map(str, dropped)))}.")
    return out + [""]


def _funnel(search: dict, confirm: dict) -> list[str]:
    steps = [f"{_n(search['space_size'])} pipelines"] if "space_size" in search else []
    steps += [f"{_n(search['raced'])} raced"] if "raced" in search else []
    steps += [_n(r.get("survivors")) for r in search.get("rungs") or []]
    if confirm.get("table"):
        steps.append(f"top {len(confirm['table'])}")
    extra = f"Model fits: {_n(search['fits'])}; stopped: {_line(search.get('stopped'))}." if "fits" in search else ""
    return ["## Search funnel", "", " -> ".join(steps), "", *([extra, ""] if extra else [])]


def _top_table(confirm: dict) -> list[str]:
    pm = _cell(confirm.get("primary_metric", "metric"))
    out = ["## Top models", "", f"| Model | Key params | {pm} [95% CI] | p vs best | Hidden | Fit s | Predict ms | "
           "Big-O train | Pareto |", "|---|---|---|---|---|---|---|---|---|"]
    for r in confirm["table"]:
        ci = r.get("ci") or [None, None]
        p = "best" if r.get("p_vs_best") is None else _n(r["p_vs_best"])
        out.append(f"| {_cell(r.get('name'))} | {_cell(_params(r.get('params')))} | "
                   f"{_cell(f'{_n(r.get('cv_mean'))} [{_n(ci[0])}, {_n(ci[1])}]')} | {_cell(p)} | "
                   f"{_cell(_n(r.get('hidden')))} | {_cell(_n(r.get('fit_s')))} | {_cell(_n(r.get('predict_ms')))} | "
                   f"{_cell((r.get('big_o') or {}).get('train') or 'n/a')} | {'yes' if r.get('pareto') else 'no'} |")
    return out + [""]


def _recommendation(state: dict, why: str) -> list[str]:
    confirm, final = state.get("confirm") or {}, state.get("final") or {}
    rec = confirm.get("recommendation") or {}
    pick = final.get("name") or rec.get("pick")
    out = ["## Recommendation", "", f"Pick: **{_line(pick)}**.", ""] + ([why, ""] if why else [])
    tied = next((g for g in confirm.get("tie_groups") or [] if pick in g and len(g) > 1), [])
    if tied:
        out += [f"Note: {_line(pick)} is statistically tied with {_line(', '.join(n for n in tied if n != pick))}; "
                "chosen by the 1-SE rule.", ""]
    if rec.get("best") and rec["best"] != pick:
        out += [f"Best raw CV score: {_line(rec['best'])}.", ""]
    feats = [str(f.get("feature", "")).split("__")[-1] for f in final.get("top_features") or []]
    return out + ([f"Top features: {_line(', '.join(feats))}.", ""] if feats else [])


def _external(trials: list) -> list[str]:
    rows = [f"- {_line(t.get('source'))}: {_line(t.get('verdict'))}, delta {_n(t.get('delta'))} "
            f"(95% CI {_n((t.get('ci') or [None, None])[0])} to {_n((t.get('ci') or [None, None])[1])})" for t in trials]
    return ["## External data", "", *(rows or ["Not tested."]), ""]


def _auto_caveats(state: dict) -> list[str]:
    """Caveats the harness can prove: dev-to-hidden gap beyond 2 std, ECE above 0.1, unhandled severity 3."""
    out = []
    for r in (state.get("confirm") or {}).get("table") or []:
        gap, ci = r.get("dev_to_hidden_gap"), r.get("ci")
        std = r.get("std") or ((ci[1] - ci[0]) / (2 * 1.96) if ci and None not in ci[:2] else None)
        if gap is not None and std and abs(gap) > 2 * std:
            out.append(f"{r['name']}: hidden score {_n(r.get('hidden'))} is {_n(gap)} away from CV {_n(r.get('cv_mean'))},"
                       " more than 2 standard deviations; treat the CV estimate as optimistic.")
        if r.get("ece") is not None and r["ece"] > 0.1:
            out.append(f"{r['name']}: calibration error (ECE) {_n(r['ece'])} is above 0.1; calibrate before "
                       "reading outputs as probabilities.")
    dropped = state.get("dropped_columns") or []
    out += [f"Severity 3 finding not handled ({f.get('check')}): {f.get('finding')}"
            for f in state.get("findings") or [] if f.get("severity") == 3 and not _handled(f, dropped)]
    return out


def build_report(state: dict, prose: dict) -> dict:
    """Deterministic markdown from harness state plus claim-checked LLM prose."""
    removed: list[str] = []

    def checked(text) -> str:
        clean, gone = claim_check(_nd(str(text or "")), state)
        removed.extend(gone)
        return clean

    why = _line(checked(prose.get("why")))
    caveats = [_line(checked(c)) for c in prose.get("caveats") or []]
    spoken = " ".join(_nd(checked(prose.get("spoken_summary"))).split()[:60])
    ds, confirm, search = state.get("dataset") or {}, state.get("confirm") or {}, state.get("search") or {}
    md = [f"# Model report: {_line(ds.get('name', 'dataset'))}", ""]
    if ds or state.get("split"):
        md += _dataset(ds, state.get("split") or {})
    if state.get("findings") or state.get("dropped_columns"):
        md += _issues(state.get("findings") or [], state.get("dropped_columns") or [])
    if search:
        md += _funnel(search, confirm)
    if confirm.get("table"):
        md += _top_table(confirm)
    if confirm.get("recommendation") or state.get("final"):
        md += _recommendation(state, why)
    if search.get("n_star") is not None:
        md += ["## Sample size", "", f"n* = {_n(search['n_star'])} training rows. {_line(search.get('justification', ''))}", ""]
    md += _external(state.get("external") or [])
    cav = [f"- {c}" for c in caveats + [_line(a) for a in _auto_caveats(state)]]
    md += ["## Caveats", "", *(cav or ["- None recorded."]), ""]
    if (state.get("export") or {}).get("script"):
        ex = state["export"]
        md += ["## How to retrain", "", f"    python {_line(ex['script'])} data.csv --target {_line(ds.get('target', 'TARGET'))}"
               " [--out model.joblib] [--cv K]", "", "data.csv is your CSV or Parquet file; --out defaults to model.joblib; "
               "--cv K sets the CV folds (default: the agent's). The script prints the CV score, then fits on all rows.",
               "", f"Params: {_line(ex.get('params', 'n/a'))}. Model card: {_line(ex.get('card', 'n/a'))}.", "",
               "The delivered model was refit on all rows (dev + search validation + hidden) after the evaluation, so the "
               "hidden score measured the same pipeline trained on the dev split: a close, slightly conservative proxy for "
               "the delivered model, not a held-out estimate of that exact fit.", ""]
    if state.get("usage"):
        u = state["usage"]
        modal = f" Modal compute: ${_n(u['modal_usd'])}." if u.get("modal_usd") is not None else ""
        md += ["## Cost", "", f"Tokens: {_n(u.get('input_tokens'))} input, {_n(u.get('output_tokens'))} output. "
               f"LLM cost: ${_n(u.get('usd'))}.{modal}", ""]
    md.append(f"Claim check: removed {len(removed)} unsupported number(s) from the written text: {', '.join(removed)}."
              if removed else "Claim check: every number in the written text matches a tool result.")
    return {"markdown": _nd("\n".join(md)) + "\n", "spoken_summary": spoken}


REPORT_PROSE_SCHEMA = {
    "type": "object",
    "properties": {"why": {"type": "string", "maxLength": 600},
                   "caveats": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                   "spoken_summary": {"type": "string"}},
    "required": ["why", "caveats", "spoken_summary"],
    "additionalProperties": False,
}
PROSE_SYSTEM = ("You write three fields of an AutoML model report from the JSON the user sends. Use only numbers that "
                "appear in that JSON; the harness deletes any other number. why: why the pick fits the stated purpose, "
                "at most 600 characters. caveats: at most 5 concrete risks a user must know. spoken_summary: at most 60 "
                "words of plain speech with the headline metric only. Active voice, plain words, no filler, no em dashes.")


def json_call(client, system: str, user: str, name: str, schema: dict, max_tokens: int) -> dict | None:
    """One schema-constrained completion with no tools (Nebius rejects both together); None on any failure."""
    if getattr(client, "client", None) is None:
        return None
    try:
        resp = client.client.chat.completions.create(
            model=client.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
            temperature=0, max_tokens=max_tokens)
        client.ledger.add(resp.usage)
        out = json.loads(resp.choices[0].message.content)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def fallback_prose(state: dict) -> dict:
    """Deterministic prose from state, used when the LLM call fails or no LLM is attached."""
    confirm, final = state.get("confirm") or {}, state.get("final") or {}
    pick = final.get("name") or (confirm.get("recommendation") or {}).get("pick")
    row = next((r for r in confirm.get("table") or [] if r.get("name") == pick), None)
    if not pick or not row:
        return {"why": "No model was confirmed in this run.", "caveats": [],
                "spoken_summary": "The run finished without a confirmed model."}
    pm = confirm.get("primary_metric", "the primary metric")
    why = f"{pick} has a cross-validated {pm} of {_n(row.get('cv_mean'))}"
    why += f" and scored {_n(row['hidden'])} on the locked hidden rows." if row.get("hidden") is not None else "."
    tied = next((g for g in confirm.get("tie_groups") or [] if pick in g and len(g) > 1), [])
    if tied:
        why += f" It ties statistically with {', '.join(n for n in tied if n != pick)}; the 1-SE rule chose it."
    score = row.get("hidden") if row.get("hidden") is not None else row.get("cv_mean")
    return {"why": why, "caveats": [], "spoken_summary": f"I recommend {pick}. It scores {_n(score)} {pm} on held-out data."}


def write_prose(client, state: dict) -> dict:
    """LLM prose for the report (why, caveats, spoken_summary); never raises, falls back to state-built prose."""
    out = json_call(client, PROSE_SYSTEM, json.dumps(state, separators=(",", ":"), default=str),
                    "report_prose", REPORT_PROSE_SCHEMA, 700)
    ok = (out and isinstance(out.get("why"), str) and isinstance(out.get("spoken_summary"), str)
          and isinstance(out.get("caveats"), list) and all(isinstance(c, str) for c in out["caveats"]))
    if not ok:
        return fallback_prose(state)
    return {"why": out["why"][:600], "caveats": out["caveats"][:5], "spoken_summary": out["spoken_summary"]}
