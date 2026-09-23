export type RecordData = Record<string, unknown>;
export type RunEvent = {
  run_id: string;
  ts: number;
  kind: string;
  payload: RecordData;
};
export type Model = {
  name: string;
  family: string;
  score?: number;
  std?: number;
  ci?: [number, number];
  holdout?: number;
  fit?: number;
  predict?: number;
  params: RecordData;
  error?: string;
};
export type Run = {
  id: string;
  label: string;
  dataset: string;
  target: string;
  task: string;
  rows?: number;
  features?: number;
  metric: string;
  models: Model[];
  winner?: string;
  complete: boolean;
  sample: boolean;
  importance: { feature: string; importance: number }[];
  findings: { check: string; severity: number; finding: string }[];
  report: string;
  summary: string;
  events: RunEvent[];
  duration?: number;
  cost?: number;
  searched?: number;
  confirmed: boolean;
  modelPath?: string;
};
const record = (v: unknown): RecordData =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as RecordData) : {};
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const str = (v: unknown, fallback = ""): string =>
  typeof v === "string" ? v : fallback;
const num = (v: unknown): number | undefined =>
  typeof v === "number" && Number.isFinite(v) ? v : undefined;
export const lowerIsBetter = (metric: string) =>
  ["rmse", "mae", "mse", "log_loss"].includes(metric.toLowerCase());
export const metricLabel = (metric: string) =>
  ({
    roc_auc: "ROC AUC",
    r2: "R²",
    f1_macro: "F1 macro",
    accuracy: "Accuracy",
  })[metric] || metric.toUpperCase();
export const familyLabel = (family: string) =>
  ({
    lightgbm: "LightGBM",
    xgboost: "XGBoost",
    random_forest: "Random Forest",
    logreg: "Logistic Regression",
    ridge: "Ridge",
    mlp: "Neural network",
  })[family] || family.replaceAll("_", " ");
export const scoreLabel = (v?: number) =>
  v === undefined
    ? "—"
    : Math.abs(v) >= 100
      ? v.toLocaleString("en-US", { maximumFractionDigits: 2 })
      : v.toFixed(3);
export function rankModels(models: Model[], metric: string): Model[] {
  return [...models].sort((a, b) =>
    a.score === undefined
      ? b.score === undefined
        ? 0
        : 1
      : b.score === undefined
        ? -1
        : (lowerIsBetter(metric) ? 1 : -1) * (a.score - b.score),
  );
}
function toModel(value: unknown, metric: string): Model {
  const r = record(value),
    m = record(record(r.metrics)[metric]);
  const ci = list(r.ci);
  return {
    name: str(r.name, "Unnamed candidate"),
    family: str(r.family, str(r.model, str(r.name))),
    score: num(r.cv_mean) ?? num(r.mean) ?? num(r[metric]) ?? num(m.mean),
    std: num(r.std) ?? num(r.cv_std) ?? num(m.std),
    ci:
      ci.length === 2 && ci.every((v) => num(v) !== undefined)
        ? (ci as [number, number])
        : undefined,
    holdout: num(r.hidden) ?? num(r.hidden_score),
    fit: num(r.fit_s) ?? num(r.fit_seconds),
    predict: num(r.predict_ms),
    params: record(r.params),
    error: str(r.error) || undefined,
  };
}
export function parseRun(text: string, filename = "Imported run"): Run {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    try {
      parsed = text
        .split(/\r?\n/)
        .filter((line) => line.trim())
        .map((line) => JSON.parse(line));
    } catch {
      throw new Error(
        "This file is not valid JSON or JSONL. Choose a Baklava results or events file.",
      );
    }
  }
  const source = record(parsed);
  const rawEvents = Array.isArray(parsed) ? parsed : list(source.events);
  if (
    rawEvents.some(
      (e) =>
        typeof record(e).kind !== "string" ||
        !record(e).payload ||
        typeof record(e).payload !== "object",
    )
  ) {
    throw new Error(
      "Each event needs a kind and payload. Choose a Baklava event export.",
    );
  }
  const ids = new Set(
    rawEvents.map((e) => str(record(e).run_id)).filter(Boolean),
  );
  if (ids.size > 1)
    throw new Error(
      "This file contains multiple runs. Import one run at a time.",
    );
  const events: RunEvent[] = rawEvents
    .map((e) => {
      const r = record(e);
      return {
        run_id: str(r.run_id),
        ts: num(r.ts) ?? 0,
        kind: str(r.kind),
        payload: record(r.payload),
      };
    })
    .sort((a, b) => a.ts - b.ts);
  if (!events.length && !(source.profile && Array.isArray(source.results)))
    throw new Error(
      "No Baklava results found. Choose a *_events.jsonl or *_results.json file.",
    );
  const latest = (kind: string) =>
    events.findLast((e) => e.kind === kind)?.payload ?? {};
  const start = latest("run_start"),
    confirm = latest("confirm"),
    board = latest("leaderboard");
  const final = events.length ? latest("final_model") : record(source.final);
  const spec = record(final.spec),
    profile = record(source.profile),
    target = record(profile.target);
  const metric = str(
    confirm.primary_metric,
    str(
      board.primary_metric,
      str(
        spec.primary_metric,
        Object.keys(record(record(list(source.results)[0]).metrics))[0] ||
          "roc_auc",
      ),
    ),
  );
  const candidates = Object.keys(confirm).length
    ? list(confirm.table)
    : events.length
      ? list(board.rows)
      : list(source.results);
  const models = rankModels(
    candidates.map((row) => toModel(row, metric)),
    metric,
  );
  const specs = events
    .filter((e) => e.kind === "round_start")
    .flatMap((e) => list(e.payload.candidates))
    .map(record);
  for (const model of models) {
    const modelSpec =
      specs.find((s) => s.name === model.name) ??
      (model.name === final.name ? spec : undefined);
    if (modelSpec) {
      model.family = str(modelSpec.model, model.family);
      if (!Object.keys(model.params).length)
        model.params = record(modelSpec.params);
    }
  }
  const id =
    events[0]?.run_id || filename.replace(/_(events|results)\.jsonl?$/, "");
  const dataset = str(
    start.dataset,
    str(profile.dataset, filename.replace(/_(events|results)\.jsonl?$/, "")),
  );
  const report = latest("report"),
    usage = latest("usage");
  const tracking = record(record(start._tracking).run);
  const task = str(
    spec.task,
    str(
      tracking.task,
      str(
        target.suggested_task,
        ["rmse", "mae", "r2", "mse"].includes(metric)
          ? "regression"
          : "classification",
      ),
    ),
  );
  const importance = list(final.top_features)
    .map(record)
    .filter((r) => num(r.importance) !== undefined)
    .map((r) => ({ feature: str(r.feature), importance: num(r.importance)! }))
    .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
  return {
    id,
    dataset,
    label: dataset,
    target: str(
      start.target,
      str(spec.target, str(target.name, "Not recorded")),
    ),
    task,
    rows: num(start.rows) ?? num(profile.n_rows),
    features: num(start.features) ?? num(profile.n_features),
    metric,
    models,
    winner:
      str(final.name, str(record(confirm.recommendation).pick)) || undefined,
    complete: !!final.name,
    sample: false,
    importance,
    findings: list(latest("diagnostics").findings)
      .map(record)
      .map((f) => ({
        check: str(f.check),
        severity: num(f.severity) ?? 1,
        finding: str(f.finding),
      }))
      .sort((a, b) => b.severity - a.severity),
    report: str(report.markdown),
    summary: str(report.spoken_summary),
    events,
    duration:
      events.length > 1 && events[0].ts > 0
        ? events[events.length - 1].ts - events[0].ts
        : undefined,
    cost: num(usage.usd),
    searched: num(latest("search_plan").space),
    confirmed: !!confirm.table,
    modelPath: str(final.model_path) || undefined,
  };
}
export function reportMarkdown(run: Run): string {
  return `# Baklava · ${run.label}\n\n${run.sample ? "Illustrative sample. No model was trained for these results.\n\n" : ""}Dataset: ${run.dataset}\nTarget: ${run.target}\nPrimary metric: ${metricLabel(run.metric)} (${lowerIsBetter(run.metric) ? "lower" : "higher"} is better)\n\n${run.report || "No agent report was recorded for this run."}\n\n## Model comparison\n\n| Candidate | CV score | Holdout | Fit seconds |\n| --- | --- | --- | --- |\n${run.models.map((m) => `| ${m.name.replaceAll("|", "\\|")} | ${scoreLabel(m.score)} | ${scoreLabel(m.holdout)} | ${m.fit ?? "—"} |`).join("\n")}\n\n## Diagnostics\n\n${run.findings.map((f) => `- ${f.finding}`).join("\n") || "No diagnostics recorded."}\n`;
}
