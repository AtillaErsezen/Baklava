import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, symlink, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { parseRun, rankModels, reportMarkdown } from "../src/results/data.ts";
import { localRuns } from "../server/runs.ts";
const event = (kind: string, payload: Record<string, unknown>, ts = 1) => ({
  run_id: "test-run",
  ts,
  kind,
  payload,
});
test("confirmation uses the recommendation, not the highest score", () => {
  const run = parseRun(
    JSON.stringify([
      event("run_start", {
        dataset: "test.csv",
        target: "churn",
        rows: 200,
        features: 4,
      }),
      event("confirm", {
        primary_metric: "roc_auc",
        recommendation: { pick: "simple" },
        table: [
          {
            name: "complex",
            family: "xgboost",
            cv_mean: 0.95,
            hidden: 0.9,
            ci: [0.92, 0.97],
          },
          {
            name: "simple",
            family: "logreg",
            cv_mean: 0.94,
            hidden: 0.91,
            fit_s: 0,
            params: { C: 1 },
          },
        ],
      }),
    ]),
  );
  assert.equal(run.winner, "simple");
  assert.equal(run.models[0].name, "complex");
  assert.equal(run.models[1].fit, 0);
  assert.equal(run.models[1].std, undefined);
  assert.equal(run.models[1].ci, undefined);
  assert.equal(run.complete, false);
  assert.equal(run.rows, 200);
});
test("JSONL accepts extra fields, orders events, and preserves null/missing values", () => {
  const run = parseRun(
    [
      event(
        "final_model",
        {
          name: "lgbm",
          top_features: [{ feature: "tenure", importance: 0.7 }],
        },
        4,
      ),
      event("run_start", { dataset: "d.csv", target: "y" }, 1),
      event(
        "leaderboard",
        {
          primary_metric: "roc_auc",
          rows: [
            { name: "lgbm", roc_auc: 0.8, std: 0.1, fit_seconds: 2 },
            { name: "broken", error: "Training failed" },
          ],
        },
        3,
      ),
    ]
      .map((e) => JSON.stringify({ ...e, extra: true }))
      .join("\n"),
  );
  assert.equal(run.complete, true);
  assert.equal(run.duration, 3);
  assert.equal(run.models[0].score, 0.8);
  assert.equal(run.models[1].score, undefined);
  assert.equal(run.models[1].error, "Training failed");
  assert.equal(run.importance[0].feature, "tenure");
  assert.equal(run.cost, undefined);
});
test("regression snapshot picks the metric from the final spec and sorts ascending", () => {
  const run = parseRun(
    JSON.stringify({
      profile: {
        n_rows: 2000,
        n_features: 6,
        target: { name: "price", suggested_task: "regression" },
      },
      results: [
        {
          name: "forest",
          metrics: { r2: { mean: 0.7 }, rmse: { mean: 35000 } },
        },
        {
          name: "xgb",
          metrics: { rmse: { mean: 28000, std: 1000 } },
          fit_seconds: 2,
        },
      ],
      final: {
        name: "xgb",
        spec: {
          model: "xgboost",
          primary_metric: "rmse",
          target: "price",
          task: "regression",
        },
      },
    }),
    "test_results.json",
  );
  assert.equal(run.metric, "rmse");
  assert.equal(run.models[0].name, "xgb");
  assert.equal(run.models[0].score, 28000);
  assert.equal(run.target, "price");
  assert.equal(run.rows, 2000);
  assert.equal(run.models[0].family, "xgboost");
  assert.match(reportMarkdown(run), /lower is better/);
});
test("all metric directions and missing scores", () => {
  const models = [
    { name: "a", family: "ridge", params: {}, score: 30 },
    { name: "b", family: "ridge", params: {}, score: 10 },
    { name: "failed", family: "ridge", params: {} },
  ];
  for (const metric of ["rmse", "mae", "mse", "log_loss"])
    assert.deepEqual(
      rankModels(models, metric).map((m) => m.name),
      ["b", "a", "failed"],
    );
  assert.deepEqual(
    rankModels(models, "r2").map((m) => m.name),
    ["a", "b", "failed"],
  );
});
test("rejects unrelated, malformed, empty, and mixed-run files", () => {
  for (const value of [
    "oops",
    "{}",
    "[]",
    JSON.stringify([{ name: "not an event" }]),
    JSON.stringify([
      event("report", {}),
      { ...event("report", {}), run_id: "another" },
    ]),
  ])
    assert.throws(() => parseRun(value));
});
test("report keeps sample disclaimer and does not invent measurements", () => {
  const run = parseRun(
    JSON.stringify([
      event("report", { markdown: "# Test report\nPlain text" }),
    ]),
  );
  run.sample = true;
  assert.match(reportMarkdown(run), /Illustrative sample/);
  assert.match(reportMarkdown(run), /Plain text/);
  assert.match(reportMarkdown(run), /No diagnostics recorded/);
  assert.equal(run.models.length, 0);
});
test("local API only serves allowed result files and rejects symlinks, writes, and traversal", async () => {
  const dir = await mkdtemp(join(tmpdir(), "baklava-results-"));
  try {
    await writeFile(
      join(dir, "test_events.jsonl"),
      JSON.stringify(event("run_start", {})),
    );
    await writeFile(join(dir, "test_results.json"), "{}");
    await writeFile(join(dir, "schema-contract-before.json"), "private");
    await symlink(
      join(dir, "schema-contract-before.json"),
      join(dir, "link_events.jsonl"),
    );
    let middleware: any;
    const plugin = localRuns(dir);
    (plugin.configureServer as Function)({
      middlewares: {
        use(fn: unknown) {
          middleware = fn;
        },
      },
    });
    const request = async (url: string, method = "GET") => {
      let body = "",
        statusCode = 200;
      const res = {
        setHeader() {},
        get statusCode() {
          return statusCode;
        },
        set statusCode(v) {
          statusCode = v;
        },
        end(v: string) {
          body = v;
        },
      };
      await middleware({ url, method }, res, () => {});
      return { statusCode, data: JSON.parse(body) };
    };
    const index = await request("/api/results");
    assert.equal(index.statusCode, 200);
    assert.ok(index.data.some((r: any) => r.file === "test_events.jsonl"));
    assert.ok(!index.data.some((r: any) => r.file === "test_results.json"));
    assert.ok(
      !index.data.some((r: any) => r.file === "schema-contract-before.json"),
    );
    assert.equal(
      (await request("/api/results/test_events.jsonl")).data.content,
      JSON.stringify(event("run_start", {})),
    );
    assert.equal(
      (await request("/api/results/schema-contract-before.json")).statusCode,
      404,
    );
    assert.equal(
      (await request("/api/results/%2e%2e%2fschema-contract-before.json"))
        .statusCode,
      404,
    );
    assert.equal(
      (await request("/api/results/link_events.jsonl")).statusCode,
      413,
    );
    assert.equal((await request("/api/results", "POST")).statusCode, 405);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
test("local API flags and serves only runs with a saved model zip", async () => {
  const dir = await mkdtemp(join(tmpdir(), "baklava-models-"));
  try {
    await writeFile(join(dir, "a_events.jsonl"), "");
    await writeFile(join(dir, "b_events.jsonl"), "");
    await writeFile(join(dir, "a_model.zip"), "zip");
    let middleware: any;
    (localRuns(dir).configureServer as Function)({
      middlewares: { use: (fn: unknown) => (middleware = fn) },
    });
    const request = (url: string) =>
      new Promise<{ status: number; body: string }>((done) => {
        const res = Object.assign(new PassThrough(), {
          statusCode: 200,
          setHeader() {},
        });
        let body = "";
        res.on("data", (c) => (body += c));
        res.on("finish", () => done({ status: res.statusCode, body }));
        void middleware({ url, method: "GET" }, res, () => {});
      });
    const index = JSON.parse((await request("/api/results")).body);
    assert.deepEqual(
      index.map((r: any) => [r.id, r.model]),
      [
        ["b", false],
        ["a", true],
      ],
    );
    assert.equal((await request("/api/results/a/model")).body, "zip");
    assert.equal((await request("/api/results/b/model")).status, 404);
    assert.equal((await request("/api/results/..%2fa/model")).status, 404);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
