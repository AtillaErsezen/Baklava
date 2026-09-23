import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { mkdtemp, readFile, writeFile, rm, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { trainingApi } from "../server/training.ts";
import type { Worker } from "../server/training.ts";
import { localRuns } from "../server/runs.ts";
import { parseRun } from "../src/results/data.ts";

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "baklava-training-"));
  let configured = true,
    starts = 0;
  const worker: Worker = {
    async invoke(command, args = []) {
      if (command === "check")
        return {
          ready: configured,
          issues: configured
            ? []
            : [{ label: "Nebius API key", detail: "Set NEBIUS_API_KEY" }],
        };
      if (command === "inspect") {
        if ((await readFile(args[1], "utf8")).includes("invalid"))
          throw new Error("Invalid fixture");
        return {
          rows: 100,
          columns: [
            { name: "x", type: "number" },
            { name: "y", type: "number" },
          ],
          preview: [["1", "0"]],
        };
      }
      if (command === "validate") return { task: "classification" };
      throw new Error("Unexpected worker command");
    },
    async start(file) {
      starts++;
      const state = JSON.parse(await readFile(file, "utf8"));
      await writeFile(
        file,
        JSON.stringify({ ...state, status: "running", pid: process.pid }),
      );
    },
  };
  const stack: any[] = [];
  for (const plugin of [
    trainingApi(root, worker),
    localRuns(join(root, "runs")),
  ])
    (plugin.configureServer as Function)({
      middlewares: {
        use(handler: unknown) {
          stack.push(handler);
        },
      },
    });
  const server = createServer((req, res) => {
    let i = 0;
    const next = () => {
      const middleware = stack[i++];
      if (middleware) void middleware(req, res, next);
      else {
        res.statusCode = 404;
        res.end();
      }
    };
    next();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  const request = async (path: string, init: RequestInit = {}) => {
    const response = await fetch(url + path, init);
    return { status: response.status, data: await response.json() };
  };
  const upload = () =>
    request("/api/training/datasets", {
      method: "POST",
      headers: { "Content-Type": "text/csv", "X-Filename": "test.csv" },
      body: "x,y\n1,0\n",
    });
  const start = (datasetId: string, requestId = randomUUID()) =>
    request("/api/training/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        datasetId,
        requestId,
        target: "y",
        task: "auto",
        purpose: "",
      }),
    });
  return {
    root,
    url,
    request,
    upload,
    start,
    setReady(value: boolean) {
      configured = value;
    },
    get starts() {
      return starts;
    },
    async close() {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      );
      await rm(root, { recursive: true, force: true });
    },
  };
}
test("upload -> start -> poll -> saved results, idempotent requests, and restart recovery", async () => {
  const f = await fixture();
  try {
    const upload = await f.upload();
    assert.equal(upload.status, 201);
    assert.equal(upload.data.filename, "test.csv");
    const id = randomUUID();
    const started = await f.start(upload.data.id, id);
    assert.equal(started.status, 202);
    assert.equal(f.starts, 1);
    const duplicate = await f.start(upload.data.id, id);
    assert.equal(duplicate.status, 200);
    assert.equal(f.starts, 1);
    assert.equal((await f.start(upload.data.id)).status, 409);
    assert.equal(
      (await f.request(`/api/training/sessions/${id}`)).data.status,
      "running",
    );
    assert.equal((await f.request("/api/training/sessions")).data[0].id, id);
    assert.equal(
      (await f.request(`/api/training/sessions/${id}`)).data.pid,
      undefined,
    );
    const sessionFile = join(f.root, "runs", ".web", "sessions", `${id}.json`);
    const state = JSON.parse(await readFile(sessionFile, "utf8"));
    const events = [
      {
        run_id: "test-result",
        ts: 1,
        kind: "run_start",
        payload: { dataset: "test.csv", target: "y" },
      },
      {
        run_id: "test-result",
        ts: 2,
        kind: "final_model",
        payload: { name: "TEST FIXTURE", model_path: "/test.joblib" },
      },
      {
        run_id: "test-result",
        ts: 3,
        kind: "report",
        payload: { markdown: "TEST FIXTURE" },
      },
    ];
    await writeFile(
      join(f.root, "runs", "test-result_events.jsonl"),
      events.map((e) => JSON.stringify(e)).join("\n"),
    );
    await writeFile(
      sessionFile,
      JSON.stringify({
        ...state,
        status: "completed",
        resultFile: "test-result_events.jsonl",
      }),
    );
    const completed = await f.request(`/api/training/sessions/${id}`);
    assert.equal(completed.data.status, "completed");
    const results = await f.request(
      `/api/results/${completed.data.resultFile}`,
    );
    assert.equal(parseRun(results.data.content).complete, true);
    assert.equal((await f.start(upload.data.id)).status, 202);
    assert.equal(f.starts, 2);
  } finally {
    await f.close();
  }
});
test("missing credentials block training while uploads remain available", async () => {
  const f = await fixture();
  try {
    f.setReady(false);
    assert.equal((await f.request("/api/training/setup")).data.ready, false);
    const data = await f.upload();
    assert.equal(data.status, 201);
    const start = await f.start(data.data.id);
    assert.equal(start.status, 503);
    assert.equal(start.data.setup.ready, false);
    assert.equal(f.starts, 0);
  } finally {
    await f.close();
  }
});
test("cross-origin starts, arbitrary paths, invalid CSV filenames, and malformed input are rejected", async () => {
  const f = await fixture();
  try {
    assert.equal(
      (
        await f.request("/api/training/sessions", {
          method: "POST",
          headers: {
            Origin: "https://untrusted.example",
            "Content-Type": "application/json",
          },
          body: "{}",
        })
      ).status,
      403,
    );
    assert.equal(
      (
        await f.request("/api/training/datasets", {
          method: "POST",
          headers: { "X-Filename": "..%2Fdata.csv" },
          body: "x,y\n1,0",
        })
      ).status,
      400,
    );
    assert.equal(
      (
        await f.request("/api/training/datasets", {
          method: "POST",
          headers: { "X-Filename": "data.exe" },
          body: "x,y\n1,0",
        })
      ).status,
      400,
    );
    assert.equal(
      (
        await f.request("/api/training/sessions", {
          method: "POST",
          headers: { "Content-Type": "text/plain" },
          body: "{}",
        })
      ).status,
      415,
    );
    assert.equal(
      (
        await f.request("/api/training/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{",
        })
      ).status,
      400,
    );
    assert.equal((await f.start("../../etc")).status, 400);
    assert.equal(f.starts, 0);
  } finally {
    await f.close();
  }
});
test("interrupted queued session becomes failed rather than staying in progress forever", async () => {
  const f = await fixture();
  try {
    const folder = join(f.root, "runs", ".web", "sessions");
    await mkdir(folder, { recursive: true });
    const id = randomUUID();
    await writeFile(
      join(folder, `${id}.json`),
      JSON.stringify({ id, status: "queued", createdAt: 1 }),
    );
    const state = await f.request(`/api/training/sessions/${id}`);
    assert.equal(state.data.status, "failed");
    assert.match(state.data.error, /stopped before finishing/);
  } finally {
    await f.close();
  }
});

test("concurrent starts launch only one worker", async () => {
  const f = await fixture();
  try {
    const upload = await f.upload();
    const responses = await Promise.all([
      f.start(upload.data.id),
      f.start(upload.data.id),
    ]);
    assert.deepEqual(responses.map((r) => r.status).sort(), [202, 409]);
    assert.equal(f.starts, 1);
  } finally {
    await f.close();
  }
});
