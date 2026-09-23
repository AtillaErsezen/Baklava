// Isolated browser QA harness. Never imported by the production Vite config.
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { trainingApi, pythonWorker } from "../server/training.ts";
import { localRuns } from "../server/runs.ts";
const project = fileURLToPath(new URL("../../", import.meta.url));
const temporary = await mkdtemp(join(tmpdir(), "baklava-browser-qa-"));
const actual = pythonWorker(project);
const server = await createServer({
  configFile: false,
  root: join(project, "frontend"),
  plugins: [
    react(),
    trainingApi(temporary, {
      invoke: async (command, args) =>
        command === "check"
          ? { ready: true, issues: [] }
          : actual.invoke(command, args),
      async start(path) {
        const session = JSON.parse(await readFile(path, "utf8"));
        await writeFile(
          path,
          JSON.stringify({
            ...session,
            status: "running",
            pid: process.pid,
            stage: "Searching model candidates",
            message:
              "UI TEST FIXTURE: Simulating session progress. No model is being trained.",
            eventCount: 2,
          }),
        );
        setTimeout(
          () =>
            void (async () => {
              const runId = `qa-${session.id}`;
              const event = (
                kind: string,
                payload: Record<string, unknown>,
                ts: number,
              ) => ({ run_id: runId, kind, payload, ts });
              const events = [
                event(
                  "run_start",
                  {
                    dataset: "UI TEST FIXTURE (simulated)",
                    target: session.target,
                    rows: 100,
                    features: 2,
                  },
                  1,
                ),
                event(
                  "leaderboard",
                  {
                    primary_metric: "accuracy",
                    rows: [{ name: "logreg", accuracy: 0.81, fit_seconds: 1 }],
                  },
                  2,
                ),
                event(
                  "final_model",
                  {
                    name: "logreg",
                    spec: { model: "logreg", task: "classification" },
                    model_path: "/TEST-FIXTURE/model.joblib",
                  },
                  3,
                ),
                event(
                  "report",
                  {
                    markdown:
                      "## UI test fixture\n\nThese are simulated scores for browser verification. No model was trained.",
                    spoken_summary:
                      "Simulated result for browser testing only.",
                  },
                  4,
                ),
              ];
              const file = `${runId}_events.jsonl`;
              await writeFile(
                join(temporary, "runs", file),
                events.map((e) => JSON.stringify(e)).join("\n"),
              );
              await writeFile(
                path,
                JSON.stringify({
                  ...session,
                  status: "completed",
                  resultFile: file,
                  stage: "Your results are ready",
                  finishedAt: Date.now() / 1000,
                  eventCount: 4,
                }),
              );
            })(),
          8000,
        );
      },
    }),
    localRuns(join(temporary, "runs")),
  ],
  server: { host: "127.0.0.1", port: 5187, strictPort: true },
});
await server.listen();
console.log(
  "Isolated UI fixture: http://127.0.0.1:5187/#training (simulated training only)",
);
const close = async () => {
  await server.close();
  await rm(temporary, { recursive: true, force: true });
  process.exit(0);
};
process.on("SIGINT", () => void close());
process.on("SIGTERM", () => void close());
