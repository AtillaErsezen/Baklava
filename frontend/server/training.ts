import { spawn, execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
  access,
  mkdir,
  open,
  readFile,
  readdir,
  rename,
  rm,
  writeFile,
  stat,
} from "node:fs/promises";
import { join } from "node:path";
import { promisify } from "node:util";
import { loadEnv } from "vite";
import type { Connect, Plugin } from "vite";
import type { IncomingMessage } from "node:http";

const execute = promisify(execFile);
const LIMIT = 20 * 1024 * 1024;
const ID = /^[a-f0-9-]{36}$/;
type Data = Record<string, unknown>;
export type Worker = {
  invoke(command: string, args?: string[]): Promise<Data>;
  start(sessionFile: string, logFile: string): Promise<void>;
};
class RequestError extends Error {
  status: number;
  constructor(message: string, status = 400) {
    super(message);
    this.status = status;
  }
}
async function atomicJson(path: string, data: unknown) {
  const temp = `${path}.${randomUUID()}.tmp`;
  await writeFile(temp, JSON.stringify(data), { mode: 0o600 });
  await rename(temp, path);
}
export function pythonWorker(root: string): Worker {
  const python =
    process.env.BAKLAVA_PYTHON ||
    join(
      root,
      ".venv",
      process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
    );
  const env = () => ({
    ...process.env,
    ...loadEnv("development", root, ""),
    PYTHONUNBUFFERED: "1",
  });
  return {
    async invoke(command, args = []) {
      try {
        const { stdout } = await execute(
          python,
          [join(root, "web_training.py"), command, ...args],
          {
            cwd: root,
            env: env(),
            timeout: 45_000,
            maxBuffer: 4 * 1024 * 1024,
          },
        );
        return JSON.parse(stdout);
      } catch (error) {
        const e = error as Error & { stdout?: string; code?: string };
        if (e.stdout) {
          try {
            const out = JSON.parse(e.stdout);
            if (out.error) throw new RequestError(String(out.error));
          } catch (parsed) {
            if (parsed instanceof RequestError) throw parsed;
          }
        }
        throw new RequestError(
          e.code === "ENOENT"
            ? "Python environment not found. Run uv sync in the repository root, then retry."
            : "Could not inspect the dataset or training setup. Check the Python environment and try again.",
          503,
        );
      }
    },
    async start(sessionFile, logFile) {
      const log = await open(logFile, "a", 0o600);
      try {
        const child = spawn(
          python,
          [
            "-u",
            join(root, "web_training.py"),
            "train",
            "--session",
            sessionFile,
          ],
          {
            cwd: root,
            env: env(),
            detached: true,
            stdio: ["ignore", log.fd, log.fd],
          },
        );
        await new Promise<void>((resolve, reject) => {
          child.once("spawn", resolve);
          child.once("error", reject);
        });
        child.unref();
      } finally {
        await log.close();
      }
    },
  };
}
async function body(req: IncomingMessage, limit: number) {
  let size = 0;
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    const data = Buffer.from(chunk);
    size += data.length;
    if (size > limit)
      throw new RequestError(
        `The upload exceeds the ${limit === LIMIT ? "20 MB" : "64 KB"} limit.`,
        413,
      );
    chunks.push(data);
  }
  if (!size) throw new RequestError("The request is empty.");
  return Buffer.concat(chunks);
}
function allowed(req: IncomingMessage) {
  const host = new URL(`http://${req.headers.host || "invalid"}`).hostname;
  if (!["localhost", "127.0.0.1", "[::1]"].includes(host)) return false;
  if (
    req.socket.remoteAddress &&
    !["::1", "127.0.0.1", "::ffff:127.0.0.1"].includes(req.socket.remoteAddress)
  )
    return false;
  if (req.headers.origin) {
    try {
      if (new URL(req.headers.origin).host !== req.headers.host) return false;
    } catch {
      return false;
    }
  }
  return req.headers["sec-fetch-site"] !== "cross-site";
}
const publicSession = (session: Data) =>
  Object.fromEntries(
    Object.entries(session).filter(([key]) => !["pid"].includes(key)),
  );
export function trainingApi(
  root: string,
  worker: Worker = pythonWorker(root),
): Plugin {
  const directory = join(root, "runs", ".web"),
    datasets = join(directory, "datasets"),
    sessions = join(directory, "sessions"),
    lockPath = join(directory, "active.lock");
  async function getSession(id: string) {
    if (!ID.test(id)) throw new RequestError("Session not found.", 404);
    const path = join(sessions, `${id}.json`);
    let session: Data;
    try {
      session = JSON.parse(await readFile(path, "utf8"));
    } catch {
      throw new RequestError("Session not found.", 404);
    }
    if (["queued", "running"].includes(String(session.status))) {
      let alive = true;
      if (typeof session.pid === "number") {
        try {
          process.kill(session.pid, 0);
        } catch {
          alive = false;
        }
      } else if (Date.now() / 1000 - Number(session.createdAt) > 20)
        alive = false;
      if (!alive) {
        session = {
          ...session,
          status: "failed",
          stage: "Training interrupted",
          error:
            "The training process stopped before finishing. Review the session log and start a new session.",
          finishedAt: Date.now() / 1000,
        };
        await atomicJson(path, session);
      }
    }
    return session;
  }
  let acquiring = false;
  async function acquire(id: string) {
    if (acquiring)
      throw new RequestError(
        "Another session is starting. Try again shortly.",
        409,
      );
    acquiring = true;
    try {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const lock = await open(lockPath, "wx", 0o600);
          await lock.writeFile(id);
          await lock.close();
          return;
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        }
        let active: Data | undefined;
        try {
          const existing = await readFile(lockPath, "utf8");
          active = await getSession(existing);
        } catch {
          const age = Date.now() - (await stat(lockPath)).mtimeMs;
          if (age > 20_000) {
            await rm(lockPath, { force: true });
            continue;
          }
          throw new RequestError(
            "Another training session is being prepared. Try again shortly.",
            409,
          );
        }
        if (active && ["queued", "running"].includes(String(active.status)))
          throw new RequestError(
            "A training session is already running. Open Training to follow its progress.",
            409,
          );
        await rm(lockPath, { force: true });
      }
      throw new RequestError(
        "Another session is starting. Try again shortly.",
        409,
      );
    } finally {
      acquiring = false;
    }
  }
  const middleware: Connect.NextHandleFunction = async (req, res, next) => {
    const path = new URL(req.url || "/", "http://localhost").pathname;
    if (!path.startsWith("/api/training/")) return next();
    const send = (value: unknown, status = 200) => {
      res.statusCode = status;
      res.setHeader("Content-Type", "application/json");
      res.setHeader("Cache-Control", "no-store");
      res.end(JSON.stringify(value));
    };
    try {
      if (!allowed(req))
        throw new RequestError(
          "Training is available only from this local workspace.",
          403,
        );
      await Promise.all([
        mkdir(datasets, { recursive: true, mode: 0o700 }),
        mkdir(sessions, { recursive: true, mode: 0o700 }),
      ]);
      if (path === "/api/training/setup" && req.method === "GET") {
        try {
          send(await worker.invoke("check"));
        } catch (e) {
          send({
            ready: false,
            issues: [
              {
                label: "Python environment",
                detail: (e as Error).message,
                command: "uv sync",
              },
            ],
          });
        }
        return;
      }
      if (path === "/api/training/datasets" && req.method === "POST") {
        const filename = decodeURIComponent(
          String(req.headers["x-filename"] || "dataset.csv"),
        );
        if (
          !filename.toLowerCase().endsWith(".csv") ||
          filename.length > 200 ||
          filename.includes("/") ||
          filename.includes("\\") ||
          [...filename].some((c) => c.charCodeAt(0) < 32)
        )
          throw new RequestError("Choose a CSV file with a valid filename.");
        const content = await body(req, LIMIT),
          id = randomUUID(),
          folder = join(datasets, id);
        if (content.includes(0))
          throw new RequestError("The upload is not a UTF-8 CSV file.");
        await mkdir(folder, { mode: 0o700 });
        try {
          await writeFile(join(folder, "dataset.csv"), content, {
            mode: 0o600,
          });
          const info = await worker.invoke("inspect", [
            "--data",
            join(folder, "dataset.csv"),
          ]);
          const data = { ...info, id, filename, bytes: content.length };
          await atomicJson(join(folder, "metadata.json"), data);
          send(data, 201);
        } catch (e) {
          await rm(folder, { recursive: true, force: true });
          throw e;
        }
        return;
      }
      if (path === "/api/training/sessions" && req.method === "GET") {
        const files = (await readdir(sessions)).filter(
          (f) => ID.test(f.slice(0, -5)) && f.endsWith(".json"),
        );
        const all = await Promise.all(
          files.map((f) => getSession(f.slice(0, -5))),
        );
        send(
          all
            .sort((a, b) => Number(b.createdAt) - Number(a.createdAt))
            .map(publicSession),
        );
        return;
      }
      if (path === "/api/training/sessions" && req.method === "POST") {
        if (!String(req.headers["content-type"]).startsWith("application/json"))
          throw new RequestError("Send session settings as JSON.", 415);
        let input: Data;
        try {
          input = JSON.parse((await body(req, 64 * 1024)).toString());
        } catch (e) {
          if (e instanceof RequestError) throw e;
          throw new RequestError("Invalid session settings.");
        }
        const {
          datasetId,
          target,
          task = "auto",
          purpose = "",
          requestId,
        } = input;
        if (
          typeof datasetId !== "string" ||
          !ID.test(datasetId) ||
          typeof requestId !== "string" ||
          !ID.test(requestId) ||
          typeof target !== "string" ||
          target.length > 1000 ||
          typeof purpose !== "string" ||
          purpose.length > 1000 ||
          !["auto", "classification", "regression"].includes(String(task))
        )
          throw new RequestError(
            "Choose a dataset, target column, and valid task.",
          );
        const sessionPath = join(sessions, `${requestId}.json`);
        try {
          await access(sessionPath);
          const existing = await getSession(requestId);
          if (
            existing.datasetId !== datasetId ||
            existing.target !== target ||
            existing.purpose !== purpose ||
            existing.requestedTask !== task
          )
            throw new RequestError(
              "This request was already used for another session.",
              409,
            );
          send(publicSession(existing));
          return;
        } catch (e) {
          if (e instanceof RequestError) throw e;
          if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
        }
        let metadata: Data;
        try {
          metadata = JSON.parse(
            await readFile(join(datasets, datasetId, "metadata.json"), "utf8"),
          );
        } catch {
          throw new RequestError(
            "Dataset not found. Upload the CSV again.",
            404,
          );
        }
        const check = await worker.invoke("check");
        if (!check.ready) {
          send(
            {
              error: "Complete the training setup before starting a session.",
              setup: check,
            },
            503,
          );
          return;
        }
        const validated = await worker.invoke("validate", [
          "--data",
          join(datasets, datasetId, "dataset.csv"),
          "--target",
          target,
          "--task",
          String(task),
        ]);
        await acquire(requestId);
        const session: Data = {
          id: requestId,
          datasetId,
          filename: metadata.filename,
          target,
          task: validated.task,
          requestedTask: task,
          purpose,
          status: "queued",
          stage: "Starting the pipeline",
          createdAt: Date.now() / 1000,
          eventCount: 0,
          logFile: `runs/.web/sessions/${requestId}.log`,
        };
        try {
          await atomicJson(sessionPath, session);
          await worker.start(sessionPath, join(sessions, `${requestId}.log`));
          send(publicSession(session), 202);
        } catch {
          session.status = "failed";
          session.error =
            "Could not start the Python training process. Check your environment and retry.";
          await atomicJson(sessionPath, session);
          await rm(lockPath, { force: true });
          throw new RequestError(String(session.error), 503);
        }
        return;
      }
      const match = path.match(/^\/api\/training\/sessions\/([a-f0-9-]{36})$/);
      if (match && req.method === "GET") {
        send(publicSession(await getSession(match[1])));
        return;
      }
      send({ error: "Training endpoint not found." }, 404);
    } catch (error) {
      send(
        {
          error:
            error instanceof RequestError
              ? error.message
              : "The training service could not complete this request. Try again.",
        },
        error instanceof RequestError ? error.status : 500,
      );
    }
  };
  return {
    name: "baklava-training",
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
  };
}
