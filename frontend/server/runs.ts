import { createReadStream } from "node:fs";
import { readdir, readFile, lstat } from "node:fs/promises";
import { resolve } from "node:path";
import type { Plugin, Connect } from "vite";

// Local, read-only result access. No dataset rows, credentials, or arbitrary paths.
export function localRuns(directory: string): Plugin {
  const middleware: Connect.NextHandleFunction = async (req, res, next) => {
    const url = new URL(req.url || "/", "http://localhost");
    if (!url.pathname.startsWith("/api/results")) return next();
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Cache-Control", "no-store");
    if (req.method !== "GET") {
      res.statusCode = 405;
      res.end(JSON.stringify({ error: "Method not allowed" }));
      return;
    }
    try {
      const all: string[] = await readdir(directory).catch(
        (error: NodeJS.ErrnoException) => {
          if (error.code === "ENOENT") return [];
          throw error;
        },
      );
      const files = all.filter((name) =>
        /^[a-zA-Z0-9_-]+_(events\.jsonl|results\.json)$/.test(name),
      );
      if (url.pathname === "/api/results") {
        const runs = new Map<string, string>();
        for (const file of files.sort()) {
          const id = file.replace(/_(events\.jsonl|results\.json)$/, "");
          if (!runs.has(id) || file.endsWith("_events.jsonl"))
            runs.set(id, file);
        }
        res.end(
          JSON.stringify(
            [...runs]
              .map(([id, file]) => ({
                id,
                file,
                model: all.includes(`${id}_model.zip`),
              }))
              .reverse(),
          ),
        );
        return;
      }
      // Fitted model + retrain script + model card, zipped by the Python export step.
      const model = /^\/api\/results\/([a-zA-Z0-9_-]+)\/model$/.exec(
        url.pathname,
      )?.[1];
      if (model) {
        const file = `${model}_model.zip`;
        if (
          !all.includes(file) ||
          !(await lstat(resolve(directory, file))).isFile()
        ) {
          res.statusCode = 404;
          res.end(JSON.stringify({ error: "No model was saved for this run" }));
          return;
        }
        res.setHeader("Content-Type", "application/zip");
        res.setHeader(
          "Content-Disposition",
          `attachment; filename="baklava-${file}"`,
        );
        createReadStream(resolve(directory, file)).pipe(res);
        return;
      }
      const file = decodeURIComponent(
        url.pathname.slice("/api/results/".length),
      );
      if (!files.includes(file)) {
        res.statusCode = 404;
        res.end(JSON.stringify({ error: "Run not found" }));
        return;
      }
      const path = resolve(directory, file),
        stat = await lstat(path);
      if (!stat.isFile() || stat.size > 20 * 1024 * 1024) {
        res.statusCode = 413;
        res.end(
          JSON.stringify({
            error: "Result file is unavailable or exceeds 20 MB",
          }),
        );
        return;
      }
      res.end(JSON.stringify({ file, content: await readFile(path, "utf8") }));
    } catch {
      res.statusCode = 500;
      res.end(JSON.stringify({ error: "Could not read saved results" }));
    }
  };
  return {
    name: "baklava-local-results",
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
  };
}
