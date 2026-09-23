import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { trainingApi } from "./server/training.ts";
import { localRuns } from "./server/runs.ts";

export default defineConfig({
  plugins: [
    react(),
    trainingApi(fileURLToPath(new URL("../", import.meta.url))),
    localRuns(fileURLToPath(new URL("../runs", import.meta.url))),
  ],
});
