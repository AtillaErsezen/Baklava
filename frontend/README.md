# Baklava website

Responsive landing page for Baklava, built with React, TypeScript, and Vite. Fonts are bundled locally; the page requires no API keys or backend services.

## Development

```sh
cd frontend
npm ci
npm run dev
```

## Checks and production build

```sh
npm run lint
npm run build
npm run format:check
npm run preview
```

`dist/` is the deployable static site. `src/App.tsx` contains the landing page. The layered illustration and interactive example live in `src/components/`; `demo-data.ts` holds the explicitly illustrative sample data. `src/App.css` contains responsive component styles; `src/index.css` contains design tokens and local fonts.

## Interactive example

The sample workspace switches between customer churn and house prices, replays an illustrative workflow, and exports a sample Markdown report. Dataset dimensions match the repository's demo CSV files. Scores and conclusions are explicitly illustrative; replay does not invoke the ML engine, train a model, upload a file, or persist data.

The training workspace at `/#training` accepts CSV datasets and runs the existing Modal + Nebius agent. Server credentials stay in the repository-root `.env` file.

## Design direction

Dark technical styling, lime accents, spacious typography, and an original layered model illustration. References supplied for visual inspiration: [Nebius Token Factory](https://nebius.com/services/token-factory) and [Accel](https://www.accel.com/).

## Cinematic interlude

`BusinessFlow.tsx` presents a live Three.js animation: gold particles gather into amber glass layers and settle into a sculpture. The renderer in `src/components/promo/createPromoScene.ts` is loaded as the section approaches the viewport. Playback pauses offscreen, in a hidden tab, or when the visitor selects Pause. Reduced-motion preferences show a still composition; a CSS sculpture remains available when WebGL cannot run. The 24-second loop uses continuous easing and display-synchronized rendering, with curved particle paths, beveled glass, studio reflections, and soft contact shadows. Scene controls ease between compositions. Multisample antialiasing runs on the postprocessing target. The reference video is not shipped or embedded.

## Results workspace

Open `/#results` from the navigation or the demo. The workspace follows the site's dark palette, Manrope typography, lime accents, and layered identity. It includes a recommendation, validation and holdout metrics, sortable/filterable model comparison, model parameters, feature importance, diagnostics, an event timeline, and Markdown report export.

Two clearly labeled illustrative runs demonstrate classification and regression. To inspect your own results, use **Import a result** or drop a `runs/*_events.jsonl` / `runs/*_results.json` file onto the workspace. Files are parsed in the browser, are not uploaded, and remain in memory until the page is refreshed. Event files include richer metadata, diagnostics, and reports than result snapshots. Missing measurements stay empty; RMSE/MAE comparisons correctly rank lower scores first.

During Vite development and preview, a read-only `/api/results` endpoint discovers saved runs from the repository's `runs/` directory. It only serves matching result/event filenames, rejects symlinks and files over 20 MB, and does not read credentials or contact cloud services. Static deployments retain browser imports and the examples; automatic local discovery requires the Vite server. Newly generated runs appear when the workspace is reopened or refreshed. Model artifacts remain in Modal; the UI presents the result files and reports produced by the agent.

Run `npm test` for adapter, metric-direction, file-validation, and local endpoint checks. `npm run lint`, `npm run build`, and `npm run format:check` validate the frontend.


## Upload a dataset and train

Open `/#training` or choose **New training**. Upload a comma-separated UTF-8 CSV (50–100,000 rows, up to 200 columns / 20 MB), review the first five rows, select the target, and optionally choose the prediction type and describe your goal. Start training to run the existing `FactoryRun` with **Nebius** and **Modal**. The page displays actual agent progress and automatically opens results only after a final model and report have been saved. Failed sessions show an error, a local log path, and any partial results.

Before the first cloud run, from the repository root:

```sh
uv sync
# Copy .env.example to .env only if you do not already have a .env file.
# Fill in NEBIUS_API_KEY and, optionally, MODAL_TOKEN_ID / MODAL_TOKEN_SECRET.
uv run modal setup   # not needed if both Modal token variables are configured
uv run --env-file .env modal deploy modal_train.py
cd frontend
npm install
npm run dev
```

The **Recheck training setup** button picks up new `.env` values without requiring a new browser session. The preflight verifies local dependencies and configured credentials; actual authentication, quotas, and the `ml-factory` deployment are verified when the pipeline runs. Existing Supabase persistence and optional Tavily enrichment retain the agent's behavior. No secret is sent to the browser. Cloud compute and API usage follow the existing pipeline's budgets.

`web_training.py` wraps the existing agent, streams full events to `runs/*_events.jsonl`, and writes durable session state. The Vite middleware in `server/training.ts` handles CSV uploads and session polling. This is a **local workspace service**: training endpoints accept only loopback, same-origin requests; static-only hosting cannot start Python jobs. Uploads and session metadata/logs are stored under gitignored `runs/.web/`. One session can run at a time. The worker survives browser refreshes and Vite restarts; reopening Training reconnects to the active session. Keep this computer running while the agent is working.

The Python executable defaults to `.venv/bin/python` (`.venv/Scripts/python.exe` on Windows). Set `BAKLAVA_PYTHON` in the terminal environment before starting Vite to use a different environment. Credential setup remains optional for uploading and previewing data, but is required to start cloud training.

Verification:

```sh
# Repository root
.venv/bin/python -m unittest test_web_training -v
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run build
```

These tests use a stubbed agent/worker and make no paid cloud calls. An optional isolated browser fixture is available with `node frontend/tests/browser-fixture.ts`; its clearly labeled simulated results use temporary storage and port 5187, and never reach Modal or Nebius. Stop it with Ctrl+C after testing.
