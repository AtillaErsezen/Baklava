import { Fragment, useEffect, useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent, ReactNode } from "react";
import {
  ArrowDown,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  ChevronRight,
  CircleCheck,
  Clock3,
  Database,
  FileJson,
  FileSpreadsheet,
  FileText,
  FlaskConical,
  Layers3,
  LoaderCircle,
  Plus,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TriangleAlert,
  Trophy,
  X,
} from "lucide-react";
import {
  familyLabel,
  lowerIsBetter,
  metricLabel,
  parseRun,
  rankModels,
  reportMarkdown,
  scoreLabel,
} from "./data";
import type { Model, Run } from "./data";
import { samples } from "./samples";
import "./Results.css";
import Training from "../training/Training";
import type { Session } from "../training/Training";

type View = "Overview" | "Models" | "Report" | "Activity";
const views: View[] = ["Overview", "Models", "Report", "Activity"];
const durationLabel = (seconds?: number) =>
  seconds === undefined
    ? "—"
    : seconds >= 60
      ? `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
      : `${seconds.toFixed(1)}s`;
function download(text: string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="result-empty">
      <Layers3 size={25} />
      <p>{children}</p>
    </div>
  );
}
function PanelTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="r-panel-title">
      <div>
        <span className="r-label">{eyebrow}</span>
        <h2>{title}</h2>
      </div>
      {children}
    </div>
  );
}
function ModelTable({ run, full = false }: { run: Run; full?: boolean }) {
  const [sort, setSort] = useState("score");
  const [filter, setFilter] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  let models = rankModels(run.models, run.metric).filter((m) =>
    `${m.name} ${familyLabel(m.family)}`
      .toLowerCase()
      .includes(filter.toLowerCase()),
  );
  if (sort === "fit")
    models = [...models].sort(
      (a, b) => (a.fit ?? Infinity) - (b.fit ?? Infinity),
    );
  return (
    <>
      {full && (
        <div className="r-table-tools">
          <label className="r-search">
            <Search size={15} />
            <input
              aria-label="Search models"
              placeholder="Find a model…"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </label>
          <label className="r-sort">
            <SlidersHorizontal size={14} />
            <select
              aria-label="Sort models"
              value={sort}
              onChange={(e) => setSort(e.target.value)}
            >
              <option value="score">Best score</option>
              <option value="fit">Fastest training</option>
            </select>
          </label>
        </div>
      )}
      {!models.length ? (
        <Empty>
          {filter
            ? "No models match your search."
            : "Model scores will appear when a leaderboard is recorded."}
        </Empty>
      ) : (
        <div className="r-table-scroll">
          <table className="r-model-table">
            <caption className="sr-only">
              Model comparison. {metricLabel(run.metric)}:{" "}
              {lowerIsBetter(run.metric) ? "lower" : "higher"} is better. Expand
              a model to inspect its parameters.
            </caption>
            <thead>
              <tr>
                <th scope="col">MODEL</th>
                <th scope="col">
                  {metricLabel(run.metric)}{" "}
                  {lowerIsBetter(run.metric) ? "↓" : "↑"}
                </th>
                {full && <th scope="col">HOLDOUT</th>}
                <th scope="col">FIT TIME</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model, i) => (
                <Fragment key={model.name}>
                  <tr
                    className={model.name === run.winner ? "r-winner-row" : ""}
                  >
                    <th scope="row">
                      <button
                        className="r-model-toggle"
                        aria-expanded={expanded === model.name}
                        onClick={() =>
                          setExpanded(
                            expanded === model.name ? null : model.name,
                          )
                        }
                      >
                        <span className="r-rank">
                          {String(i + 1).padStart(2, "0")}
                        </span>
                        <span>
                          <span className="r-model-family">
                            {familyLabel(model.family)}
                            {model.name === run.winner && (
                              <Trophy
                                size={12}
                                aria-label="Recommended model"
                              />
                            )}
                          </span>
                          {full && model.name !== model.family && (
                            <span className="r-model-id">{model.name}</span>
                          )}
                        </span>
                        <ChevronRight
                          className={expanded === model.name ? "rotated" : ""}
                          size={13}
                        />
                      </button>
                    </th>
                    <td>
                      <span className="r-score">{scoreLabel(model.score)}</span>
                      {model.std !== undefined && (
                        <small> ± {scoreLabel(model.std)}</small>
                      )}
                    </td>
                    {full && <td>{scoreLabel(model.holdout)}</td>}
                    <td>
                      {model.fit === undefined
                        ? "—"
                        : `${model.fit.toFixed(1)}s`}
                    </td>
                  </tr>
                  {expanded === model.name && (
                    <tr className="r-detail-row">
                      <td colSpan={full ? 4 : 3}>
                        <div>
                          <strong>{model.name}</strong>
                          {model.error ? (
                            <p className="r-warning-text">{model.error}</p>
                          ) : (
                            <p>
                              {model.ci
                                ? `95% confidence interval: ${scoreLabel(model.ci[0])} – ${scoreLabel(model.ci[1])}. `
                                : ""}
                              {model.predict !== undefined
                                ? `Prediction time: ${model.predict.toFixed(2)} ms.`
                                : ""}
                            </p>
                          )}
                          <pre>
                            {Object.keys(model.params).length
                              ? JSON.stringify(model.params, null, 2)
                              : "No parameters recorded for this candidate."}
                          </pre>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
function Importance({ run }: { run: Run }) {
  const features = run.importance.slice(0, 5),
    max = Math.max(...features.map((f) => Math.abs(f.importance)), 1e-10);
  return (
    <section className="r-panel r-importance">
      <PanelTitle eyebrow="THE SIGNAL" title="What matters most">
        <Sparkles size={18} />
      </PanelTitle>
      {!features.length ? (
        <Empty>No feature importance was recorded.</Empty>
      ) : (
        <>
          <div className="r-feature-bars">
            {features.map((f, i) => (
              <div key={f.feature} className="r-feature-row">
                <div>
                  <span>{f.feature}</span>
                  <span>{f.importance.toFixed(3)}</span>
                </div>
                <div className="r-feature-track">
                  <span
                    style={{
                      width: `${(Math.abs(f.importance) / max) * 100}%`,
                      opacity: 1 - i * 0.14,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="r-panel-footnote">
            Relative contribution to this model.
            <br />
            Importance explains predictions, not causation.
          </p>
        </>
      )}
    </section>
  );
}
function ScoreChart({ run }: { run: Run }) {
  const models = rankModels(run.models, run.metric)
    .filter((m): m is Model & { score: number } => m.score !== undefined)
    .slice(0, 4);
  if (!models.length) return null;
  const values = models.flatMap((m) => m.ci ?? [m.score]);
  const min = Math.min(...values),
    max = Math.max(...values),
    pad = (max - min || Math.abs(max) * 0.1 || 1) * 0.18;
  const start = min - pad,
    end = max + pad;
  const x = (v: number) => 20 + ((v - start) / (end - start)) * 260;
  return (
    <div className="r-score-chart">
      <div className="r-chart-label">
        <span>VALIDATION SCORE</span>
        <span>{lowerIsBetter(run.metric) ? "← LOWER" : "HIGHER →"}</span>
      </div>
      <svg
        viewBox={`0 0 300 ${models.length * 28 + 20}`}
        role="img"
        aria-label={`${metricLabel(run.metric)} comparison${models.some((m) => m.ci) ? " with 95% confidence intervals" : ""}. ${models.map((m) => `${familyLabel(m.family)} ${scoreLabel(m.score)}`).join(", ")}`}
      >
        {[0, 1, 2, 3, 4].map((i) => (
          <line
            key={i}
            x1={20 + i * 65}
            x2={20 + i * 65}
            y1="2"
            y2={models.length * 28}
            stroke="#33403e"
            strokeDasharray="2 4"
          />
        ))}
        {models.map((m, i) => (
          <g
            key={m.name}
            stroke={m.name === run.winner ? "#d0eea2" : "#638b7f"}
            fill={m.name === run.winner ? "#d0eea2" : "#638b7f"}
          >
            {m.ci && (
              <>
                <line
                  x1={x(m.ci[0])}
                  x2={x(m.ci[1])}
                  y1={14 + i * 28}
                  y2={14 + i * 28}
                  strokeWidth="2"
                />
                <line
                  x1={x(m.ci[0])}
                  x2={x(m.ci[0])}
                  y1={10 + i * 28}
                  y2={18 + i * 28}
                />
                <line
                  x1={x(m.ci[1])}
                  x2={x(m.ci[1])}
                  y1={10 + i * 28}
                  y2={18 + i * 28}
                />
              </>
            )}
            <circle cx={x(m.score)} cy={14 + i * 28} r="4" />
          </g>
        ))}
        <text x="20" y={models.length * 28 + 17} fill="#93a59d" fontSize="10">
          {scoreLabel(start)}
        </text>
        <text
          x="280"
          y={models.length * 28 + 17}
          fill="#93a59d"
          textAnchor="end"
          fontSize="10"
        >
          {scoreLabel(end)}
        </text>
      </svg>
      <span className="r-chart-caption">
        {models.some((m) => m.ci)
          ? "Dots: CV mean · Lines: 95% confidence interval"
          : "Dots show each candidate’s validation score"}
      </span>
    </div>
  );
}
function Report({ run }: { run: Run }) {
  // Render report text as React nodes. Imported HTML is never executed.
  const lines = run.report.split(/\r?\n/);
  return (
    <section className="r-panel r-report">
      <PanelTitle eyebrow="THE REASONING" title="A result you can understand">
        <FileText size={19} />
      </PanelTitle>
      {run.report ? (
        <div className="r-prose">
          {lines.map((line, i) =>
            line.startsWith("#") ? (
              <h3 key={i}>{line.replace(/^#+\s*/, "")}</h3>
            ) : line.startsWith("- ") ? (
              <p className="r-report-bullet" key={i}>
                <Check size={14} />
                {line.slice(2)}
              </p>
            ) : line.trim() ? (
              <p key={i}>{line}</p>
            ) : null,
          )}
        </div>
      ) : (
        <Empty>
          No report was saved for this run. The model comparison is available
          separately.
        </Empty>
      )}
      {run.modelPath && (
        <div className="r-artifact">
          <FileJson size={18} />
          <div>
            <span className="r-label">SAVED MODEL</span>
            <code>{run.modelPath}</code>
            <p>Model artifacts remain in the training environment.</p>
          </div>
        </div>
      )}
    </section>
  );
}
export default function Results({ training = false }: { training?: boolean }) {
  const [session, setSession] = useState<Session | null>(null);
  const trainingActive =
    session?.status === "queued" || session?.status === "running";
  const trainingLabel = trainingActive
    ? "View training"
    : session
      ? "View session"
      : "New training";
  const sessionLabel = trainingActive
    ? "Training in progress"
    : session?.status === "completed"
      ? "Results ready"
      : "Training needs attention";
  const [runs, setRuns] = useState<Run[]>(samples),
    [selected, setSelected] = useState(samples[0].id);
  const [view, setView] = useState<View>("Overview"),
    [localFiles, setLocalFiles] = useState<{ id: string; file: string }[]>([]);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [loading, setLoading] = useState(false),
    [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null),
    request = useRef(0);
  const run = runs.find((r) => r.id === selected) ?? samples[0];
  const winner = run.models.find((m) => m.name === run.winner);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/results", { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => {
        if (Array.isArray(data))
          setLocalFiles(
            data.filter(
              (r) => typeof r.id === "string" && typeof r.file === "string",
            ),
          );
      })
      .catch(() => {});
    return () => controller.abort();
  }, []);
  const selectRun = (id: string) => {
    request.current++;
    setLoading(false);
    setSelected(id);
    window.location.assign("#results");
    setView("Overview");
    setError("");
    setNotice("");
  };
  function addRun(next: Run) {
    setRuns((current) => [...current.filter((r) => r.id !== next.id), next]);
    setSelected(next.id);
    setView("Overview");
    setError("");
  }
  async function loadLocal(file: string) {
    const token = ++request.current;
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const response = await fetch(`/api/results/${encodeURIComponent(file)}`);
      const data = await response.json();
      if (!response.ok)
        throw new Error(data.error || "Could not open this run.");
      const next = parseRun(data.content, file);
      if (token === request.current) {
        addRun(next);
        window.location.assign("#results");
      }
    } catch (e) {
      if (token === request.current)
        setError(e instanceof Error ? e.message : "Could not open this run.");
    } finally {
      if (token === request.current) setLoading(false);
    }
  }
  async function importFile(file?: File) {
    if (!file) return;
    const token = ++request.current;
    setLoading(true);
    setError("");
    setNotice("");
    try {
      if (file.size > 20 * 1024 * 1024)
        throw new Error("Choose a result file smaller than 20 MB.");
      const next = parseRun(await file.text(), file.name);
      if (token === request.current) {
        addRun(next);
        setNotice("Result imported. Your file stays in this browser.");
      }
    } catch (e) {
      if (token === request.current)
        setError(e instanceof Error ? e.message : "Could not read this file.");
    } finally {
      if (token === request.current) setLoading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }
  function tabKey(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % views.length;
    else if (event.key === "ArrowLeft")
      next = (index + views.length - 1) % views.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = views.length - 1;
    else return;
    event.preventDefault();
    setView(views[next]);
    document.getElementById(`results-tab-${views[next]}`)?.focus();
  }
  async function openTrainingResult(file: string) {
    const response = await fetch(`/api/results/${encodeURIComponent(file)}`);
    const data = await response.json();
    if (!response.ok)
      throw new Error(data.error || "Could not open training results.");
    request.current++;
    addRun(parseRun(data.content, file));
    setNotice("Your training results are ready.");
    setLocalFiles((current) =>
      current.some((f) => f.file === file)
        ? current
        : [{ id: file.replace(/_events\.jsonl$/, ""), file }, ...current],
    );
    window.location.assign("#results");
    window.scrollTo(0, 0);
  }
  function exportReport() {
    download(
      reportMarkdown(run),
      `baklava-${run.id.replace(/[^a-zA-Z0-9_-]/g, "-")}-report.md`,
      "text/markdown",
    );
    setNotice("Report downloaded.");
  }
  return (
    <div
      className="results-app"
      onDragOver={(e) => {
        if (training) return;
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node))
          setDragging(false);
      }}
      onDrop={(e) => {
        if (training) return;
        e.preventDefault();
        setDragging(false);
        void importFile(e.dataTransfer.files[0]);
      }}
    >
      <a className="skip-link" href="#results-content">
        Skip to results
      </a>
      <header className="r-header">
        <a className="r-brand" href="#" aria-label="Baklava home">
          <Layers3 size={28} strokeWidth={1.6} />
          <span>
            baklava<span>.</span>
          </span>
        </a>
        <span className="r-header-divider" />
        <span className="r-workspace-label">Workspace</span>
        <nav aria-label="Workspace navigation">
          <a href="#how-it-works">
            How it works <ArrowUpRight size={12} />
          </a>
          <span className="r-local-label">
            <span /> LOCAL WORKSPACE
          </span>
        </nav>
      </header>
      <div className="r-layout">
        <aside className="r-sidebar">
          <div className="r-sidebar-main">
            <a className="r-home-link" href="#">
              <ArrowLeft size={14} /> Back to Baklava
            </a>
            <div className="r-sidebar-heading">
              <span className="r-label">YOUR WORKSPACE</span>
              <Layers3 size={13} />
            </div>
            <a
              href="#training"
              className={`r-button r-button-primary r-new-training ${training ? "current" : ""}`}
            >
              {trainingActive ? (
                <LoaderCircle size={15} className="spin" />
              ) : (
                <Plus size={15} />
              )}{" "}
              {trainingLabel}
            </a>
            {session && (
              <a
                href="#training"
                className={`r-session-link ${training ? "selected" : ""}`}
                aria-current={training ? "page" : undefined}
              >
                <span className="r-label">{sessionLabel}</span>
                <strong title={session.filename}>{session.filename}</strong>
                <small>{session.stage}</small>
              </a>
            )}
            <a
              href="#results"
              className={`r-nav-current ${!training ? "selected" : ""}`}
            >
              <span>
                <FileText size={16} /> Results
              </span>
              <span className="r-count">
                {runs.length +
                  localFiles.filter((f) => !runs.some((r) => r.id === f.id))
                    .length}
              </span>
            </a>
            <div className="r-sidebar-heading r-run-heading">
              <span className="r-label">SAMPLE RUNS</span>
              <FlaskConical size={13} />
            </div>
            <div className="r-run-list">
              {runs
                .filter((r) => r.sample)
                .map((r) => (
                  <button
                    key={r.id}
                    className={!training && selected === r.id ? "selected" : ""}
                    aria-pressed={!training && selected === r.id}
                    onClick={() => selectRun(r.id)}
                  >
                    <FileSpreadsheet size={16} />
                    <span>
                      {r.label}
                      <small>{r.task}</small>
                    </span>
                    {!training && selected === r.id && (
                      <span className="r-selected-dot" />
                    )}
                  </button>
                ))}
            </div>
            {(localFiles.length > 0 || runs.some((r) => !r.sample)) && (
              <>
                <div className="r-sidebar-heading r-run-heading">
                  <span className="r-label">SAVED & IMPORTED</span>
                </div>
                <div className="r-run-list">
                  {runs
                    .filter((r) => !r.sample)
                    .map((r) => (
                      <button
                        title={r.id}
                        key={r.id}
                        aria-pressed={!training && selected === r.id}
                        className={
                          !training && selected === r.id ? "selected" : ""
                        }
                        onClick={() => selectRun(r.id)}
                      >
                        <FileJson size={16} />
                        <span>
                          {r.label}
                          <small>{r.id}</small>
                        </span>
                      </button>
                    ))}
                  {localFiles
                    .filter((f) => !runs.some((r) => r.id === f.id))
                    .map((f) => (
                      <button
                        title={f.id}
                        key={f.id}
                        onClick={() => void loadLocal(f.file)}
                      >
                        <FileJson size={16} />
                        <span>
                          Saved run<small>{f.id}</small>
                        </span>
                      </button>
                    ))}
                </div>
              </>
            )}
            <button
              className="r-import-button"
              onClick={() => fileInput.current?.click()}
            >
              <Plus size={15} /> Import a result
            </button>
          </div>
          <div className="r-sidebar-bottom">
            <div className="r-layer-stack" aria-hidden="true">
              <Layers3 size={34} strokeWidth={1} />
            </div>
            <p>
              Every layer.
              <br />
              <span>A little more clarity.</span>
            </p>
            <a href="#how-it-works">
              Meet the process <ArrowUpRight size={12} />
            </a>
            <span className="r-sidebar-version">BAKLAVA / ML FACTORY</span>
          </div>
        </aside>
        <main id="results-content" className="r-main">
          <Training
            hidden={!training}
            session={session}
            onSessionChange={setSession}
            onComplete={openTrainingResult}
          />
          <div hidden={training}>
            {session && (
              <a
                href="#training"
                className={`r-session-banner ${session.status}`}
              >
                {trainingActive ? (
                  <LoaderCircle size={20} className="spin" />
                ) : session.status === "completed" ? (
                  <CircleCheck size={20} />
                ) : (
                  <TriangleAlert size={20} />
                )}
                <span>
                  <strong>{sessionLabel}</strong>
                  <small>
                    {session.filename} · {session.stage}
                  </small>
                </span>
                <span className="r-session-action">
                  {trainingLabel}
                  <ArrowRight size={14} />
                </span>
              </a>
            )}
            <div className="r-breadcrumb">
              <span>Workspace</span>
              <ChevronRight size={12} />
              <span>Results</span>
              <ChevronRight size={12} />
              <strong>{run.dataset}</strong>
            </div>
            <div className="r-page-heading">
              <div>
                <div className="r-eyebrow">
                  <span /> THE OUTCOME, UNPACKED
                </div>
                <h1>
                  Your results, <em>explained.</em>
                </h1>
                <p>From a thousand possibilities to your next best step.</p>
              </div>
              <div className="r-page-actions">
                <a className="r-button t-secondary" href="#training">
                  {trainingActive ? (
                    <LoaderCircle size={14} className="spin" />
                  ) : (
                    <Plus size={14} />
                  )}{" "}
                  {trainingLabel}
                </a>
                <button
                  className="r-button r-button-primary"
                  onClick={exportReport}
                >
                  <ArrowDownToLine size={15} /> Export report
                </button>
              </div>
            </div>
            <div className="r-mobile-picker">
              <label htmlFor="mobile-run">Current run</label>
              <select
                id="mobile-run"
                value={selected}
                onChange={(e) => {
                  const local = localFiles.find((f) => f.id === e.target.value);
                  if (local && !runs.some((r) => r.id === local.id))
                    void loadLocal(local.file);
                  else selectRun(e.target.value);
                }}
              >
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.label}
                    {r.sample ? " · Sample" : ""}
                  </option>
                ))}
                {localFiles
                  .filter((f) => !runs.some((r) => r.id === f.id))
                  .map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.id}
                    </option>
                  ))}
              </select>
              <button
                className="r-icon-button"
                aria-label="Import a result"
                onClick={() => fileInput.current?.click()}
              >
                <Plus size={18} />
              </button>
            </div>
            <div className="r-dataset-strip">
              <div>
                <FileSpreadsheet size={16} />
                <strong>{run.dataset}</strong>
                <span className="r-task-badge">{run.task}</span>
              </div>
              <span>
                <Database size={13} />
                {run.rows?.toLocaleString("en-US") ?? "—"} rows <i />{" "}
                {run.features ?? "—"} features <i /> Target:{" "}
                <strong>{run.target}</strong>
              </span>
            </div>
            {run.sample && (
              <div className="r-sample-notice">
                <FlaskConical size={13} />
                <span>
                  Sample run · Illustrative results, no live training.
                </span>
                <button onClick={() => fileInput.current?.click()}>
                  Open your results <ArrowUpRight size={12} />
                </button>
              </div>
            )}
            {error && (
              <div className="r-alert" role="alert">
                <TriangleAlert size={17} />
                <p>{error}</p>
                <button aria-label="Dismiss error" onClick={() => setError("")}>
                  <X size={15} />
                </button>
              </div>
            )}
            {notice && (
              <div className="r-toast" role="status">
                <CircleCheck size={14} />
                {notice}
              </div>
            )}
            {loading && (
              <div className="r-toast" role="status">
                <LoaderCircle className="spin" size={14} />
                Reading saved results…
              </div>
            )}
            <div className="r-tabs" role="tablist" aria-label="Result views">
              {views.map((v, i) => (
                <button
                  key={v}
                  id={`results-tab-${v}`}
                  role="tab"
                  aria-selected={view === v}
                  aria-controls="results-panel"
                  tabIndex={view === v ? 0 : -1}
                  onKeyDown={(e) => tabKey(e, i)}
                  onClick={() => setView(v)}
                >
                  {v}
                  {v === "Models" && <span>{run.models.length}</span>}
                </button>
              ))}
              <span
                className={`r-run-status ${run.complete ? "complete" : ""}`}
              >
                {run.complete ? (
                  <CircleCheck size={13} />
                ) : (
                  <Clock3 size={13} />
                )}{" "}
                {run.complete ? "Run complete" : "Partial results"}
              </span>
            </div>
            <div
              id="results-panel"
              role="tabpanel"
              aria-labelledby={`results-tab-${view}`}
              tabIndex={0}
              className="r-tab-panel"
              key={run.id + view}
            >
              {view === "Overview" && (
                <>
                  <section className="r-winner-panel">
                    <div className="r-winner-copy">
                      <span className="r-recommendation">
                        <span>
                          <Check size={11} />
                        </span>
                        {run.winner ? "RECOMMENDED MODEL" : "RESULTS SO FAR"}
                      </span>
                      <h2>
                        {winner
                          ? familyLabel(winner.family)
                          : run.winner || "The search is taking shape."}
                      </h2>
                      <p>
                        {run.summary ||
                          (run.winner
                            ? "Selected by the agent for this run. Explore the comparison and recorded report for the reasoning behind this choice."
                            : "This saved run does not include a final model yet. You can explore any recorded activity and report.")}
                      </p>
                      <button
                        className="r-text-button"
                        onClick={() => setView("Report")}
                      >
                        Read the reasoning <ArrowRight size={14} />
                      </button>
                    </div>
                    <div className="r-winner-score">
                      <span className="r-label">{metricLabel(run.metric)}</span>
                      <strong>{scoreLabel(winner?.score)}</strong>
                      <span>
                        {lowerIsBetter(run.metric) ? "Lower" : "Higher"} is
                        better <ArrowUpRight size={12} />
                      </span>
                      <span className="r-score-subtitle">
                        Cross-validation score
                      </span>
                    </div>
                    <div className="r-winner-art" aria-hidden="true">
                      <Layers3 size={128} strokeWidth={0.6} />
                    </div>
                  </section>
                  <div className="r-stats">
                    <div>
                      <span>
                        <ShieldCheck size={14} /> Holdout score
                      </span>
                      <strong>{scoreLabel(winner?.holdout)}</strong>
                      <small>
                        {winner?.holdout !== undefined
                          ? "Checked on unseen data"
                          : "Not recorded"}
                      </small>
                    </div>
                    <div>
                      <span>
                        <Layers3 size={14} /> Models compared
                      </span>
                      <strong>
                        {String(run.models.length).padStart(2, "0")}
                      </strong>
                      <small>
                        {run.searched
                          ? `From ${run.searched.toLocaleString("en-US")} pipeline possibilities`
                          : "In the recorded comparison"}
                      </small>
                    </div>
                    <div>
                      <span>
                        <Clock3 size={14} /> Run duration
                      </span>
                      <strong>{durationLabel(run.duration)}</strong>
                      <small>
                        {run.duration === undefined
                          ? "Timing not recorded"
                          : "From first to last event"}
                      </small>
                    </div>
                    <div>
                      <span>
                        <Sparkles size={14} /> Model API cost
                      </span>
                      <strong>
                        {run.cost === undefined
                          ? "—"
                          : `$${run.cost.toFixed(2)}`}
                      </strong>
                      <small>
                        {run.cost === undefined
                          ? "Usage not recorded"
                          : "Excludes training compute"}
                      </small>
                    </div>
                  </div>
                  <div className="r-overview-grid">
                    <section className="r-panel r-comparison">
                      <PanelTitle
                        eyebrow="THE CONTENDERS"
                        title="A fair comparison"
                      >
                        <button
                          className="r-text-button"
                          onClick={() => setView("Models")}
                        >
                          View all <ArrowUpRight size={13} />
                        </button>
                      </PanelTitle>
                      <ModelTable run={run} />
                      <div className="r-comparison-bottom">
                        <ShieldCheck size={14} />
                        <p>
                          {run.confirmed
                            ? "Finalists compared on paired validation folds."
                            : "Latest recorded validation results."}{" "}
                          {metricLabel(run.metric)}:{" "}
                          {lowerIsBetter(run.metric) ? "lower" : "higher"} is
                          better.
                        </p>
                      </div>
                    </section>
                    <Importance run={run} />
                  </div>
                  <div className="r-bottom-grid">
                    <section className="r-insight">
                      <span className="r-insight-icon">
                        <Sparkles size={20} />
                      </span>
                      <div>
                        <span className="r-label">THE TAKEAWAY</span>
                        <p>
                          {run.summary ||
                            "Every result has a story. Open the report to review the agent’s recorded reasoning and recommendations."}
                        </p>
                        <button
                          className="r-text-button"
                          onClick={() => setView("Report")}
                        >
                          Explore the report <ArrowRight size={13} />
                        </button>
                      </div>
                    </section>
                    <section className="r-checks">
                      <span className="r-label">
                        A HEALTHY DOSE OF SKEPTICISM
                      </span>
                      {run.findings.length ? (
                        <>
                          <p>
                            <TriangleAlert size={14} />
                            {
                              run.findings.filter((f) => f.severity >= 2).length
                            }{" "}
                            {run.findings.filter((f) => f.severity >= 2)
                              .length === 1
                              ? "check"
                              : "checks"}{" "}
                            to review
                          </p>
                          <span>{run.findings[0].finding}</span>
                          <button
                            className="r-text-button"
                            onClick={() => setView("Activity")}
                          >
                            Review diagnostics <ArrowRight size={13} />
                          </button>
                        </>
                      ) : (
                        <p>No diagnostics recorded</p>
                      )}
                    </section>
                  </div>
                </>
              )}
              {view === "Models" && (
                <div className="r-models-grid">
                  <section className="r-panel">
                    <PanelTitle
                      eyebrow="THE FULL COMPARISON"
                      title="Different models. One question."
                    />
                    <ModelTable run={run} full />
                    <div className="r-comparison-bottom">
                      <ShieldCheck size={15} />
                      <p>
                        Compare validation performance, then check the holdout.
                        An empty value means it was not recorded.
                      </p>
                    </div>
                  </section>
                  <aside className="r-panel r-model-guide">
                    <PanelTitle
                      eyebrow="READING THE RESULTS"
                      title={metricLabel(run.metric)}
                    />
                    <p>
                      {run.metric === "roc_auc"
                        ? "How well the model ranks positive examples above negative ones. 1.0 is perfect ranking; 0.5 is chance."
                        : lowerIsBetter(run.metric)
                          ? "Prediction error between the model’s estimates and observed values. A lower score is better."
                          : "The run’s primary validation metric. A higher score is better."}
                    </p>
                    <ScoreChart run={run} />
                    <div className="r-guide-note">
                      <ShieldCheck size={16} />
                      <p>
                        The recommendation can favor a simpler model when
                        validation scores are statistically close.
                      </p>
                    </div>
                  </aside>
                </div>
              )}
              {view === "Report" && <Report run={run} />}
              {view === "Activity" && (
                <div className="r-activity-grid">
                  <section className="r-panel">
                    <PanelTitle
                      eyebrow="LAYER BY LAYER"
                      title="How we got here"
                    />
                    <ol className="r-timeline">
                      {run.events.length ? (
                        run.events.map((event, i) => (
                          <li key={`${event.kind}-${i}`}>
                            <span className="r-timeline-dot">
                              {event.kind === "final_model" ? (
                                <Trophy size={13} />
                              ) : (
                                <Check size={11} />
                              )}
                            </span>
                            <div>
                              <span className="r-event-heading">
                                {event.kind.replaceAll("_", " ")}
                                <time>
                                  {durationLabel(
                                    Math.max(0, event.ts - run.events[0].ts),
                                  )}
                                </time>
                              </span>
                              <p>
                                {String(
                                  event.payload.text ||
                                    event.payload.msg ||
                                    event.payload.rationale ||
                                    event.payload.spoken_summary ||
                                    (event.kind === "run_start"
                                      ? `Opened ${run.dataset} · Target: ${run.target}`
                                      : event.kind === "final_model"
                                        ? `Selected ${String(event.payload.name)}`
                                        : event.kind === "confirm"
                                          ? "Compared finalist performance and selected a recommendation."
                                          : event.kind === "usage"
                                            ? "Recorded model API usage."
                                            : "Recorded in this run."),
                                )}
                              </p>
                              <details>
                                <summary>
                                  Event details <ChevronDown size={12} />
                                </summary>
                                <pre>
                                  {JSON.stringify(event.payload, null, 2)}
                                </pre>
                              </details>
                            </div>
                          </li>
                        ))
                      ) : (
                        <Empty>
                          No event timeline was included in this result file.
                        </Empty>
                      )}
                    </ol>
                  </section>
                  <section className="r-panel">
                    <PanelTitle
                      eyebrow="TRUST, WITH CONTEXT"
                      title="Data diagnostics"
                    />
                    {run.findings.length ? (
                      run.findings.map((f, i) => (
                        <article className="r-finding" key={i}>
                          {f.severity >= 2 ? (
                            <TriangleAlert size={17} />
                          ) : (
                            <ShieldCheck size={17} />
                          )}
                          <div>
                            <span className="r-label">
                              {f.check.replaceAll("_", " ")} ·{" "}
                              {f.severity >= 3
                                ? "HIGH"
                                : f.severity === 2
                                  ? "REVIEW"
                                  : "INFO"}
                            </span>
                            <p>{f.finding}</p>
                          </div>
                        </article>
                      ))
                    ) : (
                      <Empty>No diagnostics were recorded for this run.</Empty>
                    )}
                  </section>
                </div>
              )}
            </div>
            <footer className="r-footer">
              <span>
                <Layers3 size={13} /> Machine learning. Layer by layer.
              </span>
              <span>{run.sample ? "ILLUSTRATIVE SAMPLE" : run.id}</span>
            </footer>
          </div>
        </main>
      </div>
      <input
        ref={fileInput}
        type="file"
        className="sr-only"
        tabIndex={-1}
        accept=".json,.jsonl"
        aria-label="Import Baklava result file"
        onChange={(e: ChangeEvent<HTMLInputElement>) =>
          void importFile(e.target.files?.[0])
        }
      />
      {dragging && (
        <div className="r-drop-overlay">
          <ArrowDown size={34} />
          <h2>Drop a little clarity.</h2>
          <p>Open a Baklava JSON or JSONL result file.</p>
        </div>
      )}
    </div>
  );
}
