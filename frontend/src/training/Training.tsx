import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Check,
  CircleCheck,
  Cloud,
  Database,
  FileSpreadsheet,
  Layers3,
  LoaderCircle,
  Play,
  RefreshCw,
  Sparkles,
  Target,
  TriangleAlert,
  Upload,
  X,
} from "lucide-react";
import "./Training.css";

type Column = {
  name: string;
  type: string;
  missing: number;
  suggestedTask: string;
  uniqueSample: number;
};
type Dataset = {
  id: string;
  filename: string;
  bytes: number;
  rows: number;
  columns: Column[];
  preview: string[][];
};
type Setup = {
  ready: boolean;
  issues: { label: string; detail: string; command?: string }[];
};
type Session = {
  id: string;
  datasetId: string;
  filename: string;
  target: string;
  task: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  message?: string;
  createdAt: number;
  startedAt?: number;
  finishedAt?: number;
  error?: string;
  eventCount: number;
  resultFile?: string;
  logFile: string;
};
const stages = [
  "Prepare data",
  "Explore models",
  "Validate & select",
  "Deliver results",
];
const active = (session: Session | null) =>
  session?.status === "queued" || session?.status === "running";
function stageIndex(session: Session) {
  if (session.status === "completed") return 4;
  if (/Saving|Writing|Finishing/.test(session.stage)) return 3;
  if (/Testing/.test(session.stage)) return 2;
  if (/Searching|Comparing|Evaluating/.test(session.stage)) return 1;
  return 0;
}
async function jsonResponse(response: Response) {
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      "The training service is unavailable. Run this app with npm run dev or npm run preview.",
    );
  }
  if (!response.ok)
    throw new Error(data.error || "The request could not be completed.");
  return data;
}
function remember(id: string | null) {
  try {
    if (id) sessionStorage.setItem("baklava-active-session", id);
    else sessionStorage.removeItem("baklava-active-session");
  } catch {
    /* Session discovery also works without browser storage. */
  }
}
function remembered() {
  try {
    return sessionStorage.getItem("baklava-active-session");
  } catch {
    return null;
  }
}
export default function Training({
  hidden,
  onComplete,
}: {
  hidden: boolean;
  onComplete: (file: string) => Promise<void>;
}) {
  const [dataset, setDataset] = useState<Dataset | null>(null),
    [target, setTarget] = useState(""),
    [task, setTask] = useState("auto"),
    [purpose, setPurpose] = useState("");
  const [setup, setSetup] = useState<Setup | null>(null),
    [session, setSession] = useState<Session | null>(null),
    [error, setError] = useState("");
  const [uploading, setUploading] = useState(false),
    [starting, setStarting] = useState(false),
    [checking, setChecking] = useState(false),
    [dragging, setDragging] = useState(false),
    [connection, setConnection] = useState("");
  const input = useRef<HTMLInputElement>(null),
    uploadRequest = useRef<AbortController | null>(null),
    requestId = useRef(crypto.randomUUID()),
    delivered = useRef("");
  const complete = useRef(onComplete),
    shouldOpen = useRef(false),
    mounted = useRef(true);
  const busy = active(session) || starting;
  const sessionId = session?.id,
    sessionStatus = session?.status;
  const chosen = dataset?.columns.find((c) => c.name === target);
  const resolvedTask = task === "auto" ? chosen?.suggestedTask : task;
  const invalidTarget =
    chosen &&
    (chosen.missing > 0 ||
      chosen.uniqueSample < 2 ||
      (resolvedTask === "regression" && chosen.type !== "number"));
  useEffect(() => {
    complete.current = onComplete;
  }, [onComplete]);
  async function refreshSetup() {
    setChecking(true);
    try {
      setSetup(await jsonResponse(await fetch("/api/training/setup")));
    } catch (e) {
      setSetup({
        ready: false,
        issues: [{ label: "Training service", detail: (e as Error).message }],
      });
    } finally {
      setChecking(false);
    }
  }
  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    fetch("/api/training/setup", { signal: controller.signal })
      .then(jsonResponse)
      .then(setSetup)
      .catch((e) => {
        if (!controller.signal.aborted)
          setSetup({
            ready: false,
            issues: [{ label: "Training service", detail: e.message }],
          });
      });
    fetch("/api/training/sessions", { signal: controller.signal })
      .then(jsonResponse)
      .then((all: Session[]) => {
        const running = all.find(active),
          stored = all.find((s) => s.id === remembered());
        if (running || stored) {
          shouldOpen.current = true;
          setSession(running || stored!);
        }
      })
      .catch(() => {});
    return () => {
      mounted.current = false;
      controller.abort();
      uploadRequest.current?.abort();
    };
  }, []);
  useEffect(() => {
    if (!sessionId || !["queued", "running"].includes(sessionStatus || ""))
      return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await jsonResponse(
          await fetch(`/api/training/sessions/${sessionId}`, {
            signal: controller.signal,
          }),
        );
        setSession(next);
        setConnection("");
      } catch (e) {
        if (!controller.signal.aborted)
          setConnection(`${(e as Error).message} Retrying automatically.`);
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 2000);
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [sessionId, sessionStatus]);
  useEffect(() => {
    if (
      session?.status === "completed" &&
      session.resultFile &&
      shouldOpen.current &&
      delivered.current !== session.id
    ) {
      delivered.current = session.id;
      shouldOpen.current = false;
      remember(null);
      complete.current(session.resultFile).catch((e) => {
        if (mounted.current)
          setError(
            `Training finished, but results could not be opened: ${(e as Error).message}`,
          );
      });
    }
    if (session?.status === "failed") remember(null);
  }, [session]);
  async function upload(file?: File) {
    if (!file || busy) return;
    setError("");
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError(
        "Choose a CSV dataset. Saved result files can still be opened from Results.",
      );
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("Choose a CSV smaller than 20 MB.");
      return;
    }
    uploadRequest.current?.abort();
    const controller = new AbortController();
    uploadRequest.current = controller;
    setUploading(true);
    setDataset(null);
    setSession(null);
    setTarget("");
    setTask("auto");
    try {
      const data: Dataset = await jsonResponse(
        await fetch("/api/training/datasets", {
          method: "POST",
          headers: {
            "Content-Type": "text/csv",
            "X-Filename": encodeURIComponent(file.name),
          },
          body: file,
          signal: controller.signal,
        }),
      );
      setDataset(data);
      setTarget(data.columns[data.columns.length - 1].name);
      requestId.current = crypto.randomUUID();
    } catch (e) {
      if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      if (!controller.signal.aborted) setUploading(false);
      if (input.current) input.current.value = "";
    }
  }
  async function start() {
    if (!dataset || busy || !target || invalidTarget) return;
    setStarting(true);
    setError("");
    try {
      const response = await fetch("/api/training/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          datasetId: dataset.id,
          target,
          task,
          purpose,
          requestId: requestId.current,
        }),
      });
      const data = await response.json();
      if (data.setup) setSetup(data.setup);
      if (!response.ok)
        throw new Error(data.error || "Could not start training.");
      shouldOpen.current = true;
      remember(data.id);
      setSession(data);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  }
  function reset() {
    setSession(null);
    setDataset(null);
    setTarget("");
    setPurpose("");
    setError("");
    requestId.current = crypto.randomUUID();
  }
  const settingsChanged = () => {
    requestId.current = crypto.randomUUID();
  };
  return (
    <section
      hidden={hidden}
      className="training-screen"
      onDragOver={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!busy) setDragging(true);
      }}
      onDragLeave={(e) => {
        e.stopPropagation();
        if (!e.currentTarget.contains(e.relatedTarget as Node))
          setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragging(false);
        void upload(e.dataTransfer.files[0]);
      }}
    >
      <div className="r-breadcrumb">
        <a href="#results">Workspace</a>
        <Chevron />
        <strong>New training session</strong>
      </div>
      <div className="r-page-heading">
        <div>
          <div className="r-eyebrow">
            <span /> YOUR NEXT MODEL STARTS HERE
          </div>
          <h1>
            Bring your data.
            <br className="t-mobile-break" />
            <em> Find your signal.</em>
          </h1>
          <p>One dataset. A clear question. Let the layers do the rest.</p>
        </div>
        <a href="#results" className="r-text-button">
          <ArrowLeft size={13} /> Back to results
        </a>
      </div>
      <div className="t-flow" aria-label="Training workflow">
        {["Upload a dataset", "Choose your target", "Train & explore"].map(
          (label, i) => (
            <div
              key={label}
              className={i === 0 || dataset || session ? "available" : ""}
            >
              <span>
                {(i === 0 && (dataset || session)) || (i === 1 && session) ? (
                  <Check size={12} />
                ) : (
                  String(i + 1).padStart(2, "0")
                )}
              </span>
              {label}
              {i < 2 && <ArrowRight size={13} />}
            </div>
          ),
        )}
      </div>
      {error && (
        <div className="r-alert" role="alert">
          <TriangleAlert size={16} />
          <p>{error}</p>
          <button
            aria-label="Dismiss training error"
            onClick={() => setError("")}
          >
            <X size={14} />
          </button>
        </div>
      )}
      {session ? (
        <>
          <div className={`t-session ${session.status}`} aria-live="polite">
            <div className="t-session-top">
              <span className="t-session-icon">
                {session.status === "completed" ? (
                  <CircleCheck size={27} />
                ) : session.status === "failed" ? (
                  <TriangleAlert size={27} />
                ) : (
                  <LoaderCircle size={27} className="spin" />
                )}
              </span>
              <div>
                <span className="r-label">
                  {active(session)
                    ? "TRAINING IN PROGRESS"
                    : session.status === "completed"
                      ? "SESSION COMPLETE"
                      : "SESSION NEEDS ATTENTION"}
                </span>
                <h2>{session.stage}</h2>
                <p>
                  {session.filename} <span>·</span> Target: {session.target}{" "}
                  <span>·</span> {session.task}
                </p>
              </div>
              <span className="t-cloud-tag">
                <Cloud size={13} /> MODAL + NEBIUS
              </span>
            </div>
            <div className="t-session-stages">
              {stages.map((label, i) => (
                <div
                  key={label}
                  className={
                    stageIndex(session) > i
                      ? "done"
                      : stageIndex(session) === i
                        ? "current"
                        : ""
                  }
                >
                  <span>
                    {stageIndex(session) > i ? <Check size={13} /> : i + 1}
                  </span>
                  {label}
                </div>
              ))}
            </div>
            {session.error ? (
              <div className="t-session-message failure">
                <TriangleAlert size={17} />
                <div>
                  <p>{session.error}</p>
                  <small>
                    Session log: <code>{session.logFile}</code>
                  </small>
                </div>
              </div>
            ) : (
              <div className="t-session-message">
                <Sparkles size={18} />
                <p>
                  {session.message ||
                    "The agent is preparing your dataset and planning its first experiments."}
                </p>
              </div>
            )}
            <div className="t-session-bottom">
              <span>
                {session.eventCount} events recorded{" "}
                {active(session) && "· You can refresh this page safely."}
              </span>
              {session.resultFile && (
                <button
                  className="r-button r-button-primary"
                  onClick={() =>
                    complete
                      .current(session.resultFile!)
                      .catch((e) => setError(e.message))
                  }
                >
                  {session.status === "completed"
                    ? "View results"
                    : "View partial results"}
                  <ArrowRight size={14} />
                </button>
              )}
              {session.status === "failed" && (
                <button
                  className="r-button t-secondary"
                  onClick={() => {
                    setSession(null);
                    settingsChanged();
                    void refreshSetup();
                  }}
                >
                  Try again <RefreshCw size={13} />
                </button>
              )}
              {session.status === "completed" && (
                <button className="r-text-button" onClick={reset}>
                  Start another session <ArrowUpRight size={13} />
                </button>
              )}
            </div>
          </div>
          {connection && (
            <div className="r-alert" role="status">
              <RefreshCw size={15} />
              <p>{connection}</p>
            </div>
          )}
          <p className="t-running-note">
            {active(session)
              ? "Results will open automatically when the agent finishes its final model and report. Training duration depends on your dataset and available compute."
              : "Your session and results are saved in this workspace."}
          </p>
        </>
      ) : (
        <div className="t-form-grid">
          <div className="t-form-main">
            <section className="r-panel t-data-panel">
              <div className="r-panel-title">
                <div>
                  <span className="r-label">01 / THE INPUT</span>
                  <h2>Your dataset</h2>
                </div>
                <span className="t-limit">CSV · UP TO 20 MB</span>
              </div>
              {!dataset ? (
                <button
                  className={`t-upload ${dragging ? "dragging" : ""}`}
                  disabled={uploading}
                  onClick={() => input.current?.click()}
                >
                  <span className="t-upload-icon">
                    {uploading ? (
                      <LoaderCircle className="spin" size={28} />
                    ) : (
                      <Upload size={28} strokeWidth={1.5} />
                    )}
                  </span>
                  <strong>
                    {uploading
                      ? "Getting to know your dataset…"
                      : "Drop your CSV here"}
                  </strong>
                  <span>
                    {uploading ? (
                      "Checking columns, values, and file structure."
                    ) : (
                      <>
                        or <em>browse files</em> to get started
                      </>
                    )}
                  </span>
                  <small>Comma-separated · UTF-8 · 50–100,000 rows</small>
                </button>
              ) : (
                <>
                  <div className="t-file-summary">
                    <span className="t-file-icon">
                      <FileSpreadsheet size={23} />
                    </span>
                    <div>
                      <strong>{dataset.filename}</strong>
                      <p>
                        {dataset.rows.toLocaleString()} rows ·{" "}
                        {dataset.columns.length} columns ·{" "}
                        {(dataset.bytes / 1024).toFixed(1)} KB
                      </p>
                    </div>
                    <button
                      className="r-text-button"
                      onClick={() => input.current?.click()}
                    >
                      Replace <RefreshCw size={12} />
                    </button>
                  </div>
                  <div
                    className="t-preview"
                    tabIndex={0}
                    aria-label="Dataset preview, scroll horizontally to see more columns"
                  >
                    <table>
                      <caption>
                        First {dataset.preview.length} rows · Preview only
                      </caption>
                      <thead>
                        <tr>
                          {dataset.columns.map((c) => (
                            <th
                              key={c.name}
                              className={c.name === target ? "target" : ""}
                            >
                              {c.name}
                              {c.name === target && <Target size={11} />}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {dataset.preview.map((row, i) => (
                          <tr key={i}>
                            {row.map((v, j) => (
                              <td
                                key={j}
                                className={
                                  dataset.columns[j].name === target
                                    ? "target"
                                    : ""
                                }
                              >
                                {v || (
                                  <span className="t-empty-cell">empty</span>
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </section>
            <section
              className={`r-panel t-settings ${!dataset ? "waiting" : ""}`}
            >
              <div className="r-panel-title">
                <div>
                  <span className="r-label">02 / THE QUESTION</span>
                  <h2>What would you like to predict?</h2>
                </div>
                <Target size={18} />
              </div>
              <div className="t-fields">
                <div className="t-field">
                  <label htmlFor="training-target">Target column</label>
                  <select
                    id="training-target"
                    disabled={!dataset || uploading}
                    value={target}
                    onChange={(e) => {
                      setTarget(e.target.value);
                      settingsChanged();
                    }}
                  >
                    <option value="" disabled>
                      {dataset ? "Choose a column" : "Upload a dataset first"}
                    </option>
                    {dataset?.columns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                  <p>The value you want your model to learn to predict.</p>
                </div>
                <div className="t-field">
                  <label htmlFor="training-task">Prediction type</label>
                  <select
                    id="training-task"
                    disabled={!dataset || uploading}
                    value={task}
                    onChange={(e) => {
                      setTask(e.target.value);
                      settingsChanged();
                    }}
                  >
                    <option value="auto">
                      Auto-detect{chosen ? ` · ${chosen.suggestedTask}` : ""}
                    </option>
                    <option value="classification">
                      Classification · categories
                    </option>
                    <option value="regression">Regression · numbers</option>
                  </select>
                  <p>
                    {resolvedTask === "regression"
                      ? "Predict a number, such as price or demand."
                      : "Predict a category, such as churn or no churn."}
                  </p>
                </div>
                <div className="t-field t-purpose">
                  <label htmlFor="training-purpose">
                    Give your model some context <span>OPTIONAL</span>
                  </label>
                  <textarea
                    id="training-purpose"
                    disabled={!dataset || uploading}
                    maxLength={1000}
                    value={purpose}
                    onChange={(e) => {
                      setPurpose(e.target.value);
                      settingsChanged();
                    }}
                    placeholder="e.g. Identify customers likely to leave, with an emphasis on explainability."
                    rows={3}
                  />
                </div>
                {invalidTarget && (
                  <p className="t-target-error" role="alert">
                    <TriangleAlert size={14} />
                    {chosen!.missing
                      ? "The target contains missing values. Fill or remove those rows before training."
                      : chosen!.uniqueSample < 2
                        ? "Choose a target with at least two different values."
                        : "Regression requires a numeric target."}
                  </p>
                )}
              </div>
            </section>
          </div>
          <aside className="t-summary">
            <div className="r-panel">
              <div className="r-panel-title">
                <div>
                  <span className="r-label">03 / LET THE LAYERS WORK</span>
                  <h2>A model. And the why.</h2>
                </div>
                <Layers3 size={19} />
              </div>
              <ul className="t-promises">
                <li>
                  <Database size={16} />
                  <div>
                    <strong>Understand the data</strong>
                    <p>Profile your dataset and check its quality.</p>
                  </div>
                </li>
                <li>
                  <Layers3 size={16} />
                  <div>
                    <strong>Compare the candidates</strong>
                    <p>Run the existing ML pipeline on Modal.</p>
                  </div>
                </li>
                <li>
                  <CircleCheck size={16} />
                  <div>
                    <strong>Get an explained result</strong>
                    <p>
                      A model comparison, diagnostics, and the agent’s report.
                    </p>
                  </div>
                </li>
              </ul>
              <div className="t-training-config">
                <span>
                  <Cloud size={13} /> Compute <strong>Modal</strong>
                </span>
                <span>
                  <Sparkles size={13} /> Agent <strong>Nebius</strong>
                </span>
              </div>
              <div className="t-start-area">
                <button
                  className="r-button r-button-primary t-start"
                  disabled={
                    !dataset ||
                    !target ||
                    !!invalidTarget ||
                    uploading ||
                    starting ||
                    !setup?.ready
                  }
                  onClick={() => void start()}
                >
                  {starting ? (
                    <LoaderCircle className="spin" size={15} />
                  ) : (
                    <Play size={14} />
                  )}{" "}
                  {starting ? "Starting training…" : "Start training"}
                  <ArrowRight size={15} />
                </button>
                <p>
                  Your dataset is saved locally and sent to Modal for training.
                  The agent uses Nebius. Your configured cloud usage applies.
                </p>
              </div>
            </div>
            <div className={`t-setup ${setup?.ready ? "ready" : ""}`}>
              <div>
                <span>
                  {setup?.ready ? (
                    <CircleCheck size={15} />
                  ) : (
                    <Cloud size={15} />
                  )}{" "}
                  {setup === null
                    ? "Checking training setup…"
                    : setup.ready
                      ? "Training environment configured"
                      : "Finish your training setup"}
                </span>
                <button
                  aria-label="Recheck training setup"
                  onClick={() => void refreshSetup()}
                  disabled={checking}
                >
                  <RefreshCw size={13} className={checking ? "spin" : ""} />
                </button>
              </div>
              {setup && !setup.ready ? (
                <ul>
                  {setup.issues.map((issue) => (
                    <li key={issue.label}>
                      <strong>{issue.label}</strong>
                      <p>{issue.detail}</p>
                      {issue.command && <code>{issue.command}</code>}
                    </li>
                  ))}
                </ul>
              ) : (
                setup && (
                  <p>
                    Make sure <code>ml-factory</code> is deployed to your Modal
                    workspace. Authentication and deployment are checked by the
                    pipeline when it starts.
                  </p>
                )
              )}
              <span className="t-setup-note">
                You can upload and configure a dataset before adding
                credentials.
              </span>
            </div>
          </aside>
        </div>
      )}
      <input
        ref={input}
        type="file"
        accept=".csv,text/csv"
        className="sr-only"
        tabIndex={-1}
        aria-label="Upload CSV dataset"
        onChange={(e) => void upload(e.target.files?.[0])}
      />
    </section>
  );
}
function Chevron() {
  return <ArrowRight size={11} />;
}
