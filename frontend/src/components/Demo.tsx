import { useEffect, useRef, useState } from "react";
import {
  CheckCheck,
  ChevronRight,
  CircleCheck,
  Download,
  FileSpreadsheet,
  FlaskConical,
  Layers3,
  LoaderCircle,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Trophy,
} from "lucide-react";
import { examples } from "./demo-data";
import type { ExampleKey } from "./demo-data";

const stages = [
  "Understanding the data",
  "Comparing the candidates",
  "Finding the best fit",
  "Your sample result is ready",
];

export default function Demo() {
  const [selected, setSelected] = useState<ExampleKey>("churn");
  const [stage, setStage] = useState(3);
  const [tab, setTab] = useState<"models" | "report">("models");
  const [replay, setReplay] = useState(0);
  const example = examples[selected];
  const running = stage < 3;
  const modelTab = useRef<HTMLButtonElement>(null);
  const reportTab = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!replay) return;
    const timers = [1, 2, 3].map((next) =>
      window.setTimeout(() => setStage(next), next * 1100),
    );
    return () => timers.forEach(window.clearTimeout);
  }, [replay, selected]);

  function changeExample(key: ExampleKey) {
    setSelected(key);
    setStage(3);
    setReplay(0);
    setTab("models");
  }

  function replayDemo() {
    setStage(0);
    setTab("models");
    setReplay((value) => value + 1);
  }

  function downloadReport() {
    const body = `# Baklava — ${example.label}\n\nIllustrative demo only. These scores are sample values, not results of a live training run.\n\n## Dataset\n- File: ${example.file}\n- Rows: ${example.rows}\n- Input columns: ${example.features}\n- Target: ${example.target}\n- Task: ${example.task}\n\n## Sample comparison\n${example.models.map((model) => `- ${model.name}: ${model.score} ${example.metric}`).join("\n")}\n\n## Summary\n${example.insight}\n\n## Before using a real model\nReview cross-validation variability, inspect potential data leakage, and evaluate on held-out data.\n`;
    const url = URL.createObjectURL(
      new Blob([body], { type: "text/markdown" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `baklava-${selected}-sample-report.md`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <section
      className="demo-section section-shell"
      id="demo"
      aria-labelledby="demo-title"
    >
      <div className="section-heading demo-heading">
        <div>
          <p className="eyebrow">
            <span /> A LOOK INSIDE
          </p>
          <h2 id="demo-title">
            Inside the engine.
            <br />
            <em>Beyond the black box.</em>
          </h2>
        </div>
        <p>
          From a question to a model you can understand.
          <br className="desktop-break" /> Take a look at how it all comes
          together.
        </p>
      </div>
      <a href="#results" className="text-button" style={{ marginBottom: 22 }}>
        Open the results workspace <ChevronRight size={16} />
      </a>
      <div className="demo-window">
        <div className="window-toolbar">
          <div className="window-title">
            <Layers3 size={17} />
            <span>Baklava workspace</span>
            <ChevronRight size={13} />
            <span className="toolbar-file">{example.file}</span>
          </div>
          <span className="demo-badge">
            <span /> INTERACTIVE DEMO
          </span>
        </div>
        <div className="workspace-body">
          <aside className="workspace-sidebar">
            <span className="small-label">PICK A QUESTION</span>
            {(Object.keys(examples) as ExampleKey[]).map((key) => (
              <button
                key={key}
                className={`dataset-choice ${selected === key ? "selected" : ""}`}
                onClick={() => changeExample(key)}
                aria-pressed={selected === key}
              >
                <FileSpreadsheet size={17} />
                <span>{examples[key].label}</span>
                <ChevronRight size={14} />
              </button>
            ))}
            <div className="dataset-details">
              <span className="small-label">THE DATASET</span>
              <dl>
                <div>
                  <dt>Rows</dt>
                  <dd>{example.rows}</dd>
                </div>
                <div>
                  <dt>Input columns</dt>
                  <dd>{example.features}</dd>
                </div>
                <div>
                  <dt>Target</dt>
                  <dd>{example.target}</dd>
                </div>
                <div>
                  <dt>Task</dt>
                  <dd>{example.task}</dd>
                </div>
              </dl>
            </div>
            <div className="sample-disclaimer">
              <FlaskConical size={16} />
              <p>
                A sample Baklava run.
                <br />
                Illustrative scores, no live training.
              </p>
            </div>
          </aside>
          <div className="workspace-main">
            <div className="workspace-heading">
              <div>
                <span className="small-label">
                  {example.task.toUpperCase()}
                </span>
                <h3>{example.question}</h3>
              </div>
              <button
                className="replay-button"
                onClick={replayDemo}
                disabled={running}
              >
                {running ? (
                  <LoaderCircle className="spin" size={14} />
                ) : (
                  <RotateCcw size={14} />
                )}
                <span>{running ? "Running demo" : "Replay demo"}</span>
              </button>
            </div>
            <div className="run-steps" aria-label="Sample workflow">
              {["Profile data", "Run experiments", "Select model"].map(
                (label, index) => (
                  <div
                    key={label}
                    className={
                      stage > index
                        ? "complete"
                        : stage === index
                          ? "active"
                          : ""
                    }
                  >
                    {stage > index ? (
                      <CircleCheck size={16} />
                    ) : stage === index && running ? (
                      <LoaderCircle size={16} className="spin" />
                    ) : (
                      <span className="step-dot" />
                    )}
                    <span>{label}</span>
                    {index < 2 && <span className="step-line" />}
                  </div>
                ),
              )}
            </div>
            <div
              className="demo-tabs"
              role="tablist"
              aria-label="Sample results"
            >
              {(["models", "report"] as const).map((value) => (
                <button
                  key={value}
                  id={`tab-${value}`}
                  role="tab"
                  ref={value === "models" ? modelTab : reportTab}
                  aria-selected={tab === value}
                  aria-controls={`panel-${value}`}
                  tabIndex={tab === value ? 0 : -1}
                  className={tab === value ? "active" : ""}
                  onClick={() => setTab(value)}
                  onKeyDown={(event) => {
                    if (
                      ["ArrowRight", "ArrowLeft", "Home", "End"].includes(
                        event.key,
                      )
                    ) {
                      event.preventDefault();
                      const next =
                        event.key === "Home"
                          ? "models"
                          : event.key === "End"
                            ? "report"
                            : tab === "models"
                              ? "report"
                              : "models";
                      setTab(next);
                      (next === "models"
                        ? modelTab
                        : reportTab
                      ).current?.focus();
                    }
                  }}
                >
                  {value === "models" ? "Model comparison" : "The takeaway"}
                  {value === "models" && <span>4</span>}
                </button>
              ))}
            </div>
            <div
              className="results-panel"
              role="tabpanel"
              id={`panel-${tab}`}
              aria-labelledby={`tab-${tab}`}
              tabIndex={0}
              aria-busy={running}
            >
              {running ? (
                <div className="running-state">
                  <div className="running-icon">
                    <Layers3 size={28} />
                  </div>
                  <h4>{stages[stage]}</h4>
                  <p>Walking through an illustrative Baklava run…</p>
                  <div className="running-progress">
                    <span style={{ width: `${(stage + 1) * 30}%` }} />
                  </div>
                </div>
              ) : tab === "models" ? (
                <>
                  <div className="leaderboard-label">
                    <span>MODEL</span>
                    <span>
                      {example.metric}{" "}
                      <span className="metric-hint">↑ BETTER</span>
                    </span>
                  </div>
                  <div className="model-list">
                    {example.models.map((model, index) => (
                      <div
                        className={`model-row ${index === 0 ? "winner" : ""}`}
                        key={model.name}
                      >
                        <div className="model-name">
                          <span className="model-rank">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          <span>{model.name}</span>
                          {index === 0 && (
                            <span className="best-fit">
                              <Trophy size={10} /> BEST FIT
                            </span>
                          )}
                        </div>
                        <div className="score-bar">
                          <span style={{ width: `${model.width}%` }} />
                        </div>
                        <span className="model-score">{model.score}</span>
                      </div>
                    ))}
                  </div>
                  <div className="result-note">
                    <Sparkles size={16} />
                    <p>
                      Different models. A fair comparison.{" "}
                      <strong>One clear next step.</strong>
                    </p>
                    <CheckCheck size={17} />
                  </div>
                </>
              ) : (
                <div className="report-panel">
                  <span className="report-icon">
                    <Trophy size={20} />
                  </span>
                  <div>
                    <h4>{example.winner} is the best fit in this example.</h4>
                    <p>{example.insight}</p>
                    <p className="report-caveat">
                      A real run includes cross-validation scores, overfitting
                      checks, and a report explaining the selection.
                    </p>
                    <button className="text-button" onClick={downloadReport}>
                      Download sample report <Download size={14} />
                    </button>
                  </div>
                </div>
              )}
            </div>
            <p className="sr-only" role="status">
              {stages[stage]}
            </p>
          </div>
        </div>
      </div>
      <p className="demo-caption">
        <ShieldCheck size={14} /> Illustrative demo. Runs in your browser; no
        data upload or live training.
      </p>
    </section>
  );
}
