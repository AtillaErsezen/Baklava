import { useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  Check,
  Layers3,
  Menu,
  Play,
  ScanSearch,
  ShieldCheck,
  Trophy,
  Upload,
  Workflow,
  X,
} from "lucide-react";
import Demo from "./components/Demo";
import Results from "./results/Results";
import BusinessFlow from "./components/BusinessFlow";
import LayerIllustration from "./components/LayerIllustration";
import "./App.css";

function Brand({ footer = false }: { footer?: boolean }) {
  return (
    <a
      className={`brand ${footer ? "brand-footer" : ""}`}
      href="#"
      aria-label="Baklava home"
    >
      <svg viewBox="0 0 36 40" fill="none" aria-hidden="true">
        <path
          d="m3 24 15 9 15-9M3 18l15 9 15-9M3 12l15 9 15-9L18 3 3 12Z"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
      <span>
        baklava<span className="brand-period">.</span>
      </span>
    </a>
  );
}

const steps = [
  {
    number: "01",
    Icon: Upload,
    title: "Bring your data.",
    text: "Start with a CSV or Parquet file and choose what you want to predict. Churn, prices, demand — your question sets the direction.",
    caption: "YOUR DATA. YOUR QUESTION.",
  },
  {
    number: "02",
    Icon: Workflow,
    title: "Let the layers work.",
    text: "Baklava profiles your data, plans experiments, and compares proven models in parallel. Every decision has a reason.",
    caption: "PROFILE → EXPERIMENT → REFINE",
  },
  {
    number: "03",
    Icon: Trophy,
    title: "Meet your best fit.",
    text: "Get a fitted model, a clear comparison, and a plain-language report. Understand what worked, and where to go next.",
    caption: "A MODEL. AND THE WHY BEHIND IT.",
  },
];

function Landing() {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setMenuOpen(false);
        menuButton.current?.focus();
      }
    }
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [menuOpen]);

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="site-header section-shell">
        <Brand />
        <nav className="desktop-nav" aria-label="Main navigation">
          <a href="#how-it-works">How it works</a>
          <a href="#why-baklava">Why Baklava</a>
          <a href="#results">
            Results <ArrowUpRight size={13} />
          </a>
        </nav>
        <a className="button button-dark nav-cta" href="#training">
          Start training <ArrowUpRight size={15} />
        </a>
        <button
          className="menu-toggle"
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          aria-expanded={menuOpen}
          aria-controls="mobile-navigation"
          ref={menuButton}
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? <X size={23} /> : <Menu size={23} />}
        </button>
        {menuOpen && (
          <nav
            className="mobile-nav"
            id="mobile-navigation"
            aria-label="Mobile navigation"
          >
            <a href="#how-it-works" onClick={() => setMenuOpen(false)}>
              How it works
            </a>
            <a href="#why-baklava" onClick={() => setMenuOpen(false)}>
              Why Baklava
            </a>
            <a href="#results" onClick={() => setMenuOpen(false)}>
              View results <ArrowUpRight size={16} />
            </a>
            <a href="#demo" onClick={() => setMenuOpen(false)}>
              Explore the demo <ArrowUpRight size={16} />
            </a>
          </nav>
        )}
      </header>
      <main id="main">
        <section className="hero section-shell" aria-labelledby="hero-title">
          <div className="hero-copy">
            <p className="eyebrow">
              <span /> AUTONOMOUS MACHINE LEARNING
            </p>
            <h1 id="hero-title">
              Your data.
              <br />
              Layers of
              <br />
              <em>intelligence.</em>
            </h1>
            <p className="hero-description">
              An autonomous ML engineer for your tabular data. From the first
              experiment to the best-fit model — with the reasoning behind every
              decision.
            </p>
            <div className="hero-actions">
              <a href="#demo" className="button button-primary">
                Explore the demo <ArrowUpRight size={17} />
              </a>
              <a href="#how-it-works" className="quiet-link">
                <span className="play-icon">
                  <Play size={11} fill="currentColor" />
                </span>
                How it works
              </a>
            </div>
            <div className="hero-footnote">
              <Check size={13} /> Classification. Regression. A clearer way
              forward.
            </div>
          </div>
          <LayerIllustration />
          <a
            className="hero-scroll"
            href="#how-it-works"
            aria-label="Discover how Baklava works"
          >
            <ArrowDown size={17} />
          </a>
        </section>
        <div className="model-strip section-shell">
          <p>
            PROVEN MODEL FAMILIES.
            <br />
            <strong>ORCHESTRATED BY BAKLAVA.</strong>
          </p>
          <div className="model-families">
            <span className="model-wordmark xgboost">
              XGBoost<span className="wordmark-symbol">↗</span>
            </span>
            <span className="model-wordmark lightgbm">
              <span className="little-bars">
                <i />
                <i />
                <i />
                <i />
              </span>
              LightGBM
            </span>
            <span className="model-wordmark forest">
              <Workflow size={22} />
              Random Forest
            </span>
            <span className="model-wordmark sklearn">
              <span className="sklearn-mark" />
              scikit-learn
            </span>
          </div>
        </div>
        <section
          className="how-section section-shell"
          id="how-it-works"
          aria-labelledby="how-title"
        >
          <BusinessFlow />
          <div className="section-heading">
            <div>
              <p className="eyebrow">
                <span /> FROM RAW DATA TO READY
              </p>
              <h2 id="how-title">
                From a dataset
                <br />
                <em>to a decision.</em>
              </h2>
            </div>
            <p>
              One connected workflow.
              <br />
              From your first question to an explained result.
            </p>
          </div>
          <div className="steps-grid">
            {steps.map(({ number, Icon, title, text, caption }) => (
              <article className="process-step" key={number}>
                <div className="step-heading">
                  <span className="process-icon">
                    <Icon size={23} strokeWidth={1.5} />
                  </span>
                  <span className="step-number">{number}</span>
                </div>
                <h3>{title}</h3>
                <p>{text}</p>
                <span className="step-caption">{caption}</span>
              </article>
            ))}
          </div>
        </section>
        <Demo />
        <section
          className="why-section section-shell"
          id="why-baklava"
          aria-labelledby="why-title"
        >
          <div className="why-intro">
            <p className="eyebrow">
              <span /> THOUGHTFUL BY DESIGN
            </p>
            <h2 id="why-title">
              Intelligence you
              <br />
              <em>can interrogate.</em>
            </h2>
            <p>
              A great result is only useful if you know how you got there. Every
              layer of Baklava is built with that in mind.
            </p>
            <a href="#demo" className="text-button">
              Take a closer look <ArrowRight size={16} />
            </a>
          </div>
          <div className="principles">
            <article>
              <ScanSearch size={22} />
              <div>
                <h3>Reasons, not just results.</h3>
                <p>
                  Follow the reasoning behind each experiment and get a report
                  that puts the outcome in context.
                </p>
              </div>
            </article>
            <article>
              <ShieldCheck size={22} />
              <div>
                <h3>A healthy dose of skepticism.</h3>
                <p>
                  Cross-validation, overfitting checks, and leakage warnings
                  help you look beyond a promising score.
                </p>
              </div>
            </article>
            <article>
              <Layers3 size={22} />
              <div>
                <h3>Proven models. Purposeful choices.</h3>
                <p>
                  A focused set of established model families, with bounded
                  experiment rounds to keep the search on track.
                </p>
              </div>
            </article>
          </div>
        </section>
        <section className="closing-section" aria-labelledby="closing-title">
          <div className="closing-inner section-shell">
            <div className="closing-decoration" aria-hidden="true">
              <Layers3 size={50} strokeWidth={1} />
            </div>
            <p className="eyebrow">YOUR DATA IS JUST THE BEGINNING.</p>
            <h2 id="closing-title">
              Your next model
              <br />
              <em>starts here.</em>
            </h2>
            <a href="#demo" className="button button-dark">
              Explore the demo <ArrowUpRight size={17} />
            </a>
            <p>Explore a sample run. See what comes next.</p>
          </div>
        </section>
      </main>
      <footer className="site-footer section-shell">
        <div>
          <Brand footer />
          <p>Machine learning. Layer by layer.</p>
        </div>
        <div className="footer-links">
          <a href="#how-it-works">How it works</a>
          <a href="#demo">
            Explore the demo <ArrowUpRight size={12} />
          </a>
        </div>
        <span className="footer-note">
          Made with curiosity. © {new Date().getFullYear()} Baklava
        </span>
      </footer>
    </>
  );
}

function App() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const onHash = () => {
      setHash(window.location.hash);
      if (["#results", "#training", ""].includes(window.location.hash))
        window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const results = ["#results", "#results-content", "#training"].includes(hash);
  useEffect(() => {
    document.title = results
      ? hash === "#training"
        ? "Train a model — Baklava"
        : "Results — Baklava"
      : "Baklava — Machine learning. Layer by layer.";
    if (!results && hash)
      requestAnimationFrame(() =>
        document.getElementById(hash.slice(1))?.scrollIntoView(),
      );
  }, [hash, results]);
  return results ? <Results training={hash === "#training"} /> : <Landing />;
}

export default App;
