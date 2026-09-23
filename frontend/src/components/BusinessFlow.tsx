import { useEffect, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";
import type { PromoScene } from "./promo/createPromoScene";
import "./BusinessFlow.css";

const chapters = [
  {
    name: "Possibility",
    title: "Your data.",
    subtitle: "Full of possibility.",
  },
  { name: "Intelligence", title: "Layers of", subtitle: "intelligence." },
  { name: "Clarity", title: "A clearer path", subtitle: "forward." },
];

export default function BusinessFlow() {
  const container = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const engine = useRef<PromoScene | null>(null);
  const preferences = useRef({ visible: false, paused: false, reduced: false });
  const [paused, setPaused] = useState(false);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const [chapter, setChapter] = useState(0);

  useEffect(() => {
    const element = container.current;
    const surface = canvas.current;
    if (!element || !surface) return;
    let disposed = false;
    let requested = false;
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    preferences.current.reduced = motion.matches;

    function syncPlayback() {
      const { visible, paused: isPaused, reduced } = preferences.current;
      engine.current?.configure(
        visible && !isPaused && !document.hidden,
        reduced,
      );
    }
    function motionChanged() {
      preferences.current.reduced = motion.matches;
      syncPlayback();
    }
    async function loadScene() {
      if (requested) return;
      requested = true;
      try {
        const { createPromoScene } = await import("./promo/createPromoScene");
        if (disposed) return;
        engine.current = createPromoScene(surface!, {
          onChapter: setChapter,
          onProgress: (progress) =>
            element!.style.setProperty("--scene-progress", String(progress)),
          onContextLost: () => {
            setFailed(true);
            setReady(false);
            setChapter(2);
          },
        });
        setReady(true);
        syncPlayback();
      } catch {
        if (!disposed) {
          setFailed(true);
          setChapter(2);
        }
      }
    }
    const preloadObserver = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          void loadScene();
          preloadObserver.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    const visibilityObserver = new IntersectionObserver(
      ([entry]) => {
        preferences.current.visible = entry.intersectionRatio >= 0.12;
        syncPlayback();
      },
      { threshold: 0.12 },
    );
    preloadObserver.observe(element);
    visibilityObserver.observe(element);
    motion.addEventListener("change", motionChanged);
    document.addEventListener("visibilitychange", syncPlayback);
    return () => {
      disposed = true;
      preloadObserver.disconnect();
      visibilityObserver.disconnect();
      motion.removeEventListener("change", motionChanged);
      document.removeEventListener("visibilitychange", syncPlayback);
      engine.current?.dispose();
      engine.current = null;
    };
  }, []);

  function togglePlayback() {
    const next = !paused;
    preferences.current.paused = next;
    setPaused(next);
    engine.current?.configure(
      !next && preferences.current.visible && !document.hidden,
      preferences.current.reduced,
    );
  }

  return (
    <div
      className="business-flow"
      ref={container}
      data-ready={ready}
      data-failed={failed}
      data-chapter={chapter}
    >
      <div
        className="promo-art"
        role="img"
        aria-label="Golden particles gather into floating amber glass layers, then settle into one luminous sculpture. From possibility to clarity."
      >
        <canvas ref={canvas} className="promo-canvas" aria-hidden="true" />
        <div className="promo-fallback" aria-hidden="true">
          <div className="fallback-sculpture">
            {Array.from({ length: 10 }, (_, i) => (
              <i key={i} style={{ top: `${i * 15}px` }} />
            ))}
          </div>
        </div>
        <div className="promo-light" aria-hidden="true" />
        <div className="promo-shade" aria-hidden="true" />
        <div className="promo-grain" aria-hidden="true" />
      </div>
      <div className="promo-wordmark" aria-hidden="true">
        <svg viewBox="0 0 36 40" fill="none">
          <path
            d="m3 24 15 9 15-9M3 18l15 9 15-9M3 12l15 9 15-9L18 3 3 12Z"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinejoin="round"
          />
        </svg>
        <span>baklava.</span>
      </div>
      <div className="promo-copy">
        {chapters.map((item, index) => (
          <div
            className="promo-chapter"
            data-current={chapter === index}
            aria-hidden={chapter !== index}
            key={item.name}
          >
            <p>{item.title}</p>
            <span>{item.subtitle}</span>
          </div>
        ))}
      </div>
      <div className="promo-footer">
        <p>From data to decisions.</p>
        <div className="promo-controls">
          <div
            className="promo-chapters"
            role="group"
            aria-label="Animation scenes"
          >
            {chapters.map((item, index) => (
              <button
                type="button"
                key={item.name}
                aria-label={`Show ${item.name.toLowerCase()} scene`}
                aria-pressed={chapter === index}
                data-current={chapter === index}
                onClick={() => engine.current?.seek(index)}
                disabled={!ready || failed}
              >
                <span />
                <i />
              </button>
            ))}
          </div>
          <button
            type="button"
            className="promo-pause"
            aria-label={paused ? "Play animation" : "Pause animation"}
            onClick={togglePlayback}
            disabled={!ready || failed}
          >
            {paused ? <Play size={13} /> : <Pause size={13} />}
          </button>
        </div>
      </div>
    </div>
  );
}
