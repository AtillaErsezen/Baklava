import { Check, FileSpreadsheet, Sparkles } from "lucide-react";

export default function LayerIllustration() {
  return (
    <div
      className="layer-illustration"
      aria-label="Data passes through layers of analysis, experimentation, and model selection"
      role="img"
    >
      <div className="illustration-grid" />
      <div className="illustration-coordinate coordinate-top">
        BAKLAVA ENGINE / V.01
      </div>
      <svg
        className="layer-svg"
        viewBox="0 0 560 470"
        fill="none"
        aria-hidden="true"
      >
        <defs>
          <pattern
            id="data-grid"
            width="24"
            height="24"
            patternUnits="userSpaceOnUse"
            patternTransform="matrix(1 .57 -1 .57 280 208)"
          >
            <rect x="3" y="3" width="16" height="16" rx="2" fill="#365354" />
          </pattern>
          <pattern
            id="top-grid"
            width="30"
            height="30"
            patternUnits="userSpaceOnUse"
            patternTransform="matrix(1 .57 -1 .57 280 15)"
          >
            <path d="M30 0H0V30" stroke="#3b5555" strokeWidth="1" />
          </pattern>
          <filter
            id="layer-shadow"
            x="-50%"
            y="-100%"
            width="200%"
            height="300%"
          >
            <feGaussianBlur stdDeviation="17" />
          </filter>
        </defs>
        <ellipse
          cx="290"
          cy="396"
          rx="157"
          ry="24"
          fill="#759c66"
          opacity=".12"
          filter="url(#layer-shadow)"
        />
        <path
          d="m75 295 205 118 205-118M75 237l205 118 205-118M75 179l205 118 205-118"
          stroke="#42645e"
          strokeDasharray="3 6"
          opacity=".7"
        />
        <g className="data-layer">
          <path
            d="m107 277 173 100 173-100v13L280 390 107 290Z"
            fill="#132326"
            stroke="#496463"
          />
          <path
            d="m107 277 173-100 173 100-173 100Z"
            fill="#1a2e30"
            stroke="#496463"
          />
          <path d="m126 277 154-89 154 89-154 89Z" fill="url(#data-grid)" />
          <path d="M280 377v13" stroke="#496463" />
        </g>
        <g className="intelligence-layer">
          <path
            d="m107 203 173 100 173-100v16L280 319 107 219Z"
            fill="#698e58"
            stroke="#aecf80"
          />
          <path
            d="m107 203 173-100 173 100-173 100Z"
            fill="#a9cb78"
            stroke="#698e58"
          />
          <path
            d="m132 203 148-85 148 85-148 85Z"
            stroke="#e4ffbb"
            strokeWidth="1"
          />
          <path
            d="m175 179 106 61 103-60M175 224l106-61 103 60M227 149v107m106-108v107"
            stroke="#527b44"
            opacity=".5"
          />
          <path d="m245 203 35-20 35 20-35 20Z" fill="#edffd0" />
          <path
            d="m263 202 12 7 23-13"
            stroke="#426132"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path d="M280 303v16" stroke="#91b266" />
          <circle cx="174" cy="179" r="4" fill="#edffd0" />
          <circle cx="385" cy="223" r="4" fill="#edffd0" />
          <circle cx="228" cy="254" r="4" fill="#edffd0" />
          <circle cx="333" cy="149" r="4" fill="#edffd0" />
        </g>
        <g className="model-layer">
          <path
            d="m107 122 173 100 173-100v10L280 232 107 132Z"
            fill="#152527"
            stroke="#68857a"
          />
          <path
            d="m107 122 173-100 173 100-173 100Z"
            fill="#182c2c"
            fillOpacity=".88"
            stroke="#68857a"
          />
          <path d="m107 122 173-100 173 100-173 100Z" fill="url(#top-grid)" />
          <path
            d="m207 129 33-2 17-27 26 5 27-32 41 3"
            stroke="#d2efa7"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle
            cx="351"
            cy="76"
            r="5"
            fill="#d2efa7"
            stroke="#162727"
            strokeWidth="2"
          />
          <path d="M280 222v10" stroke="#68857a" />
        </g>
        <path
          d="M114 164H52v-44M435 263h61v40"
          stroke="#719080"
          strokeDasharray="3 4"
        />
        <circle cx="114" cy="164" r="3" fill="#719080" />
        <circle cx="435" cy="263" r="3" fill="#719080" />
      </svg>
      <div className="dataset-note">
        <span className="file-icon">
          <FileSpreadsheet size={17} />
        </span>
        <div>
          <strong>your_data.csv</strong>
          <span>CSV / PARQUET INPUT</span>
        </div>
        <Check size={14} className="note-check" />
      </div>
      <div className="model-note">
        <span className="success-icon">
          <Check size={15} />
        </span>
        <div>
          <strong>Your best-fit model</strong>
          <span>VALIDATED. EXPLAINED.</span>
        </div>
        <Sparkles size={15} />
      </div>
      <span className="layer-tag tag-one">UNDERSTAND</span>
      <span className="layer-tag tag-two">EXPERIMENT</span>
      <span className="layer-tag tag-three">LEARN</span>
      <div className="illustration-coordinate coordinate-bottom">
        <span className="tiny-cross">+</span> DATA → EXPERIMENTS → INTELLIGENCE{" "}
        <span className="tiny-cross">+</span>
      </div>
    </div>
  );
}
