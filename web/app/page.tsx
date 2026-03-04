"use client";

import { useState, useRef, useCallback } from "react";
import Image from "next/image";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ------------------------------------------------------------------ */
/*  Static data                                                        */
/* ------------------------------------------------------------------ */

const CLASS_LABELS: Record<string, string> = {
  ippon_seoi_nage: "Ippon Seoi Nage",
  o_goshi: "O Goshi",
  osoto_gari: "Osoto Gari",
  uchi_mata: "Uchi Mata",
};

const CLASS_EMOJI: Record<string, string> = {
  ippon_seoi_nage: "🔵",
  o_goshi: "🟢",
  osoto_gari: "🟣",
  uchi_mata: "🟠",
};

const CLASS_GRADIENT: Record<string, string> = {
  ippon_seoi_nage: "from-blue-500 to-cyan-400",
  o_goshi: "from-emerald-500 to-teal-400",
  osoto_gari: "from-violet-500 to-purple-400",
  uchi_mata: "from-orange-500 to-amber-400",
};

const CLASS_BAR: Record<string, string> = {
  ippon_seoi_nage: "bg-gradient-to-r from-blue-500 to-cyan-400",
  o_goshi: "bg-gradient-to-r from-emerald-500 to-teal-400",
  osoto_gari: "bg-gradient-to-r from-violet-500 to-purple-400",
  uchi_mata: "bg-gradient-to-r from-orange-500 to-amber-400",
};

const MODEL_DISPLAY: Record<
  string,
  { label: string; tag: string; accent: string; ring: string; accuracy: string }
> = {
  x3d_s: {
    label: "X3D-S",
    tag: "Small · Fast",
    accent: "from-cyan-500 to-blue-600",
    ring: "#22d3ee",
    accuracy: "88.1%",
  },
  x3d_m: {
    label: "X3D-M",
    tag: "Medium · Deeper",
    accent: "from-fuchsia-500 to-violet-600",
    ring: "#d946ef",
    accuracy: "75.2%",
  },
};

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface ModelResult {
  predicted_class: string;
  confidence: number;
  scores: Record<string, number>;
  inference_ms: number;
}

interface CompareResult {
  x3d_s: ModelResult;
  x3d_m: ModelResult;
  agree: boolean;
}

type InputMode = "file" | "url";

/* ------------------------------------------------------------------ */
/*  Tiny SVG icons (inline to avoid deps)                              */
/* ------------------------------------------------------------------ */

const IconUpload = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12m0-12L8 8m4-4l4 4" />
  </svg>
);

const IconLink = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101M10.172 13.828a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
  </svg>
);

const IconSearch = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M11 19a8 8 0 100-16 8 8 0 000 16z" />
  </svg>
);

const IconRefresh = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h5M20 20v-5h-5M5.05 15A7.97 7.97 0 014 12a8 8 0 0114.29-4.97M18.95 9A7.97 7.97 0 0120 12a8 8 0 01-14.29 4.97" />
  </svg>
);

/* ------------------------------------------------------------------ */
/*  Confidence ring                                                    */
/* ------------------------------------------------------------------ */

function ConfidenceRing({
  pct,
  color,
  size = 100,
}: {
  pct: number;
  color: string;
  size?: number;
}) {
  const r = (size - 10) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;
  return (
    <svg width={size} height={size} className="rotate-[-90deg]">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#e2e2ea" strokeWidth={8} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={8}
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={offset}
        className="transition-all duration-1000 ease-out"
      />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Page component                                                     */
/* ------------------------------------------------------------------ */

export default function Home() {
  const [mode, setMode] = useState<InputMode>("file");
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [urlInput, setUrlInput] = useState("");
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  /* handlers */
  const handleFile = (f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    setVideoUrl(URL.createObjectURL(f));
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0]);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  }, []);

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  };
  const onDragLeave = () => setDragging(false);

  const hasInput = mode === "file" ? !!file : urlInput.trim().length > 0;

  const predict = async () => {
    if (!hasInput) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let res: Response;
      if (mode === "file" && file) {
        const form = new FormData();
        form.append("file", file);
        res = await fetch(`${API_BASE}/compare`, {
          method: "POST",
          body: form,
        });
      } else {
        res = await fetch(`${API_BASE}/compare-url`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: urlInput.trim() }),
        });
      }
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setVideoUrl(null);
    setUrlInput("");
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const switchMode = (m: InputMode) => {
    reset();
    setMode(m);
  };

  const isYouTubeOrSocial = /youtube\.com|youtu\.be|tiktok\.com|instagram\.com/i.test(
    urlInput
  );
  const previewUrl = mode === "file" ? videoUrl : urlInput.trim() || null;

  /* ---------------------------------------------------------------- */
  return (
    <main className="min-h-screen flex flex-col items-center px-4 py-16 sm:py-20">
      {/* ---- Header ---- */}
      <header className="text-center max-w-2xl mx-auto mb-14 animate-fade-in">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-[#6d5aff] to-[#ff5ae0] mb-6 shadow-lg shadow-purple-500/20">
          <span className="text-3xl">🥋</span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-gray-900 via-gray-700 to-gray-500 bg-clip-text text-transparent">
          Judo Throw Classifier
        </h1>
        <p className="text-[var(--text-muted)] mt-3 text-base sm:text-lg leading-relaxed">
          Upload a video or paste a link — dual X3D models classify your throw
          in real time
        </p>

        {/* Class pills */}
        <div className="flex flex-wrap gap-2 justify-center mt-5">
          {Object.entries(CLASS_LABELS).map(([key, label]) => (
            <span
              key={key}
              className={`text-xs font-semibold px-3 py-1 rounded-full bg-gradient-to-r ${CLASS_GRADIENT[key]} text-white/90 shadow-sm`}
            >
              {CLASS_EMOJI[key]} {label}
            </span>
          ))}
        </div>
      </header>

      {/* ---- Content area ---- */}
      <div className="w-full max-w-4xl space-y-6 animate-fade-in animate-delay-100">
        {/* Mode tabs */}
        {!result && (
          <div className="flex justify-center">
            <div className="inline-flex glass rounded-full p-1 gap-1">
              {(["file", "url"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => switchMode(m)}
                  className={`flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-semibold transition-all duration-300 ${
                    mode === m
                      ? "bg-[var(--accent)] text-white shadow-md shadow-purple-500/25"
                      : "text-[var(--text-muted)] hover:text-gray-900"
                  }`}
                >
                  {m === "file" ? <IconUpload /> : <IconLink />}
                  {m === "file" ? "Upload File" : "Paste URL"}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* FILE upload zone */}
        {mode === "file" && !file && !result && (
          <div
            onClick={() => inputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            className={`glass glass-hover rounded-3xl p-14 sm:p-20 text-center cursor-pointer
              transition-all duration-300 group
              ${dragging ? "border-[var(--accent)] bg-[var(--accent-soft)] scale-[1.01]" : ""}`}
          >
            <div className="w-20 h-20 mx-auto rounded-2xl bg-[var(--surface-2)] flex items-center justify-center mb-5 group-hover:scale-105 transition-transform duration-300">
              <span className="text-4xl">🎬</span>
            </div>
            <p className="text-xl font-bold text-gray-800">
              Drop your video here
            </p>
            <p className="text-[var(--text-muted)] mt-1.5">
              or click to browse files
            </p>
            <p className="text-[var(--text-muted)] text-xs mt-4 font-mono">
              MP4 · AVI · MOV · MKV
            </p>
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              onChange={onFileChange}
              className="hidden"
            />
          </div>
        )}

        {/* FILE preview */}
        {mode === "file" && file && !result && (
          <div className="glass rounded-3xl overflow-hidden animate-fade-in">
            <video
              src={videoUrl!}
              controls
              className="w-full max-h-72 object-contain bg-gray-100"
            />
            <div className="p-5 flex items-center justify-between">
              <div className="min-w-0">
                <p className="font-semibold text-gray-800 truncate">
                  {file.name}
                </p>
                <p className="text-[var(--text-muted)] text-sm">
                  {(file.size / 1024 / 1024).toFixed(1)} MB
                </p>
              </div>
              <button
                onClick={reset}
                className="text-[var(--text-muted)] hover:text-gray-900 transition text-sm flex items-center gap-1.5"
              >
                <IconRefresh /> Change
              </button>
            </div>
          </div>
        )}

        {/* URL input */}
        {mode === "url" && !result && (
          <div className="glass rounded-3xl p-6 sm:p-8 space-y-5 animate-fade-in">
            <div className="relative">
              <input
                type="url"
                value={urlInput}
                onChange={(e) => {
                  setUrlInput(e.target.value);
                  setError(null);
                }}
                placeholder="https://youtube.com/shorts/... or TikTok link"
                className="w-full bg-[var(--surface-2)] border border-[var(--border)] rounded-2xl pl-5 pr-14 py-4 text-gray-900 placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent)]/20 transition font-medium"
              />
              <div className="absolute right-4 top-1/2 -translate-y-1/2 text-[var(--text-muted)]">
                <IconLink />
              </div>
            </div>

            {/* Platform badges */}
            <div className="flex flex-wrap gap-2">
              {[
                { icon: "▶", name: "YouTube", bg: "bg-red-500/10 text-red-400 border-red-500/20" },
                { icon: "♪", name: "TikTok", bg: "bg-white/5 text-gray-300 border-white/10" },
                { icon: "◎", name: "Instagram", bg: "bg-fuchsia-500/10 text-fuchsia-400 border-fuchsia-500/20" },
                { icon: "⊕", name: "Direct URL", bg: "bg-blue-500/10 text-blue-400 border-blue-500/20" },
              ].map((p) => (
                <span
                  key={p.name}
                  className={`text-xs font-medium px-2.5 py-1 rounded-lg border ${p.bg}`}
                >
                  {p.icon} {p.name}
                </span>
              ))}
            </div>

            {/* Direct-link preview */}
            {urlInput.trim() && !isYouTubeOrSocial && (
              <div className="rounded-2xl overflow-hidden border border-[var(--border)]">
                <video
                  src={urlInput.trim()}
                  controls
                  className="w-full max-h-60 object-contain bg-gray-100"
                  onError={() => {}}
                />
              </div>
            )}
          </div>
        )}

        {/* Predict button */}
        {hasInput && !result && (
          <button
            onClick={predict}
            disabled={loading}
            className="w-full py-4 rounded-2xl font-bold text-base
              bg-gradient-to-r from-[#6d5aff] to-[#a855f7]
              hover:from-[#7c6bff] hover:to-[#b76ef8]
              disabled:from-[var(--surface-2)] disabled:to-[var(--surface-2)] disabled:text-[var(--text-muted)]
              transition-all duration-300 shadow-lg shadow-purple-600/20
              flex items-center justify-center gap-3 animate-fade-in animate-delay-200"
          >
            {loading ? (
              <>
                <svg
                  className="animate-spin h-5 w-5"
                  viewBox="0 0 24 24"
                  fill="none"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8v8H4z"
                  />
                </svg>
                <span>
                  {mode === "url"
                    ? "Downloading & analysing…"
                    : "Running both models…"}
                </span>
              </>
            ) : (
              <>
                <IconSearch /> Compare Models
              </>
            )}
          </button>
        )}

        {/* Error */}
        {error && (
          <div className="glass rounded-2xl p-4 border-red-500/30 bg-red-500/5 text-red-400 text-sm flex items-start gap-3 animate-fade-in">
            <span className="text-lg leading-none">⚠</span>
            <span>{error}</span>
          </div>
        )}

        {/* ---- RESULTS ---- */}
        {result && (
          <div className="space-y-8 animate-fade-in">
            {/* Video playback */}
            {previewUrl && !isYouTubeOrSocial && (
              <div className="glass rounded-3xl overflow-hidden animate-fade-in">
                <video
                  src={previewUrl}
                  controls
                  className="w-full max-h-60 object-contain bg-gray-100"
                />
                {mode === "url" && (
                  <div className="px-5 py-3 border-t border-[var(--border)]">
                    <p className="text-[var(--text-muted)] text-xs truncate font-mono">
                      {urlInput}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Agreement badge */}
            <div className="text-center animate-fade-in animate-delay-100">
              {result.agree ? (
                <span className="inline-flex items-center gap-2.5 glass px-6 py-3 rounded-full border-emerald-500/30 bg-emerald-500/5 text-emerald-400 text-sm font-semibold">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot inline-block" />
                  Both models agree:{" "}
                  <strong>{CLASS_LABELS[result.x3d_s.predicted_class]}</strong>
                </span>
              ) : (
                <span className="inline-flex items-center gap-2.5 glass px-6 py-3 rounded-full border-amber-500/30 bg-amber-500/5 text-amber-400 text-sm font-semibold">
                  <span className="w-2 h-2 rounded-full bg-amber-400 pulse-dot inline-block" />
                  Models disagree — X3D-S:{" "}
                  {CLASS_LABELS[result.x3d_s.predicted_class]} vs X3D-M:{" "}
                  {CLASS_LABELS[result.x3d_m.predicted_class]}
                </span>
              )}
            </div>

            {/* Model cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 animate-fade-in animate-delay-200">
              {(["x3d_s", "x3d_m"] as const).map((modelKey) => {
                const r = result[modelKey];
                const meta = MODEL_DISPLAY[modelKey];
                const pctConf = r.confidence * 100;

                return (
                  <div
                    key={modelKey}
                    className="glass glass-hover rounded-3xl overflow-hidden transition-all duration-300"
                  >
                    {/* Gradient header strip */}
                    <div
                      className={`h-1.5 bg-gradient-to-r ${meta.accent}`}
                    />

                    <div className="p-6 space-y-6">
                      {/* Model info row */}
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-bold tracking-tight">
                            {meta.label}
                          </h3>
                          <p className="text-[var(--text-muted)] text-xs mt-0.5">
                            {meta.tag} · Test acc {meta.accuracy}
                          </p>
                        </div>
                        <span className="text-[var(--text-muted)] text-xs font-mono bg-[var(--surface-2)] px-2.5 py-1 rounded-lg">
                          {r.inference_ms}ms
                        </span>
                      </div>

                      {/* Center confidence ring */}
                      <div className="flex flex-col items-center gap-1">
                        <div className="relative">
                          <ConfidenceRing
                            pct={pctConf}
                            color={meta.ring}
                            size={110}
                          />
                          <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span className="text-2xl font-extrabold">
                              {pctConf.toFixed(0)}%
                            </span>
                          </div>
                        </div>
                        <p className="text-sm font-semibold mt-2">
                          {CLASS_LABELS[r.predicted_class]}
                        </p>
                        <p className="text-[var(--text-muted)] text-xs">
                          Predicted class
                        </p>
                      </div>

                      {/* Score breakdown */}
                      <div className="space-y-3">
                        {Object.entries(r.scores)
                          .sort(([, a], [, b]) => b - a)
                          .map(([cls, score]) => {
                            const isTop = cls === r.predicted_class;
                            return (
                              <div key={cls}>
                                <div className="flex justify-between text-xs mb-1.5">
                                  <span
                                    className={`font-medium ${isTop ? "text-gray-900" : "text-[var(--text-muted)]"}`}
                                  >
                                    {CLASS_EMOJI[cls]} {CLASS_LABELS[cls]}
                                  </span>
                                  <span
                                    className={`font-mono ${isTop ? "text-gray-900 font-bold" : "text-[var(--text-muted)]"}`}
                                  >
                                    {(score * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="w-full bg-[var(--surface-2)] rounded-full h-1.5 overflow-hidden">
                                  <div
                                    className={`h-full rounded-full animate-bar ${isTop ? CLASS_BAR[cls] : "bg-[var(--text-muted)]/30"}`}
                                    style={{ width: `${score * 100}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Try another */}
            <button
              onClick={reset}
              className="w-full py-3.5 rounded-2xl glass glass-hover text-[var(--text-muted)] hover:text-gray-900 transition-all duration-300 text-sm font-semibold flex items-center justify-center gap-2 animate-fade-in animate-delay-300"
            >
              <IconRefresh /> Try another video
            </button>
          </div>
        )}
      </div>

      {/* ================================================================== */}
      {/* MODEL ANALYSIS SECTION                                            */}
      {/* ================================================================== */}
      <section className="w-full max-w-5xl mt-24 space-y-10 animate-fade-in">
        {/* Section heading */}
        <div className="text-center">
          <span className="text-xs font-mono tracking-widest text-[var(--accent)] uppercase">
            Training Analysis
          </span>
          <h2 className="text-3xl sm:text-4xl font-extrabold mt-2 bg-gradient-to-r from-gray-900 to-gray-500 bg-clip-text text-transparent">
            Why X3D-S Outperforms X3D-M
          </h2>
          <p className="text-[var(--text-muted)] mt-3 max-w-2xl mx-auto leading-relaxed">
            With only ~600 training videos (~150 per class), the larger X3D-M
            model <strong className="text-amber-400">memorizes</strong> the
            training data instead of learning generalizable patterns — a
            classic case of <strong className="text-amber-400">overfitting</strong>.
          </p>
        </div>

        {/* Key stats comparison */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {[
            {
              model: "X3D-S",
              acc: "88.1%",
              verdict: "Recommended",
              color: "from-cyan-500 to-blue-600",
              tag: "text-emerald-400",
              tagBg: "bg-emerald-500/10 border-emerald-500/20",
              details: [
                "Smaller capacity → less overfitting",
                "Smooth validation loss convergence",
                "Best F1: Osoto Gari (0.98)",
                "Weakest: Uchi Mata (recall 0.73)",
              ],
            },
            {
              model: "X3D-M",
              acc: "75.2%",
              verdict: "Overfits",
              color: "from-fuchsia-500 to-violet-600",
              tag: "text-amber-400",
              tagBg: "bg-amber-500/10 border-amber-500/20",
              details: [
                "Too powerful for ~600 videos",
                "Validation loss oscillates wildly",
                "Train acc reaches ~100% (memorization)",
                "Weakest: O Goshi (F1 0.57)",
              ],
            },
          ].map((m) => (
            <div key={m.model} className="glass rounded-3xl overflow-hidden">
              <div className={`h-1.5 bg-gradient-to-r ${m.color}`} />
              <div className="p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xl font-bold">{m.model}</h3>
                    <p className="text-3xl font-extrabold mt-1">{m.acc}</p>
                  </div>
                  <span
                    className={`text-xs font-semibold px-3 py-1 rounded-full border ${m.tagBg} ${m.tag}`}
                  >
                    {m.verdict}
                  </span>
                </div>
                <ul className="space-y-1.5">
                  {m.details.map((d) => (
                    <li
                      key={d}
                      className="text-[var(--text-muted)] text-sm flex items-start gap-2"
                    >
                      <span className="text-[var(--accent)] mt-0.5">›</span>
                      {d}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>

        {/* Overfitting explanation card */}
        <div className="glass rounded-3xl p-6 sm:p-8 border-amber-500/10">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center shrink-0">
              <span className="text-xl">⚠</span>
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-bold text-amber-400">
                What is Overfitting?
              </h3>
              <p className="text-[var(--text-muted)] text-sm leading-relaxed">
                When a model is too complex relative to the amount of training
                data, it <em>memorizes</em> individual examples rather than
                learning general patterns. The training accuracy shoots to ~100%
                while validation accuracy plateaus or drops. This is exactly
                what happened with X3D-M — the training loss converges to near
                zero, but the validation loss remains high and oscillates wildly.
              </p>
              <p className="text-[var(--text-muted)] text-sm leading-relaxed">
                <strong className="text-gray-800">X3D-S</strong>, being a
                smaller model, has less capacity to memorize and is forced to
                learn the actual throw patterns, resulting in much better
                generalization (88.1% vs 75.2%).
              </p>
            </div>
          </div>
        </div>

        {/* Training Curves — side by side */}
        <div>
          <h3 className="text-xl font-bold mb-1">Training Curves</h3>
          <p className="text-[var(--text-muted)] text-sm mb-5">
            Notice how X3D-M&apos;s validation loss (red) oscillates and never
            converges, while X3D-S shows much smoother learning.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {[
              { key: "x3d_s", label: "X3D-S" },
              { key: "x3d_m", label: "X3D-M" },
            ].map((m) => (
              <div key={m.key} className="glass rounded-2xl overflow-hidden">
                <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between">
                  <span className="text-sm font-semibold">{m.label}</span>
                  <span className="text-xs text-[var(--text-muted)] font-mono">
                    training_curves.png
                  </span>
                </div>
                <Image
                  src={`/plots/${m.key}/training_curves.png`}
                  alt={`${m.label} training curves`}
                  width={800}
                  height={400}
                  className="w-full h-auto"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Confusion Matrices */}
        <div>
          <h3 className="text-xl font-bold mb-1">Confusion Matrices</h3>
          <p className="text-[var(--text-muted)] text-sm mb-5">
            X3D-S makes fewer misclassifications across all classes. X3D-M
            particularly struggles with O Goshi (F1 = 0.57).
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {[
              { key: "x3d_s", label: "X3D-S" },
              { key: "x3d_m", label: "X3D-M" },
            ].map((m) => (
              <div key={m.key} className="glass rounded-2xl overflow-hidden">
                <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between">
                  <span className="text-sm font-semibold">{m.label}</span>
                  <span className="text-xs text-[var(--text-muted)] font-mono">
                    confusion_matrix.png
                  </span>
                </div>
                <Image
                  src={`/plots/${m.key}/confusion_matrix.png`}
                  alt={`${m.label} confusion matrix`}
                  width={800}
                  height={600}
                  className="w-full h-auto"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Per-Class Accuracy */}
        <div>
          <h3 className="text-xl font-bold mb-1">Per-Class Accuracy</h3>
          <p className="text-[var(--text-muted)] text-sm mb-5">
            X3D-S achieves consistent performance across all four throw types,
            while X3D-M drops significantly on O Goshi.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {[
              { key: "x3d_s", label: "X3D-S" },
              { key: "x3d_m", label: "X3D-M" },
            ].map((m) => (
              <div key={m.key} className="glass rounded-2xl overflow-hidden">
                <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between">
                  <span className="text-sm font-semibold">{m.label}</span>
                  <span className="text-xs text-[var(--text-muted)] font-mono">
                    per_class_accuracy.png
                  </span>
                </div>
                <Image
                  src={`/plots/${m.key}/per_class_accuracy.png`}
                  alt={`${m.label} per-class accuracy`}
                  width={800}
                  height={400}
                  className="w-full h-auto"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Classification Report Table */}
        <div>
          <h3 className="text-xl font-bold mb-1">Classification Report</h3>
          <p className="text-[var(--text-muted)] text-sm mb-5">
            Full precision, recall, and F1 scores for each throw class.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {[
              {
                model: "X3D-S (88.1%)",
                accent: "from-cyan-500 to-blue-600",
                rows: [
                  { cls: "Ippon Seoi Nage", p: "0.81", r: "0.84", f1: "0.82" },
                  { cls: "O Goshi", p: "0.79", r: "0.96", f1: "0.87" },
                  { cls: "Osoto Gari", p: "1.00", r: "0.97", f1: "0.98" },
                  { cls: "Uchi Mata", p: "0.94", r: "0.73", f1: "0.82" },
                ],
              },
              {
                model: "X3D-M (75.2%)",
                accent: "from-fuchsia-500 to-violet-600",
                rows: [
                  { cls: "Ippon Seoi Nage", p: "0.68", r: "0.76", f1: "0.72" },
                  { cls: "O Goshi", p: "0.67", r: "0.50", f1: "0.57" },
                  { cls: "Osoto Gari", p: "0.81", r: "0.87", f1: "0.84" },
                  { cls: "Uchi Mata", p: "0.83", r: "0.86", f1: "0.84" },
                ],
              },
            ].map((m) => (
              <div key={m.model} className="glass rounded-2xl overflow-hidden">
                <div className={`h-1 bg-gradient-to-r ${m.accent}`} />
                <div className="px-4 py-3 border-b border-[var(--border)]">
                  <span className="text-sm font-semibold">{m.model}</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-[var(--text-muted)] text-xs">
                        <th className="text-left px-4 py-2 font-medium">Class</th>
                        <th className="text-right px-4 py-2 font-medium">Prec</th>
                        <th className="text-right px-4 py-2 font-medium">Recall</th>
                        <th className="text-right px-4 py-2 font-medium">F1</th>
                      </tr>
                    </thead>
                    <tbody>
                      {m.rows.map((row) => (
                        <tr key={row.cls} className="border-t border-[var(--border)]/50">
                          <td className="px-4 py-2.5 font-medium text-gray-800">{row.cls}</td>
                          <td className="px-4 py-2.5 text-right font-mono">{row.p}</td>
                          <td className="px-4 py-2.5 text-right font-mono">{row.r}</td>
                          <td className="px-4 py-2.5 text-right font-mono font-bold">{row.f1}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Dataset info */}
        <div className="glass rounded-3xl p-6 sm:p-8">
          <h3 className="text-lg font-bold mb-4">Dataset Summary</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            {[
              { label: "Train Videos", value: "598" },
              { label: "Val Videos", value: "102" },
              { label: "Test Videos", value: "101" },
              { label: "Classes", value: "4" },
            ].map((s) => (
              <div key={s.label} className="bg-[var(--surface-2)] rounded-xl p-4">
                <p className="text-2xl font-extrabold">{s.value}</p>
                <p className="text-[var(--text-muted)] text-xs mt-1">{s.label}</p>
              </div>
            ))}
          </div>
          <p className="text-[var(--text-muted)] text-sm mt-4 leading-relaxed">
            The dataset contains short video clips of four judo throw techniques.
            With ~150 videos per class, simpler architectures like X3D-S generalize
            significantly better than larger models that tend to overfit.
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-20 text-center animate-fade-in animate-delay-400">
        <p className="text-[var(--text-muted)]/50 text-xs font-mono tracking-wide">
          DATA.ML.330 Media Analysis — Group 4
        </p>
        <p className="text-[var(--text-muted)]/30 text-[10px] mt-1 font-mono">
          X3D-S (88.1%) · X3D-M (75.2%) · PyTorch + FastAPI + Next.js
        </p>
      </footer>
    </main>
  );
}
