"use client";

import { useState, useRef, useCallback } from "react";

const CLASS_LABELS: Record<string, string> = {
  ippon_seoi_nage: "Ippon Seoi Nage",
  o_goshi: "O Goshi",
  osoto_gari: "Osoto Gari",
  uchi_mata: "Uchi Mata",
};

const CLASS_COLORS: Record<string, string> = {
  ippon_seoi_nage: "bg-blue-500",
  o_goshi: "bg-emerald-500",
  osoto_gari: "bg-violet-500",
  uchi_mata: "bg-orange-500",
};

interface PredictResult {
  predicted_class: string;
  confidence: number;
  scores: Record<string, number>;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

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

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  const predict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("http://localhost:8000/predict", { method: "POST", body: form });
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || "Prediction failed"); }
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null); setVideoUrl(null); setResult(null); setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center py-12 px-4">
      {/* Header */}
      <div className="mb-10 text-center">
        <div className="text-5xl mb-3">🥋</div>
        <h1 className="text-4xl font-bold tracking-tight">Judo Throw Classifier</h1>
        <p className="text-gray-400 mt-2 text-lg">Upload a judo video and AI will identify the throw technique</p>
        <div className="flex flex-wrap gap-2 justify-center mt-4">
          {Object.entries(CLASS_LABELS).map(([key, label]) => (
            <span key={key} className={`text-xs px-2 py-1 rounded-full text-white font-medium ${CLASS_COLORS[key]}`}>{label}</span>
          ))}
        </div>
      </div>

      <div className="w-full max-w-2xl space-y-6">
        {/* Upload zone */}
        {!file ? (
          <div
            onClick={() => inputRef.current?.click()}
            onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
            className={`border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all duration-200 ${
              dragging ? "border-blue-400 bg-blue-950/30 scale-[1.01]" : "border-gray-700 hover:border-gray-500 bg-gray-900"
            }`}
          >
            <div className="text-5xl mb-4">🎬</div>
            <p className="text-xl font-semibold text-gray-200">Drop your video here</p>
            <p className="text-gray-500 mt-1">or click to browse</p>
            <p className="text-gray-600 text-sm mt-3">MP4, AVI, MOV, MKV supported</p>
            <input ref={inputRef} type="file" accept="video/*" onChange={onFileChange} className="hidden" />
          </div>
        ) : (
          <div className="bg-gray-900 rounded-2xl overflow-hidden border border-gray-800">
            <video src={videoUrl!} controls className="w-full max-h-72 object-contain bg-black" />
            <div className="p-4 flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-200 truncate max-w-xs">{file.name}</p>
                <p className="text-gray-500 text-sm">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
              </div>
              <button onClick={reset} className="text-gray-500 hover:text-gray-300 transition text-sm underline">Change video</button>
            </div>
          </div>
        )}

        {/* Predict button */}
        {file && !result && (
          <button
            onClick={predict} disabled={loading}
            className="w-full py-4 rounded-2xl font-bold text-lg bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 transition-all duration-200 flex items-center justify-center gap-3"
          >
            {loading ? (
              <><svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>Analysing video…</>
            ) : <>🔍 Classify Throw</>}
          </button>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-950 border border-red-800 text-red-300 rounded-2xl p-4 text-sm">⚠️ {error}</div>
        )}

        {/* Result */}
        {result && (
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-5">
            <div className={`rounded-xl p-5 border ${CLASS_COLORS[result.predicted_class]} bg-opacity-10 border-opacity-30`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-300 uppercase tracking-widest font-semibold">Predicted</p>
                  <h2 className="text-3xl font-bold mt-1">{CLASS_LABELS[result.predicted_class]}</h2>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-300">Confidence</p>
                  <p className="text-4xl font-bold mt-1">{(result.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>

            <div>
              <p className="text-sm text-gray-400 uppercase tracking-widest font-semibold mb-3">All scores</p>
              <div className="space-y-3">
                {Object.entries(result.scores).sort(([, a], [, b]) => b - a).map(([cls, score]) => (
                  <div key={cls}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className={`font-medium ${cls === result.predicted_class ? "text-white" : "text-gray-400"}`}>{CLASS_LABELS[cls]}</span>
                      <span className={cls === result.predicted_class ? "text-white font-bold" : "text-gray-500"}>{(score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div className={`h-2 rounded-full transition-all duration-700 ${CLASS_COLORS[cls]}`} style={{ width: `${score * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <button onClick={reset} className="w-full py-3 rounded-xl border border-gray-700 text-gray-300 hover:bg-gray-800 transition text-sm font-medium">
              Try another video
            </button>
          </div>
        )}
      </div>

      <p className="text-gray-700 text-sm mt-12">DATA.ML.330 Media Analysis — Group 4 · X3D-S · 88.1% test accuracy</p>
    </main>
  );
}
