import React, { useState, useRef } from "react";

export default function OptivDesktopUI() {
  // ==================== MASKING STATES ====================
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef();

  const pickEndpoint = (fileName) => {
    return fileName.toLowerCase().endsWith(".zip")
      ? "/api/mask-zip"
      : "/api/mask-file";
  };

  const handleUpload = async (file) => {
    if (!file) return;
    const endpoint = pickEndpoint(file.name);
    const form = new FormData();
    form.append("file", file, file.name);

    setUploading(true);
    setStatus("Uploading...");
    setProgress(10);

    try {
      const res = await fetch(endpoint, { method: "POST", body: form });
      setProgress(60);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }

      const disp = res.headers.get("Content-Disposition") || "";
      let fname = "masked_download";
      const match = disp.match(/filename="?([^";]+)"?/i);
      if (match && match[1]) fname = match[1];

      const blob = await res.blob();
      setProgress(90);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fname;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      setProgress(100);
      setStatus("✅ Done — download should start automatically.");
    } catch (e) {
      console.error(e);
      setStatus("❌ Error: " + e.message);
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  // ==================== AI INSIGHTS STATES ====================
  const [insStatus, setInsStatus] = useState("");
  const [insProgress, setInsProgress] = useState(0);
  const [insUploading, setInsUploading] = useState(false);
  const insFileInput = useRef();

  const pickInsightsEndpoint = (fileName) =>
    fileName.toLowerCase().endsWith(".zip")
      ? "/api/insights-zip"
      : "/api/insights-file";

  const handleInsightsUpload = async (file) => {
    if (!file) return;
    const endpoint = pickInsightsEndpoint(file.name);
    const form = new FormData();
    form.append("file", file, file.name);

    setInsUploading(true);
    setInsStatus("Uploading…");
    setInsProgress(10);

    try {
      const res = await fetch(endpoint, { method: "POST", body: form });
      setInsProgress(60);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }

      const disp = res.headers.get("Content-Disposition") || "";
      let fname = "insights_result.xlsx";
      const match = disp.match(/filename=\"?([^\";]+)\"?/i);
      if (match && match[1]) fname = match[1];

      const blob = await res.blob();
      setInsProgress(90);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fname;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      setInsProgress(100);
      setInsStatus("✅ Done — AI Insights Excel downloaded.");
    } catch (e) {
      console.error(e);
      setInsStatus("❌ Error: " + (e.message || e));
    } finally {
      setInsUploading(false);
      setInsProgress(0);
    }
  };

  // ==================== UI ====================
  return (
    <div
      style={{
        background: "#0b0c10",
        color: "#f2f5f7",
        minHeight: "100vh",
        padding: "2rem",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <div
        style={{
          maxWidth: 720,
          margin: "0 auto",
          background: "#111417",
          borderRadius: 16,
          padding: 24,
          boxShadow: "0 10px 30px rgba(0,0,0,.25)",
        }}
      >
        {/* ==================== MASKING BLOCK ==================== */}
        <h1>Optiv File Masking Suite</h1>
        <p style={{ color: "#9fb3c8" }}>
          Upload a <b>PDF</b>, <b>PPTX</b>, <b>XLSX</b>, or <b>image</b> — or a
          <b> .zip</b> folder of multiple files.
        </p>

        <div
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files && e.dataTransfer.files[0];
            if (f) handleUpload(f);
          }}
          onDragOver={(e) => e.preventDefault()}
          style={{
            border: "2px dashed #2a3038",
            borderRadius: 12,
            padding: 28,
            textAlign: "center",
            background: "#0f1317",
            marginBottom: 16,
          }}
        >
          Drop file here or click below
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <input type="file" ref={fileInput} disabled={uploading} />
          <button
            disabled={uploading}
            onClick={() => handleUpload(fileInput.current.files[0])}
            style={{
              background: "#2a7bf6",
              color: "#fff",
              border: 0,
              borderRadius: 10,
              padding: "10px 16px",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            {uploading ? "Processing..." : "Mask & Download"}
          </button>
        </div>

        {uploading && (
          <div style={{ marginTop: 12 }}>
            <progress
              value={progress}
              max="100"
              style={{ width: "100%", height: "12px" }}
            />
          </div>
        )}
        <p style={{ marginTop: 18, opacity: 0.8 }}>{status}</p>
      </div>

      {/* ==================== AI INSIGHTS BLOCK ==================== */}
      <div
        style={{
          maxWidth: 720,
          margin: "2rem auto",
          background: "#111417",
          borderRadius: 16,
          padding: 24,
          boxShadow: "0 10px 30px rgba(0,0,0,.25)",
        }}
      >
        <h2>Optiv AI Insights </h2>
        <p style={{ color: "#9fb3c8" }}>
          Upload a <b>PDF</b>, <b>PPTX</b>, <b>XLSX</b>, <b>image</b> — or a{" "}
          <b>.zip</b> folder of multiple files — to generate a detailed{" "}
          <b>Insights Excel report</b>.
        </p>

        <div
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files && e.dataTransfer.files[0];
            if (f) handleInsightsUpload(f);
          }}
          onDragOver={(e) => e.preventDefault()}
          style={{
            border: "2px dashed #2a3038",
            borderRadius: 12,
            padding: 28,
            textAlign: "center",
            background: "#0f1317",
            marginBottom: 16,
          }}
        >
          Drop file here or click below
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <input type="file" ref={insFileInput} disabled={insUploading} />
          <button
            disabled={insUploading}
            onClick={() => handleInsightsUpload(insFileInput.current.files[0])}
            style={{
              background: "#22a06b",
              color: "#fff",
              border: 0,
              borderRadius: 10,
              padding: "10px 16px",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            {insUploading ? "Processing…" : "Generate Insights"}
          </button>
        </div>

        {insUploading && (
          <div style={{ marginTop: 12 }}>
            <progress
              value={insProgress}
              max="100"
              style={{ width: "100%", height: "12px" }}
            />
          </div>
        )}

        <p style={{ marginTop: 18, opacity: 0.8 }}>{insStatus}</p>
      </div>
    </div>
  );
}
