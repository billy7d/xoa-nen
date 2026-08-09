import { useEffect, useMemo, useState } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { coordinatorCall, isTauriRuntime } from "./bridge";
import EditorCanvas from "./components/EditorCanvas";
import type {
  ExportResult,
  HealthPayload,
  ModelManifest,
  OutputMode,
  PreflightReport,
  ProjectPayload,
  ToolMode,
} from "./types";

const tools: Array<{ id: ToolMode; label: string; key: string }> = [
  { id: "pan", label: "Di chuyển", key: "H" },
  { id: "keep", label: "Giữ", key: "K" },
  { id: "remove", label: "Xóa", key: "E" },
  { id: "wand-keep", label: "Wand giữ", key: "W" },
  { id: "wand-remove", label: "Wand xóa", key: "S" },
];

function basename(path: string): string {
  return path.split(/[\\/]/).pop() || "artwork";
}

function stem(path: string): string {
  return basename(path).replace(/\.[^.]+$/, "");
}

function friendlyError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function App() {
  const [project, setProject] = useState<ProjectPayload | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [models, setModels] = useState<ModelManifest[]>([]);
  const [tool, setTool] = useState<ToolMode>("pan");
  const [tolerance, setTolerance] = useState(30);
  const [softness, setSoftness] = useState(18);
  const [radius, setRadius] = useState(20);
  const [hardness, setHardness] = useState(82);
  const [contiguous, setContiguous] = useState(true);
  const [quality, setQuality] = useState<"FAST" | "QUALITY">("QUALITY");
  const [background, setBackground] = useState<"checker" | "white" | "black" | "garment">("checker");
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string>("Sẵn sàng — mọi xử lý diễn ra trên máy này.");
  const [error, setError] = useState<string | null>(null);
  const [preflight, setPreflight] = useState<PreflightReport | null>(null);
  const [printWidth, setPrintWidth] = useState(12);
  const [printHeight, setPrintHeight] = useState(12);
  const [printUnit, setPrintUnit] = useState<"inch" | "cm">("inch");
  const [trim, setTrim] = useState(false);
  const [padding, setPadding] = useState(0);
  const [panel, setPanel] = useState<"controls" | "preflight" | "export">("controls");

  const run = async <T,>(label: string, operation: () => Promise<T>): Promise<T | null> => {
    setBusy(label);
    setError(null);
    setMessage(label);
    try {
      const result = await operation();
      setMessage(`${label} — hoàn tất`);
      return result;
    } catch (caught) {
      const text = friendlyError(caught);
      setError(text);
      setMessage(`${label} — lỗi`);
      return null;
    } finally {
      setBusy(null);
    }
  };

  useEffect(() => {
    if (!isTauriRuntime()) {
      setError("Hãy chạy `npm run tauri dev` để sử dụng coordinator local và hộp thoại file.");
      return;
    }
    void Promise.all([
      coordinatorCall<HealthPayload>("health"),
      coordinatorCall<{ models: ModelManifest[] }>("list_models"),
    ]).then(([healthResult, modelResult]) => {
      setHealth(healthResult);
      setModels(modelResult.models);
    }).catch((caught) => setError(friendlyError(caught)));
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      const mapping: Record<string, ToolMode> = {
        h: "pan", k: "keep", e: "remove", w: "wand-keep", s: "wand-remove",
      };
      if (mapping[event.key.toLowerCase()]) setTool(mapping[event.key.toLowerCase()]);
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        void (event.shiftKey ? redo() : undo());
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  const importImage = async () => {
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "Ảnh V1", extensions: ["png", "jpg", "jpeg", "webp"] }],
    });
    if (!selected) return;
    const imported = await run("Đang canonical decode ảnh", () =>
      coordinatorCall<ProjectPayload>("import_image", { path: selected }),
    );
    if (imported) {
      setProject(imported);
      setPreflight(null);
      setTool("pan");
      setMessage("Đã nhập ảnh. Chọn Xóa nền để tạo alpha ARTWORK.");
    }
  };

  const processArtwork = async () => {
    if (!project) return;
    const processed = await run("Đang phân tích nền và bảo toàn component", () =>
      coordinatorCall<ProjectPayload>("process_artwork", {
        project_id: project.project_id,
        tolerance,
        softness,
        quality_preset: quality,
      }),
    );
    if (processed) {
      setProject(processed);
      setPreflight(null);
      setTool("remove");
    }
  };

  const applyBrush = async (points: Array<{ x: number; y: number }>) => {
    if (!project) return;
    const edited = await run(tool === "keep" ? "Đang giữ vùng cọ" : "Đang xóa vùng cọ", () =>
      coordinatorCall<ProjectPayload>("apply_brush", {
        project_id: project.project_id,
        points,
        radius,
        hardness: hardness / 100,
        opacity: 1,
        mode: tool === "keep" ? "keep" : "remove",
      }),
    );
    if (edited) {
      setProject(edited);
      setPreflight(null);
    }
  };

  const applyWand = async (point: { x: number; y: number }) => {
    if (!project) return;
    const edited = await run("Đang áp dụng Magic Wand", () =>
      coordinatorCall<ProjectPayload>("apply_magic_wand", {
        project_id: project.project_id,
        x: Math.floor(point.x),
        y: Math.floor(point.y),
        tolerance,
        softness,
        contiguous,
        mode: tool === "wand-keep" ? "keep" : "remove",
      }),
    );
    if (edited) {
      setProject(edited);
      setPreflight(null);
    }
  };

  const undo = async () => {
    if (!project || !project.history.can_undo || busy) return;
    const result = await run("Undo", () => coordinatorCall<ProjectPayload>("undo", { project_id: project.project_id }));
    if (result) setProject(result);
  };

  const redo = async () => {
    if (!project || !project.history.can_redo || busy) return;
    const result = await run("Redo", () => coordinatorCall<ProjectPayload>("redo", { project_id: project.project_id }));
    if (result) setProject(result);
  };

  const runPreflight = async () => {
    if (!project) return;
    const result = await run("Đang kiểm tra file in", () =>
      coordinatorCall<PreflightReport>("preflight", {
        project_id: project.project_id,
        output_mode: "POD_READY",
        print_width: printWidth,
        print_height: printHeight,
        print_unit: printUnit,
      }),
    );
    if (result) {
      setPreflight(result);
      setPanel("preflight");
    }
  };

  const exportOutput = async (mode: OutputMode) => {
    if (!project) return;
    const suffix = mode === "MASTER_SOURCE_FAITHFUL" ? "master" : mode === "POD_READY" ? "pod-ready" : "alpha-16bit";
    const destination = await save({
      defaultPath: `${stem(project.source_path)}-${suffix}.png`,
      filters: [{ name: "PNG", extensions: ["png"] }],
    });
    if (!destination) return;
    const result = await run(`Đang xuất ${mode}`, () => coordinatorCall<ExportResult>("export", {
      project_id: project.project_id,
      output_mode: mode,
      destination,
      settings: {
        trim: mode === "POD_READY" && trim,
        padding: mode === "POD_READY" ? padding : 0,
        target_ppi: 300,
      },
    }));
    if (result) setMessage(`Đã xuất ${result.width}×${result.height}: ${result.path}`);
  };

  const effectivePpi = useMemo(() => {
    if (!project || printWidth <= 0 || printHeight <= 0) return null;
    const unitToInch = printUnit === "cm" ? 1 / 2.54 : 1;
    return Math.floor(Math.min(
      project.width / (printWidth * unitToInch),
      project.height / (printHeight * unitToInch),
    ));
  }, [printHeight, printUnit, printWidth, project]);

  const changePrintUnit = (unit: "inch" | "cm") => {
    if (unit === printUnit) return;
    const factor = unit === "cm" ? 2.54 : 1 / 2.54;
    setPrintWidth(Number((printWidth * factor).toFixed(2)));
    setPrintHeight(Number((printHeight * factor).toFixed(2)));
    setPrintUnit(unit);
    setPreflight(null);
  };

  const installedModels = models.filter((model) => model.installed).length;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">C</span>
          <div><strong>Cutout Local</strong><small>POD Artwork Editor · v0.1</small></div>
        </div>
        <div className="top-actions">
          <button className="button secondary" onClick={importImage} disabled={!!busy}>Mở ảnh</button>
          <button className="icon-button" onClick={undo} disabled={!project?.history.can_undo || !!busy} title="Undo">↶</button>
          <button className="icon-button" onClick={redo} disabled={!project?.history.can_redo || !!busy} title="Redo">↷</button>
          <button className="button primary" onClick={() => setPanel("export")} disabled={!project}>Xuất file</button>
        </div>
        <div className="local-status"><i /> Offline · local only</div>
      </header>

      <section className="workspace">
        <aside className="tool-rail" aria-label="Công cụ">
          {tools.map((item) => (
            <button
              key={item.id}
              className={tool === item.id ? "active" : ""}
              onClick={() => setTool(item.id)}
              title={`${item.label} (${item.key})`}
              disabled={!project || !!busy}
            >
              <span>{item.id === "pan" ? "✥" : item.id === "keep" ? "+" : item.id === "remove" ? "−" : item.id === "wand-keep" ? "W+" : "W−"}</span>
              <small>{item.label}</small>
            </button>
          ))}
        </aside>

        <section className="stage">
          {project ? (
            <EditorCanvas
              project={project}
              tool={tool}
              radius={radius}
              background={background}
              disabled={!!busy}
              onBrush={applyBrush}
              onWand={applyWand}
            />
          ) : (
            <button className="empty-state" onClick={importImage}>
              <span className="empty-art">✦</span>
              <strong>Thả artwork vào đây</strong>
              <span>PNG · JPEG · static WebP, tối đa bảo đảm 40 MP</span>
              <em>Chọn ảnh</em>
            </button>
          )}
          {busy && <div className="busy-overlay"><span className="spinner" />{busy}</div>}
          <div className="background-picker" aria-label="Nền preview">
            {(["checker", "white", "black", "garment"] as const).map((item) => (
              <button key={item} className={`${item} ${background === item ? "active" : ""}`} onClick={() => setBackground(item)} title={`Nền ${item}`} />
            ))}
          </div>
        </section>

        <aside className="inspector">
          <nav className="panel-tabs">
            <button className={panel === "controls" ? "active" : ""} onClick={() => setPanel("controls")}>Xử lý</button>
            <button className={panel === "preflight" ? "active" : ""} onClick={() => setPanel("preflight")}>Preflight</button>
            <button className={panel === "export" ? "active" : ""} onClick={() => setPanel("export")}>Xuất</button>
          </nav>

          {panel === "controls" && (
            <div className="panel-content">
              <section className="control-section">
                <div className="section-heading"><div><strong>ARTWORK</strong><small>Mặc định cho POD / ảnh AI</small></div><span className="pill">LOCAL</span></div>
                <label>Profile
                  <select value={quality} onChange={(event) => setQuality(event.target.value as "FAST" | "QUALITY")}>
                    <option value="QUALITY">QUALITY — tối ưu cạnh</option>
                    <option value="FAST">FAST — preview nhanh</option>
                  </select>
                </label>
                <label className="range-row"><span>Color tolerance <b>{tolerance}</b></span><input type="range" min="1" max="100" value={tolerance} onChange={(event) => setTolerance(+event.target.value)} /></label>
                <label className="range-row"><span>Softness <b>{softness}</b></span><input type="range" min="0" max="60" value={softness} onChange={(event) => setSoftness(+event.target.value)} /></label>
                <button className="button primary full" onClick={processArtwork} disabled={!project || !!busy}>Xóa nền</button>
                <p className="hint">Flood-fill bắt đầu từ biên; component rời, lỗ và distressed texture không bị tự xóa.</p>
              </section>

              <section className="control-section">
                <h3>Chỉnh alpha</h3>
                <label className="range-row"><span>Kích thước cọ <b>{radius}px</b></span><input type="range" min="1" max="240" value={radius} onChange={(event) => setRadius(+event.target.value)} /></label>
                <label className="range-row"><span>Hardness <b>{hardness}%</b></span><input type="range" min="0" max="100" value={hardness} onChange={(event) => setHardness(+event.target.value)} /></label>
                <label className="check-row"><input type="checkbox" checked={contiguous} onChange={(event) => setContiguous(event.target.checked)} /><span>Magic Wand contiguous</span></label>
              </section>

              <section className="control-section compact">
                <h3>Component inspector</h3>
                {project ? <div className="stats-grid"><span>Tổng<b>{project.components?.count ?? 0}</b></span><span>Needs Review<b>{project.components?.suspicious_count ?? 0}</b></span><span>Undo<b>{project.history.length}</b></span></div> : <p className="muted">Chưa có project.</p>}
              </section>

              <section className="control-section model-card">
                <div className="section-heading"><h3>Model qualification</h3><span>{installedModels}/{models.length}</span></div>
                <p>{installedModels ? "Model đã pin sẵn sàng." : "V0.1 dùng Artwork Color/Edge engine. Các model AI chưa được bundle nếu chưa qua license/checksum/benchmark."}</p>
              </section>
            </div>
          )}

          {panel === "preflight" && (
            <div className="panel-content">
              <section className="control-section">
                <h3>Kích thước in thực</h3>
                <label>Đơn vị<select value={printUnit} onChange={(event) => changePrintUnit(event.target.value as "inch" | "cm")}><option value="inch">Inch</option><option value="cm">Centimet (cm)</option></select></label>
                <div className="two-columns"><label>Rộng ({printUnit})<input type="number" min="0.1" step={printUnit === "cm" ? "0.1" : "0.25"} value={printWidth} onChange={(event) => setPrintWidth(+event.target.value)} /></label><label>Cao ({printUnit})<input type="number" min="0.1" step={printUnit === "cm" ? "0.1" : "0.25"} value={printHeight} onChange={(event) => setPrintHeight(+event.target.value)} /></label></div>
                <div className={`ppi-card ${(effectivePpi ?? 0) < 150 ? "warn" : "good"}`}><span>Effective PPI</span><strong>{effectivePpi ?? "—"}</strong><small>Metadata 300 DPI không tăng chất lượng pixel.</small></div>
                <button className="button primary full" onClick={runPreflight} disabled={!project || !!busy}>Chạy preflight</button>
              </section>
              {preflight ? (
                <section className="report">
                  <div className={`report-status ${preflight.status.toLowerCase()}`}><strong>{preflight.status}</strong><span>{preflight.failures.length} lỗi · {preflight.warnings.length} cảnh báo</span></div>
                  {[...preflight.failures, ...preflight.warnings].map((item) => <article key={item.code}><b>{item.code}</b><p>{item.message}</p></article>)}
                  {!preflight.failures.length && !preflight.warnings.length && <p className="success-note">Không phát hiện vấn đề tự động.</p>}
                </section>
              ) : <p className="empty-panel">Preflight chỉ cảnh báo; không tự xóa stray pixel hay component nhỏ.</p>}
            </div>
          )}

          {panel === "export" && (
            <div className="panel-content">
              <section className="control-section export-list">
                <button onClick={() => exportOutput("MASTER_SOURCE_FAITHFUL")} disabled={!project || !!busy}><span><strong>Master source-faithful</strong><small>RGB canonical delta 0 · alpha mới</small></span><em>PNG 8-bit</em></button>
                <button onClick={() => exportOutput("POD_READY")} disabled={!project || !!busy}><span><strong>POD-ready</strong><small>sRGB · straight alpha · decontaminate viền</small></span><em>PNG 8-bit</em></button>
                <button onClick={() => exportOutput("ALPHA_ONLY")} disabled={!project || !!busy}><span><strong>Alpha only</strong><small>Matte cho QA và trao đổi</small></span><em>PNG 16-bit</em></button>
              </section>
              <section className="control-section">
                <h3>Tùy chọn POD-ready</h3>
                <label className="check-row"><input type="checkbox" checked={trim} onChange={(event) => setTrim(event.target.checked)} /><span>Trim vùng trong suốt</span></label>
                <label>Padding (px)<input type="number" min="0" max="2000" value={padding} onChange={(event) => setPadding(+event.target.value)} disabled={!trim} /></label>
                <p className="hint">Mặc định giữ nguyên kích thước pixel. App không tự scale hoặc giả 300 DPI.</p>
              </section>
              <section className="contract-note"><strong>Hai output độc lập</strong><p>Master ưu tiên bảo toàn RGB. POD-ready được phép xử lý RGB quanh viền bán trong suốt để composite sạch hơn.</p></section>
            </div>
          )}
        </aside>
      </section>

      <footer className="statusbar">
        <span className={error ? "status-error" : ""}>{error ? `Lỗi: ${error}` : message}</span>
        <span>{project ? `${basename(project.source_path)} · ${project.width.toLocaleString()}×${project.height.toLocaleString()} px · schema ${project.schema_version}` : health ? `${health.processing_engine} · Python ${health.python}` : "Coordinator chưa kết nối"}</span>
      </footer>
    </main>
  );
}
