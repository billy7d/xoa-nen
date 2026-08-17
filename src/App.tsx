import { useEffect, useMemo, useState } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { coordinatorCall, isTauriRuntime } from "./bridge";
import EditorCanvas from "./components/EditorCanvas";
import type {
  EngineProfile,
  ExportResult,
  ForegroundPoint,
  HealthPayload,
  ModelManifest,
  OutputMode,
  PreflightReport,
  ProjectPayload,
  QualityPreset,
  ToolMode,
  WandAlgorithm,
  WandPreview,
} from "./types";

const tools: Array<{ id: ToolMode; label: string; key: string; icon: string }> = [
  { id: "pan", label: "Di chuyển", key: "H", icon: "✥" },
  { id: "subject", label: "Vật thể", key: "O", icon: "▣" },
  { id: "protect", label: "Khóa vật thể", key: "P", icon: "◉" },
  { id: "keep", label: "Giữ", key: "K", icon: "+" },
  { id: "remove", label: "Xóa", key: "E", icon: "−" },
  { id: "wand-keep", label: "Wand giữ", key: "W", icon: "W+" },
  { id: "wand-remove", label: "Wand xóa", key: "S", icon: "W−" },
  { id: "watermark", label: "Watermark", key: "M", icon: "✦" },
];

const basename = (path: string) => path.split(/[\\/]/).pop() || "artwork";
const stem = (path: string) => basename(path).replace(/\.[^.]+$/, "");
const friendlyError = (error: unknown) => error instanceof Error ? error.message : String(error);

export default function App() {
  const [project, setProject] = useState<ProjectPayload | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [models, setModels] = useState<ModelManifest[]>([]);
  const [tool, setTool] = useState<ToolMode>("pan");
  const [foregroundPoints, setForegroundPoints] = useState<ForegroundPoint[]>([]);
  const [engineProfile, setEngineProfile] = useState<EngineProfile>("V3_BALANCED");
  const [quality, setQuality] = useState<QualityPreset>("QUALITY");
  const [legacyTolerance, setLegacyTolerance] = useState(30);
  const [legacySoftness, setLegacySoftness] = useState(18);
  const [autoTolerance, setAutoTolerance] = useState(30);
  const [autoSoftness, setAutoSoftness] = useState(18);
  const [wandTolerance, setWandTolerance] = useState(30);
  const [wandSoftness, setWandSoftness] = useState(18);
  const [wandAlgorithm, setWandAlgorithm] = useState<WandAlgorithm>("SMART");
  const [wandPreview, setWandPreview] = useState<WandPreview | null>(null);
  const [radius, setRadius] = useState(20);
  const [hardness, setHardness] = useState(82);
  const [contiguous, setContiguous] = useState(true);
  const [background, setBackground] = useState<"checker" | "white" | "black" | "garment">("checker");
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState("Sẵn sàng — mọi xử lý diễn ra trên máy này.");
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

  const refreshModels = async () => {
    const result = await coordinatorCall<{ models: ModelManifest[] }>("list_models");
    setModels(result.models);
  };

  useEffect(() => {
    if (!isTauriRuntime()) {
      setError("Hãy chạy `npm run tauri dev` để dùng coordinator local.");
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
    setForegroundPoints(project?.process?.foreground_points ?? []);
  }, [project?.project_id, project?.revision]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // Khi Wand đang preview, Enter/Esc được ưu tiên để xác nhận hoặc hủy nhanh.
      if (wandPreview && !busy && !event.repeat && !event.isComposing) {
        if (event.key === "Enter") {
          event.preventDefault();
          event.stopPropagation();
          void commitWand();
          return;
        }
        if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          void cancelWand();
          return;
        }
      }
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      const mapping: Record<string, ToolMode> = {
        h: "pan", o: "subject", p: "protect", k: "keep", e: "remove", w: "wand-keep", s: "wand-remove", m: "watermark",
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
      filters: [{ name: "Ảnh", extensions: ["png", "jpg", "jpeg", "webp"] }],
    });
    if (!selected || Array.isArray(selected)) return;
    const imported = await run("Đang nhập ảnh", () => coordinatorCall<ProjectPayload>("import_image", { path: selected }));
    if (imported) {
      setProject(imported);
      setWandPreview(null);
      setPreflight(null);
      setTool("pan");
      setForegroundPoints([]);
    }
  };

  const processArtwork = async () => {
    if (!project) return;
    const legacy = engineProfile === "LEGACY_V1";
    const processed = await run("Đang chạy V3 Hybrid Cutout", () => coordinatorCall<ProjectPayload>("process_artwork", {
      project_id: project.project_id,
      tolerance: legacy ? legacyTolerance : autoTolerance,
      softness: legacy ? legacySoftness : autoSoftness,
      quality_preset: quality,
      engine_profile: engineProfile,
      subject_policy: "ALL_DETECTED",
      foreground_points: foregroundPoints,
      background_points: [],
      protection_mode: "CONSERVATIVE",
      shadow_policy: "REMOVE",
    }));
    if (processed) {
      setProject(processed);
      setWandPreview(null);
      setPreflight(null);
      setTool(processed.process?.result_status === "NEEDS_PROTECTION"
        ? "protect"
        : processed.process?.review_regions.length ? "subject" : "remove");
    }
  };

  const lockForegroundPoint = async (point: ForegroundPoint, append: boolean) => {
    setForegroundPoints((current) => append ? [...current, point].slice(-16) : [point]);
    setError(null);
    setMessage(append
      ? "Đã thêm điểm khóa. Chạy Xóa nền để áp dụng bảo vệ."
      : "Đã khóa vật thể. Shift+bấm để thêm quai, ống hút hoặc phần rời.");
  };

  const applyBrush = async (points: Array<{ x: number; y: number }>) => {
    if (!project) return;
    const edited = await run(tool === "keep" ? "Đang giữ vùng cọ" : "Đang xóa vùng cọ", () => coordinatorCall<ProjectPayload>("apply_brush", {
      project_id: project.project_id,
      points,
      radius,
      hardness: hardness / 100,
      opacity: 1,
      mode: tool === "keep" ? "keep" : "remove",
    }));
    if (edited) { setProject(edited); setPreflight(null); }
  };

  const removeWatermark = async (mode: "AUTO" | "MANUAL", points: Array<{ x: number; y: number }> = []) => {
    if (!project) return;
    const edited = await run(mode === "AUTO" ? "Đang tìm và xóa watermark" : "Đang lấp vùng watermark", () => coordinatorCall<ProjectPayload>("remove_watermark", {
      project_id: project.project_id, mode, points, radius,
    }));
    if (edited) { setProject(edited); setPreflight(null); }
  };

  const previewWand = async (point: { x: number; y: number }) => {
    if (!project) return;
    if (wandPreview) await cancelWand(false);
    const preview = await run("Đang tạo preview Wand", () => coordinatorCall<WandPreview>("preview_magic_wand", {
      project_id: project.project_id,
      x: Math.floor(point.x),
      y: Math.floor(point.y),
      tolerance: wandTolerance,
      softness: wandSoftness,
      contiguous,
      mode: tool === "wand-keep" ? "keep" : "remove",
      wand_algorithm: wandAlgorithm,
    }));
    if (preview) setWandPreview(preview);
  };

  const commitWand = async () => {
    if (!project || !wandPreview) return;
    const edited = await run("Đang áp dụng Wand", () => coordinatorCall<ProjectPayload>("commit_magic_wand", {
      project_id: project.project_id,
      selection_id: wandPreview.selection_id,
      mode: wandPreview.mode,
      wand_algorithm: wandPreview.wand_algorithm,
    }));
    if (edited) { setProject(edited); setWandPreview(null); setPreflight(null); }
  };

  const cancelWand = async (showBusy = true) => {
    if (!project || !wandPreview) return;
    const operation = () => coordinatorCall<{ cancelled: boolean }>("cancel_magic_wand", {
      project_id: project.project_id,
      selection_id: wandPreview.selection_id,
    });
    if (showBusy) await run("Đang hủy preview Wand", operation); else await operation().catch(() => undefined);
    setWandPreview(null);
  };

  const updateSubjectSelection = async (selected: number[]) => {
    if (!project?.process) return;
    const updated = await run("Đang cập nhật vật thể", () => coordinatorCall<ProjectPayload>("set_subject_selection", {
      project_id: project.project_id,
      selected_subject_ids: selected,
    }));
    if (updated) setProject(updated);
  };

  const selectSubjectAt = async (point: { x: number; y: number }) => {
    if (!project?.process) return;
    const matches = project.process.subjects.filter((subject) => {
      const [x0, y0, x1, y1] = subject.bbox;
      return point.x >= x0 && point.x <= x1 && point.y >= y0 && point.y <= y1;
    }).sort((a, b) => a.area_px - b.area_px);
    if (!matches.length) return;
    const current = new Set(project.process.selected_subject_ids);
    if (current.has(matches[0].id)) current.delete(matches[0].id); else current.add(matches[0].id);
    await updateSubjectSelection([...current]);
  };

  const undo = async () => {
    if (!project || !project.history.can_undo || busy) return;
    const result = await run("Undo", () => coordinatorCall<ProjectPayload>("undo", { project_id: project.project_id }));
    if (result) { setProject(result); setWandPreview(null); }
  };
  const redo = async () => {
    if (!project || !project.history.can_redo || busy) return;
    const result = await run("Redo", () => coordinatorCall<ProjectPayload>("redo", { project_id: project.project_id }));
    if (result) { setProject(result); setWandPreview(null); }
  };

  const installModelPack = async () => {
    const selected = await open({ multiple: false, directory: false, filters: [{ name: "Cutout model-pack", extensions: ["cutout-modelpack"] }] });
    if (!selected || Array.isArray(selected)) return;
    const result = await run("Đang xác minh và cài model-pack", () => coordinatorCall<{ models: ModelManifest[] }>("install_model_pack", { path: selected }));
    if (result) setModels(result.models);
  };

  const removeModel = async (modelId: string) => {
    const result = await run("Đang gỡ model-pack", () => coordinatorCall<{ models: ModelManifest[] }>("remove_model_pack", { model_id: modelId }));
    if (result) setModels(result.models);
  };

  const downloadModel = async (modelId: string) => {
    const result = await run("Đang tải và xác minh model-pack", () => coordinatorCall<{ models: ModelManifest[] }>("download_model_pack", { model_id: modelId }));
    if (result) setModels(result.models);
  };

  const runPreflight = async () => {
    if (!project) return;
    const result = await run("Đang kiểm tra file in", () => coordinatorCall<PreflightReport>("preflight", {
      project_id: project.project_id, output_mode: "POD_READY", print_width: printWidth,
      print_height: printHeight, print_unit: printUnit,
    }));
    if (result) { setPreflight(result); setPanel("preflight"); }
  };

  const exportOutput = async (mode: OutputMode) => {
    if (!project) return;
    const suffix = mode === "MASTER_SOURCE_FAITHFUL" ? "master" : mode === "POD_READY" ? "pod-ready" : "alpha-16bit";
    const destination = await save({ defaultPath: `${stem(project.source_path)}-${suffix}.png`, filters: [{ name: "PNG", extensions: ["png"] }] });
    if (!destination) return;
    const result = await run(`Đang xuất ${mode}`, () => coordinatorCall<ExportResult>("export", {
      project_id: project.project_id, output_mode: mode, destination,
      settings: { trim: mode === "POD_READY" && trim, padding: mode === "POD_READY" ? padding : 0, target_ppi: 300 },
    }));
    if (result) setMessage(`Đã xuất ${result.width}×${result.height}: ${result.path}`);
  };

  const effectivePpi = useMemo(() => {
    if (!project || printWidth <= 0 || printHeight <= 0) return null;
    const unitToInch = printUnit === "cm" ? 1 / 2.54 : 1;
    return Math.floor(Math.min(project.width / (printWidth * unitToInch), project.height / (printHeight * unitToInch)));
  }, [printHeight, printUnit, printWidth, project]);

  const changePrintUnit = (unit: "inch" | "cm") => {
    if (unit === printUnit) return;
    const factor = unit === "cm" ? 2.54 : 1 / 2.54;
    setPrintWidth(Number((printWidth * factor).toFixed(2)));
    setPrintHeight(Number((printHeight * factor).toFixed(2)));
    setPrintUnit(unit);
    setPreflight(null);
  };

  const installedModels = models.filter((model) => model.installed);
  const aiReady = installedModels.some((model) => model.role === "base_alpha_proposal");
  const tolerance = engineProfile === "LEGACY_V1" ? legacyTolerance : autoTolerance;
  const softness = engineProfile === "LEGACY_V1" ? legacySoftness : autoSoftness;
  const canvasProject = project && wandPreview
    ? { ...project, preview_path: wandPreview.preview_path, revision: wandPreview.selection_id }
    : project;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">C</span><div><strong>Cutout Local</strong><small>V3 Hybrid Cutout · v0.3</small></div></div>
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
          {tools.map((item) => <button key={item.id} className={tool === item.id ? "active" : ""} onClick={() => setTool(item.id)} title={`${item.label} (${item.key})`} disabled={!project || !!busy}><span>{item.icon}</span><small>{item.label}</small></button>)}
        </aside>

        <section className="stage">
          {canvasProject ? <EditorCanvas project={canvasProject} tool={tool} radius={radius} background={background} disabled={!!busy} foregroundPoints={foregroundPoints} onBrush={applyBrush} onWand={previewWand} onSubject={selectSubjectAt} onProtect={lockForegroundPoint} onWatermark={(points) => removeWatermark("MANUAL", points)} /> : (
            <button className="empty-state" onClick={importImage}><span className="empty-art">✦</span><strong>Thả artwork vào đây</strong><span>PNG · JPEG · static WebP, tối đa bảo đảm 40 MP</span><em>Chọn ảnh</em></button>
          )}
          {busy && <div className="busy-overlay"><span className="spinner" />{busy}</div>}
          {wandPreview && <div className="wand-preview-bar"><span>{wandPreview.selected_pixel_count.toLocaleString()} px được chọn</span><button onClick={commitWand} title="Enter">Áp dụng ↵</button><button onClick={() => cancelWand()} title="Escape">Hủy Esc</button></div>}
          <div className="background-picker" aria-label="Nền preview">{(["checker", "white", "black", "garment"] as const).map((item) => <button key={item} className={`${item} ${background === item ? "active" : ""}`} onClick={() => setBackground(item)} title={`Nền ${item}`} />)}</div>
        </section>

        <aside className="inspector">
          <nav className="panel-tabs"><button className={panel === "controls" ? "active" : ""} onClick={() => setPanel("controls")}>Xử lý</button><button className={panel === "preflight" ? "active" : ""} onClick={() => setPanel("preflight")}>Preflight</button><button className={panel === "export" ? "active" : ""} onClick={() => setPanel("export")}>Xuất</button></nav>

          {panel === "controls" && <div className="panel-content">
            <section className="control-section">
              <div className="section-heading"><div><strong>XÓA WATERMARK</strong><small>Inpainting native — giữ nguyên kích thước pixel</small></div><span className="pill">LOCAL</span></div>
              <button className="button primary full" onClick={() => removeWatermark("AUTO")} disabled={!project || !!busy}>Tự động tìm & xóa</button>
              <div className="subject-actions"><button type="button" className={tool === "watermark" ? "active" : ""} onClick={() => setTool("watermark")}>Xóa bằng cọ (M)</button></div>
              <p className="hint">Tự động chỉ chọn các nét chữ/logo nhỏ có tín hiệu watermark để tránh xóa nhầm artwork. Với watermark phức tạp, chọn Xóa bằng cọ, tô phủ toàn bộ watermark rồi thả chuột để lấp nền. Undo/Redo hoạt động cho cả hai chế độ.</p>
              {project?.retouch?.watermark_removed ? <p className="success-note">Watermark đã được xử lý trong ảnh xuất.</p> : null}
            </section>

            <section className="control-section">
              <div className="section-heading"><div><strong>HYBRID CUTOUT</strong><small>V3 mặc định · V1 luôn khả dụng</small></div><span className="pill">LOCAL</span></div>
              <label>Profile<select value={engineProfile} onChange={(event) => setEngineProfile(event.target.value as EngineProfile)}><option value="V3_BALANCED">V3 Cân bằng — mặc định</option><option value="V3_AI_LOCAL" disabled={!aiReady}>V3 AI Local {aiReady ? "" : "— cần model-pack"}</option><option value="LEGACY_V1">V1 nền phẳng — pixel-exact</option></select></label>
              <label>Tốc độ<select value={quality} onChange={(event) => setQuality(event.target.value as QualityPreset)}><option value="QUALITY">QUALITY — refine native</option><option value="FAST">FAST — proxy nhanh</option></select></label>
              <label className="range-row"><span>Tolerance {engineProfile === "LEGACY_V1" ? "V1" : "V3"} <b>{tolerance}</b></span><input type="range" min="1" max="100" value={tolerance} onChange={(event) => engineProfile === "LEGACY_V1" ? setLegacyTolerance(+event.target.value) : setAutoTolerance(+event.target.value)} /></label>
              <label className="range-row"><span>Softness <b>{softness}</b></span><input type="range" min="0" max="60" value={softness} onChange={(event) => engineProfile === "LEGACY_V1" ? setLegacySoftness(+event.target.value) : setAutoSoftness(+event.target.value)} /></label>
              <button className="button primary full" onClick={processArtwork} disabled={!project || !!busy}>Xóa nền</button>
              <div className="subject-actions">
                <button type="button" className={tool === "protect" ? "active" : ""} onClick={() => setTool("protect")}>Khóa vật thể (P)</button>
                <button type="button" onClick={() => setForegroundPoints([])} disabled={!foregroundPoints.length}>Xóa điểm khóa</button>
              </div>
              <p className="hint">Bấm Khóa vật thể rồi bấm vào thân. Giữ Shift và bấm thêm ống hút, quai hoặc phần rời; các pixel đã khóa không được phép bị xóa.</p>
              <p className="hint">V3 dùng field nền theo vị trí, graph-cut và refine vùng bất định. Khi thiếu tự tin, app ưu tiên giữ theo V1 và tô vàng Needs Review.</p>
            </section>

            {project?.warnings?.length ? <section className="control-section review-card"><h3>Needs Review</h3>{project.warnings.map((warning) => <p key={warning}>{warning}</p>)}</section> : null}

            {project?.process?.subjects.length ? <section className="control-section">
              <div className="section-heading"><h3>Vật thể phát hiện</h3><span>{project.process.selected_subject_ids.length}/{project.process.subjects.length}</span></div>
              <div className="subject-actions"><button onClick={() => updateSubjectSelection(project.process!.subjects.map((item) => item.id))}>Giữ tất cả</button><button onClick={() => updateSubjectSelection([])}>Bỏ tất cả</button></div>
              <p className="hint">Chọn công cụ Vật thể rồi bấm khung xanh/đỏ để bật hoặc tắt từng phần.</p>
            </section> : null}

            <section className="control-section">
              <h3>Chỉnh alpha</h3>
              <label className="range-row"><span>Kích thước cọ <b>{radius}px</b></span><input type="range" min="1" max="240" value={radius} onChange={(event) => setRadius(+event.target.value)} /></label>
              <label className="range-row"><span>Hardness <b>{hardness}%</b></span><input type="range" min="0" max="100" value={hardness} onChange={(event) => setHardness(+event.target.value)} /></label>
              <label>Wand<select value={wandAlgorithm} onChange={(event) => setWandAlgorithm(event.target.value as WandAlgorithm)}><option value="SMART">Smart — field + geodesic</option><option value="LEGACY_COLOR">Legacy Color — đúng V1</option></select></label>
              <label className="range-row"><span>Wand tolerance <b>{wandTolerance}</b></span><input type="range" min="1" max="100" value={wandTolerance} onChange={(event) => setWandTolerance(+event.target.value)} /></label>
              <label className="range-row"><span>Wand softness <b>{wandSoftness}</b></span><input type="range" min="0" max="60" value={wandSoftness} onChange={(event) => setWandSoftness(+event.target.value)} /></label>
              <label className="check-row"><input type="checkbox" checked={contiguous} onChange={(event) => setContiguous(event.target.checked)} /><span>Chỉ vùng liên thông</span></label>
            </section>

            <section className="control-section compact"><h3>Component inspector</h3>{project ? <div className="stats-grid"><span>Tổng<b>{project.components?.count ?? 0}</b></span><span>Review<b>{project.components?.suspicious_count ?? 0}</b></span><span>Undo<b>{project.history.length}</b></span></div> : <p className="muted">Chưa có project.</p>}</section>

            <section className="control-section model-card">
              <div className="section-heading"><h3>AI model-pack</h3><span>{installedModels.length} sẵn sàng</span></div>
              <p>Chỉ nhận ONNX pack có chữ ký, SHA-256, license thương mại và backend đã qualification. Sau cài đặt xử lý hoàn toàn offline.</p>
              <button className="button secondary full model-import" onClick={installModelPack}>Nhập .cutout-modelpack</button>
              {models.filter((model) => !model.installed && model.download_url).map((model) => <button className="button secondary full model-import" key={model.model_id} onClick={() => downloadModel(model.model_id)}>Tải {model.model_id}</button>)}
              {installedModels.map((model) => <div className="installed-model" key={model.model_id}><span>{model.model_id}<small>{model.revision}</small></span><button onClick={() => removeModel(model.model_id)}>Gỡ</button></div>)}
              <button className="text-button" onClick={refreshModels}>Kiểm tra lại model</button>
            </section>
          </div>}

          {panel === "preflight" && <div className="panel-content">
            <section className="control-section"><h3>Kích thước in thực</h3><label>Đơn vị<select value={printUnit} onChange={(event) => changePrintUnit(event.target.value as "inch" | "cm")}><option value="inch">Inch</option><option value="cm">Centimet (cm)</option></select></label><div className="two-columns"><label>Rộng ({printUnit})<input type="number" min="0.1" step={printUnit === "cm" ? "0.1" : "0.25"} value={printWidth} onChange={(event) => setPrintWidth(+event.target.value)} /></label><label>Cao ({printUnit})<input type="number" min="0.1" step={printUnit === "cm" ? "0.1" : "0.25"} value={printHeight} onChange={(event) => setPrintHeight(+event.target.value)} /></label></div><div className={`ppi-card ${(effectivePpi ?? 0) < 150 ? "warn" : "good"}`}><span>Effective PPI</span><strong>{effectivePpi ?? "—"}</strong><small>Metadata 300 DPI không tăng chất lượng pixel.</small></div><button className="button primary full" onClick={runPreflight} disabled={!project || !!busy}>Chạy preflight</button></section>
            {preflight ? <section className="report"><div className={`report-status ${preflight.status.toLowerCase()}`}><strong>{preflight.status}</strong><span>{preflight.failures.length} lỗi · {preflight.warnings.length} cảnh báo</span></div>{[...preflight.failures, ...preflight.warnings].map((item) => <article key={item.code}><b>{item.code}</b><p>{item.message}</p></article>)}{!preflight.failures.length && !preflight.warnings.length && <p className="success-note">Không phát hiện vấn đề tự động.</p>}</section> : <p className="empty-panel">Preflight chỉ cảnh báo; không tự xóa stray pixel.</p>}
          </div>}

          {panel === "export" && <div className="panel-content">
            <section className="control-section export-list"><button onClick={() => exportOutput("MASTER_SOURCE_FAITHFUL")} disabled={!project || !!busy}><span><strong>{project?.retouch?.watermark_removed ? "Master đã chỉnh watermark" : "Master source-faithful"}</strong><small>{project?.retouch?.watermark_removed ? "RGB đã lấp watermark · giữ nguyên pixel" : "RGB canonical delta 0 · alpha mới"}</small></span><em>PNG 8-bit</em></button><button onClick={() => exportOutput("POD_READY")} disabled={!project || !!busy}><span><strong>POD-ready</strong><small>sRGB · straight alpha · decontaminate cục bộ</small></span><em>PNG 8-bit</em></button><button onClick={() => exportOutput("ALPHA_ONLY")} disabled={!project || !!busy}><span><strong>Alpha only</strong><small>Matte cho QA và trao đổi</small></span><em>PNG 16-bit</em></button></section>
            <section className="control-section"><h3>Tùy chọn POD-ready</h3><label className="check-row"><input type="checkbox" checked={trim} onChange={(event) => setTrim(event.target.checked)} /><span>Trim vùng trong suốt</span></label><label>Padding (px)<input type="number" min="0" max="2000" value={padding} onChange={(event) => setPadding(+event.target.value)} disabled={!trim} /></label><p className="hint">Mặc định giữ nguyên kích thước pixel. App không tự scale hoặc giả 300 DPI.</p></section>
            <section className="contract-note"><strong>Hai output độc lập</strong><p>Master giữ nguyên RGB canonical. POD-ready chỉ decontaminate RGB ở pixel bán trong suốt bằng field nền cục bộ.</p></section>
          </div>}
        </aside>
      </section>

      <footer className="statusbar"><span className={error ? "status-error" : ""}>{error ? `Lỗi: ${error}` : message}</span><span>{project ? `${basename(project.source_path)} · ${project.width.toLocaleString()}×${project.height.toLocaleString()} px · schema ${project.schema_version}` : health ? `${health.processing_engine} · Python ${health.python}` : "Coordinator chưa kết nối"}</span></footer>
    </main>
  );
}
