import { spawn } from "node:child_process";
import { copyFile, mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createInterface } from "node:readline";

const [executable, imagePath, modelsDirectory, outputDirectory] = process.argv.slice(2);
if (!executable || !imagePath || !modelsDirectory) {
  throw new Error(
    "Usage: node scripts/smoke-watermark-sidecar.mjs <sidecar> <image> <models-dir> [output-dir]",
  );
}

// Dùng project tạm để smoke test không thay đổi project thật của người dùng.
const smokeRoot = await mkdtemp(join(tmpdir(), "cutout-watermark-smoke-"));
const child = spawn(resolve(executable), [], {
  stdio: ["pipe", "pipe", "inherit"],
  env: {
    ...process.env,
    CUTOUT_PROJECTS_DIR: join(smokeRoot, "projects"),
    CUTOUT_MODELS_DIR: resolve(modelsDirectory),
  },
});
const lines = createInterface({ input: child.stdout });
const pending = new Map();
lines.on("line", (line) => {
  const response = JSON.parse(line);
  const waiter = pending.get(response.id);
  if (!waiter) return;
  pending.delete(response.id);
  response.ok
    ? waiter.resolve(response.result)
    : waiter.reject(new Error(response.error?.message ?? "Sidecar trả lỗi không xác định"));
});

let sequence = 0;
function request(method, params = {}) {
  sequence += 1;
  const id = `watermark-${sequence}`;
  return new Promise((resolvePromise, reject) => {
    pending.set(id, { resolve: resolvePromise, reject });
    child.stdin.write(`${JSON.stringify({ id, method, params })}\n`);
  });
}

function gate(session) {
  return session.preview_diagnostics?.quality_gate ?? {};
}

try {
  const importedAuto = await request("import_image", { path: resolve(imagePath) });
  const autoSession = await request("begin_watermark_session", {
    project_id: importedAuto.project_id,
    quality: "BALANCED",
    feather: 8,
    expand: "MEDIUM",
  });
  const detected = await request("auto_detect_watermark", {
    project_id: importedAuto.project_id,
    session_id: autoSession.session_id,
    feather: 8,
    expand: "MEDIUM",
  });
  const autoPreview = await request("preview_watermark", {
    project_id: importedAuto.project_id,
    session_id: autoSession.session_id,
    quality: "BALANCED",
  });
  if (detected.diagnostics?.detector !== "GEMINI_SPARKLE_NCC") {
    throw new Error(`Auto detector sai: ${detected.diagnostics?.detector ?? "missing"}`);
  }
  if (gate(autoPreview).status !== "PASS") {
    throw new Error("Auto preview không vượt quality gate");
  }

  const [x0, y0, x1, y1] = detected.bounds;
  const importedManual = await request("import_image", { path: resolve(imagePath) });
  const manualSession = await request("begin_watermark_session", {
    project_id: importedManual.project_id,
    quality: "BALANCED",
    feather: 8,
    expand: "MEDIUM",
  });
  const centerX = (x0 + x1) / 2;
  const centerY = (y0 + y1) / 2;
  const radius = Math.max(x1 - x0, y1 - y0) * 0.58;
  await request("update_watermark_mask", {
    project_id: importedManual.project_id,
    session_id: manualSession.session_id,
    mode: "ADD",
    points: [{ x: centerX, y: centerY }],
    radius,
    hardness: 0.8,
    feather: 8,
  });
  const manualPreview = await request("preview_watermark", {
    project_id: importedManual.project_id,
    session_id: manualSession.session_id,
    quality: "BALANCED",
  });
  if (gate(manualPreview).status !== "PASS") {
    throw new Error("Manual preview không vượt quality gate");
  }

  if (outputDirectory) {
    const destination = resolve(outputDirectory);
    await mkdir(destination, { recursive: true });
    await copyFile(autoPreview.preview_path, join(destination, "packaged-auto-preview.png"));
    await copyFile(manualPreview.preview_path, join(destination, "packaged-manual-preview.png"));
  }

  process.stdout.write(`${JSON.stringify({
    detector: detected.diagnostics.detector,
    detected_bounds: detected.bounds,
    auto_mask_pixels: detected.mask_pixel_count,
    auto_quality_gate: gate(autoPreview),
    manual_mask_pixels: manualPreview.mask_pixel_count,
    manual_quality_gate: gate(manualPreview),
  }, null, 2)}\n`);
} finally {
  child.stdin.end();
  await new Promise((resolvePromise) => child.once("exit", resolvePromise));
  await rm(smokeRoot, { recursive: true, force: true });
}
