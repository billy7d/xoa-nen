import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { resolve } from "node:path";

const [executable, imagePath] = process.argv.slice(2);
if (!executable || !imagePath) {
  throw new Error("Usage: node scripts/smoke-bundled-sidecar.mjs <sidecar> <image>");
}

const child = spawn(resolve(executable), [], {
  stdio: ["pipe", "pipe", "inherit"],
  env: { ...process.env, CUTOUT_PROJECTS_DIR: "/tmp/cutout-v3-bundled-smoke/projects" },
});
const lines = createInterface({ input: child.stdout });
const pending = new Map();
lines.on("line", (line) => {
  const response = JSON.parse(line);
  const waiter = pending.get(response.id);
  if (waiter) {
    pending.delete(response.id);
    response.ok ? waiter.resolve(response.result) : waiter.reject(new Error(response.error?.message));
  }
});

function request(id, method, params = {}) {
  return new Promise((resolvePromise, reject) => {
    pending.set(id, { resolve: resolvePromise, reject });
    child.stdin.write(`${JSON.stringify({ id, method, params })}\n`);
  });
}

try {
  const health = await request("health", "health");
  const imported = await request("import", "import_image", { path: resolve(imagePath) });
  const processed = await request("process", "process_artwork", {
    project_id: imported.project_id,
    engine_profile: "V3_BALANCED",
    quality_preset: "FAST",
    tolerance: 30,
    softness: 18,
  });
  if (health.processing_engine !== "hybrid-cutout-v3") throw new Error("Sai processing engine");
  if (processed.process?.engine_profile !== "V3_BALANCED") throw new Error("Sai engine profile");
  process.stdout.write(JSON.stringify({
    health: health.status,
    version: health.version,
    engine: health.processing_engine,
    processed: [processed.width, processed.height],
    strategy: processed.process.diagnostics.selected_strategy,
  }, null, 2) + "\n");
} finally {
  child.stdin.end();
  await new Promise((resolvePromise) => child.once("exit", resolvePromise));
}
