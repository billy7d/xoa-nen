import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join } from "node:path";

const candidates = [
  process.env.CUTOUT_BUILD_PYTHON,
  process.env.CUTOUT_PYTHON,
  join(".venv", "Scripts", "python.exe"),
  join(".venv", "bin", "python"),
  "python3",
  "python",
].filter(Boolean);

const python = candidates.find((candidate) => {
  if (!candidate) return false;
  return candidate.includes("\\") || candidate.includes("/") ? existsSync(candidate) : true;
});

if (!python) {
  throw new Error("Không tìm thấy Python để provision model local.");
}

const result = spawnSync(
  python,
  ["scripts/provision-models.py", ...process.argv.slice(2)],
  { stdio: "inherit", shell: false },
);

if (result.error) {
  throw result.error;
}
if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
