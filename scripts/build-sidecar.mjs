import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join } from "node:path";

const candidates = [
  process.env.CUTOUT_BUILD_PYTHON,
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
  throw new Error("Không tìm thấy Python để build sidecar.");
}

const check = spawnSync(
  python,
  ["-c", "import PIL, numpy, PyInstaller"],
  { stdio: "inherit", shell: false },
);

if (check.status !== 0) {
  throw new Error(`Thiếu Pillow, NumPy hoặc PyInstaller. Cài bằng: ${python} -m pip install -r sidecar/requirements-build.txt`);
}

const env = {
  ...process.env,
  // Tách cache PyInstaller để build lại sidecar ổn định trên Windows.
  PYINSTALLER_CONFIG_DIR: join("sidecar", "build", "pyinstaller-config"),
};

const args = [
  "-m",
  "PyInstaller",
  "--noconfirm",
  "--clean",
  "--onefile",
  // Sidecar chỉ giao tiếp qua pipe JSON, không cần console Windows hiện ra cho người dùng.
  "--noconsole",
  "--name",
  "cutout-sidecar",
  "--distpath",
  join("sidecar", "dist"),
  "--workpath",
  join("sidecar", "build"),
  "--specpath",
  join("sidecar", "build"),
  join("sidecar", "main.py"),
];

const result = spawnSync(python, args, { stdio: "inherit", env, shell: false });
if (result.status !== 0) {
  process.exit(result.status ?? 1);
}

console.log("Sidecar standalone đã tạo trong sidecar/dist/.");
