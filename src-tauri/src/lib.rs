use serde_json::Value;
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, State};

struct CoordinatorProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl CoordinatorProcess {
    fn spawn(app: &AppHandle) -> Result<Self, String> {
        let dev_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("sidecar")
            .join("main.py");

        let resource_dir = app
            .path()
            .resource_dir()
            .map_err(|error| format!("Không xác định được resource directory: {error}"))?;
        let resource_script = resource_dir
            .join("sidecar")
            .join("main.py");

        let executable_name = if cfg!(target_os = "windows") {
            "cutout-sidecar.exe"
        } else {
            "cutout-sidecar"
        };
        let packaged_executable = resource_dir
            .join("sidecar")
            .join("dist")
            .join(executable_name);

        let script = if resource_script.exists() {
            resource_script
        } else {
            dev_script
        };

        if !packaged_executable.exists() && !script.exists() {
            return Err(format!("Không tìm thấy Python sidecar: {}", script.display()));
        }

        let app_data = app
            .path()
            .app_data_dir()
            .map_err(|error| format!("Không xác định được app data directory: {error}"))?;
        std::fs::create_dir_all(&app_data)
            .map_err(|error| format!("Không tạo được app data directory: {error}"))?;

        let mut command = if packaged_executable.exists() {
            Command::new(&packaged_executable)
        } else {
            let python = env::var("CUTOUT_PYTHON").unwrap_or_else(|_| "python3".to_string());
            let mut python_command = Command::new(&python);
            python_command.arg("-u").arg(&script);
            python_command
        };
        command
            .current_dir(&app_data)
            .env("CUTOUT_PROJECTS_DIR", app_data.join("projects"))
            .env("CUTOUT_MODELS_DIR", app_data.join("models"));

        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                format!(
                    "Không khởi động được sidecar. Ở dev, đặt CUTOUT_PYTHON tới runtime có Pillow/NumPy. Chi tiết: {error}"
                )
            })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Sidecar không có stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Sidecar không có stdout".to_string())?;

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn request(&mut self, request: &Value) -> Result<Value, String> {
        let line = serde_json::to_string(request)
            .map_err(|error| format!("Không serialize được request: {error}"))?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|error| format!("Không gửi được request tới sidecar: {error}"))?;

        let mut response = String::new();
        let read = self
            .stdout
            .read_line(&mut response)
            .map_err(|error| format!("Không đọc được response từ sidecar: {error}"))?;
        if read == 0 {
            return Err("Sidecar đã kết thúc trước khi trả response".to_string());
        }

        serde_json::from_str(response.trim_end())
            .map_err(|error| format!("Response sidecar không hợp lệ: {error}"))
    }
}

impl Drop for CoordinatorProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct CoordinatorState {
    process: Arc<Mutex<Option<CoordinatorProcess>>>,
}

impl CoordinatorState {
    fn new() -> Self {
        Self {
            process: Arc::new(Mutex::new(None)),
        }
    }
}

fn blocking_request(
    app: AppHandle,
    process_state: Arc<Mutex<Option<CoordinatorProcess>>>,
    request: Value,
) -> Result<Value, String> {
    let mut process_guard = process_state
        .lock()
        .map_err(|_| "Coordinator state bị poison".to_string())?;

    let needs_spawn = process_guard
        .as_mut()
        .map(|process| !process.is_running())
        .unwrap_or(true);

    if needs_spawn {
        *process_guard = Some(CoordinatorProcess::spawn(&app)?);
    }

    let process = process_guard
        .as_mut()
        .ok_or_else(|| "Không có coordinator process".to_string())?;

    match process.request(&request) {
        Ok(response) => Ok(response),
        Err(first_error) => {
            *process_guard = Some(CoordinatorProcess::spawn(&app)?);
            let retry = process_guard
                .as_mut()
                .ok_or_else(|| "Không restart được coordinator".to_string())?;
            retry.request(&request).map_err(|retry_error| {
                format!("Sidecar lỗi và retry thất bại. Lần đầu: {first_error}. Retry: {retry_error}")
            })
        }
    }
}

#[tauri::command]
async fn coordinator_request(
    app: AppHandle,
    state: State<'_, CoordinatorState>,
    request: Value,
) -> Result<Value, String> {
    let process_state = Arc::clone(&state.process);
    tauri::async_runtime::spawn_blocking(move || blocking_request(app, process_state, request))
        .await
        .map_err(|error| format!("Coordinator task bị hủy: {error}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(CoordinatorState::new())
        .invoke_handler(tauri::generate_handler![coordinator_request])
        .run(tauri::generate_context!())
        .expect("error while running Local POD Cutout Editor");
}
