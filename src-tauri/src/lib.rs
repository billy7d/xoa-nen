use serde_json::Value;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, State};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "macos")]
use tauri::Emitter;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

fn copy_model_directory(source: &Path, destination: &Path) -> Result<(), String> {
    fs::create_dir_all(destination).map_err(|error| {
        format!(
            "Không tạo được thư mục model local {}: {error}",
            destination.display()
        )
    })?;

    for entry in fs::read_dir(source)
        .map_err(|error| format!("Không đọc được model bundled {}: {error}", source.display()))?
    {
        let entry = entry.map_err(|error| format!("Không đọc được file model bundled: {error}"))?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let kind = entry
            .file_type()
            .map_err(|error| format!("Không xác định được loại file model: {error}"))?;
        if kind.is_dir() {
            copy_model_directory(&source_path, &destination_path)?;
        } else if kind.is_file() {
            fs::copy(&source_path, &destination_path).map_err(|error| {
                format!(
                    "Không sao chép được model AI local {}: {error}",
                    source_path.display()
                )
            })?;
        }
    }
    Ok(())
}

fn seed_bundled_model_packs(source_root: &Path, destination_root: &Path) -> Result<(), String> {
    if !source_root.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(destination_root).map_err(|error| {
        format!(
            "Không tạo được thư mục model của ứng dụng {}: {error}",
            destination_root.display()
        )
    })?;

    for entry in fs::read_dir(source_root).map_err(|error| {
        format!(
            "Không đọc được thư mục model bundled {}: {error}",
            source_root.display()
        )
    })? {
        let entry = entry.map_err(|error| format!("Không đọc được model bundled: {error}"))?;
        let source_pack = entry.path();
        let destination_pack = destination_root.join(entry.file_name());
        // Chỉ seed một model-pack hoàn chỉnh và không ghi đè model do người dùng đã cài.
        if source_pack.is_dir()
            && source_pack.join("manifest.json").is_file()
            && !destination_pack.exists()
        {
            copy_model_directory(&source_pack, &destination_pack)?;
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, serde::Serialize)]
struct MacosMagnifyPayload {
    magnification: f64,
    x: f64,
    y: f64,
}

#[cfg(target_os = "macos")]
fn install_macos_magnify_monitor(app: AppHandle) {
    use block2::RcBlock;
    use objc2_app_kit::{NSEvent, NSEventMask};
    use std::ptr::NonNull;

    let handler = RcBlock::new(move |event_ptr: NonNull<NSEvent>| -> *mut NSEvent {
        let event = unsafe { event_ptr.as_ref() };
        let location = event.locationInWindow();
        let _ = app.emit(
            "macos-preview-magnify",
            MacosMagnifyPayload {
                magnification: event.magnification(),
                x: location.x,
                y: location.y,
            },
        );
        event_ptr.as_ptr()
    });

    // The monitor is owned by AppKit for the lifetime of the process. Keeping
    // the token alive avoids WKWebView discarding native trackpad magnification.
    if let Some(monitor) = unsafe {
        NSEvent::addLocalMonitorForEventsMatchingMask_handler(NSEventMask::Magnify, &handler)
    } {
        std::mem::forget(monitor);
    }
}

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

        let app_models_dir = app_data.join("models");
        let bundled_models_dir = resource_dir.join("models");
        let dev_models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models");
        // Dev dùng pack trong workspace; bản đóng gói dùng resource rồi seed vào AppData.
        let seed_source = if bundled_models_dir.is_dir() {
            bundled_models_dir
        } else {
            dev_models_dir
        };
        seed_bundled_model_packs(&seed_source, &app_models_dir)?;

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
            .env("CUTOUT_MODELS_DIR", &app_models_dir);

        #[cfg(target_os = "windows")]
        // Ngăn sidecar và các worker con tự tạo cửa sổ terminal khi app chạy nền.
        command.creation_flags(CREATE_NO_WINDOW);

        let sidecar_stderr = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            // Ghi log theo từng phiên để không phình AppData mà vẫn có dữ liệu chẩn đoán lỗi.
            .open(app_data.join("sidecar.stderr.log"))
            .map_err(|error| format!("Không mở được log sidecar: {error}"))?;

        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(sidecar_stderr))
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
        .setup(|_app| {
            #[cfg(target_os = "macos")]
            install_macos_magnify_monitor(_app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![coordinator_request])
        .run(tauri::generate_context!())
        .expect("error while running Local POD Cutout Editor");
}

#[cfg(test)]
mod tests {
    use super::seed_bundled_model_packs;
    use std::fs;

    #[test]
    fn seeds_complete_bundled_packs_without_overwriting_user_model() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time hợp lệ")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cutout-model-seed-test-{}-{unique}",
            std::process::id()
        ));
        let source_root = root.join("bundled");
        let destination_root = root.join("app-data").join("models");

        let existing_source = source_root.join("existing");
        fs::create_dir_all(&existing_source).expect("tạo source pack");
        fs::write(existing_source.join("manifest.json"), "new-manifest").expect("ghi manifest source");
        fs::write(existing_source.join("model.onnx"), b"new-model").expect("ghi model source");

        let existing_destination = destination_root.join("existing");
        fs::create_dir_all(&existing_destination).expect("tạo model người dùng");
        fs::write(existing_destination.join("manifest.json"), "user-manifest")
            .expect("ghi manifest người dùng");
        fs::write(existing_destination.join("model.onnx"), b"user-model")
            .expect("ghi model người dùng");

        let fresh_source = source_root.join("fresh");
        fs::create_dir_all(&fresh_source).expect("tạo pack mới");
        fs::write(fresh_source.join("manifest.json"), "fresh-manifest").expect("ghi manifest mới");
        fs::write(fresh_source.join("model.onnx"), b"fresh-model").expect("ghi model mới");

        let incomplete_source = source_root.join("incomplete");
        fs::create_dir_all(&incomplete_source).expect("tạo pack thiếu manifest");
        fs::write(incomplete_source.join("model.onnx"), b"ignored").expect("ghi model thiếu manifest");

        seed_bundled_model_packs(&source_root, &destination_root).expect("seed model bundled");

        assert_eq!(
            fs::read(existing_destination.join("model.onnx")).expect("đọc model người dùng"),
            b"user-model"
        );
        assert_eq!(
            fs::read(destination_root.join("fresh").join("model.onnx")).expect("đọc model mới"),
            b"fresh-model"
        );
        assert!(!destination_root.join("incomplete").exists());

        fs::remove_dir_all(root).expect("dọn thư mục test");
    }
}
