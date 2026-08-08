# Local POD Cutout Editor

Ứng dụng desktop local-first để tách nền artwork/ảnh AI cho DTG/DTF. Đây là implementation chạy được của nền tảng V1 trong PRD v2.2: single-image editor, project không phá hủy, preflight POD và ba output có contract riêng.

## Chức năng hiện có

- Tauri v2 + React/TypeScript; không mở HTTP/FastAPI localhost.
- Coordinator Python giao tiếp line-delimited JSON qua stdin/stdout.
- Processing chạy trong worker subprocess tuần tự; worker chết được restart mà không làm sập UI.
- Canonical decode PNG/JPEG/static WebP, EXIF 1–8, source alpha và CMYK → sRGB có cảnh báo.
- ARTWORK engine: background sample từ biên, color range, softness, flood-fill contiguous; không tự xóa component rời.
- Brush Keep/Remove native-coordinate, Magic Wand contiguous/global, undo/redo bằng tile delta.
- Alpha authoritative float32, tile 512 px nén zlib trong `.cutoutproj`; manifest/journal được ghi nguyên tử.
- Preview checker/white/black/garment, component inspector và effective-PPI preflight.
- Export `MASTER_SOURCE_FAITHFUL`, `POD_READY` và `ALPHA_ONLY` 16-bit.
- Release sidecar bằng PyInstaller; máy người dùng không cần Python.

Model AI chưa được bundle tự động. BiRefNet/SAM2/ViTMatte chỉ xuất hiện dưới dạng candidate manifest cho đến khi exact revision, checksum, license và benchmark GTX 1660 Super/M4 đều đạt gate. Runtime không dùng `trust_remote_code` và không tải model từ mạng.

## Chạy development

Yêu cầu Node.js, Rust stable và Python có Pillow + NumPy.

```sh
npm install
python3 -m venv .venv
.venv/bin/pip install -r sidecar/requirements-build.txt
./scripts/dev-desktop.sh
```

Nếu Python cần dùng không nằm ở `.venv`, đặt `CUTOUT_PYTHON` khi chạy script dev.

## Kiểm thử

```sh
npm run check
cargo check --manifest-path src-tauri/Cargo.toml
```

Test sidecar bao phủ canonical RGBA, Unicode path, EXIF orientation, animated WebP rejection, source-alpha constraint, component rời, worker crash recovery, brush/undo/redo, preflight, master RGB delta 0, POD sRGB và alpha 16-bit.

## Build desktop

Build phải chạy trên chính hệ điều hành đích; PyInstaller không cross-compile executable Windows từ macOS.

```sh
npm run build:desktop-assets
npm run tauri build -- --bundles app
```

- macOS output: `src-tauri/target/release/bundle/macos/Local POD Cutout Editor.app`
- Windows output được tạo bởi cùng lệnh trên Windows 11 và chứa `cutout-sidecar.exe`; không cần WSL, Python, CUDA toolkit, `nvcc` hay compiler ở máy người dùng.
- Ký/notarize và installer production là release step riêng; debug bundle không phải artifact phân phối cuối.

## Cấu trúc

```text
src/                         React editor và canvas preview
src-tauri/                   Native shell, capability và sidecar supervisor
sidecar/cutout_sidecar/      Image core, project store, edits, export, preflight
tests/                       Contract/integration tests
scripts/                     Dev, test và standalone-sidecar build
projects/                    Dev projects (ignored; release dùng app-data)
```

## Giới hạn của build hiện tại

- Engine ARTWORK color/edge đã dùng được cho hard-edge artwork và nền tương đối đồng nhất. Hair/fur/glow phức tạp vẫn cần model stack sau Phase 0 hoặc sửa tay.
- Chưa có batch, provider preset, mockup/listing, shadow chuyển nền, glass/refraction, upscaler hay vectorizer.
- Chưa công bố SLA/quality claim cho GTX 1660 Super và M4 cho đến khi chạy đủ dataset POD hợp pháp theo PRD.
- Viewer hiện dùng preview mip 2K và alpha native-resolution ở sidecar; WebGL tile renderer đầy đủ là hạng mục tiếp theo cho ảnh cực lớn.

PRD nguồn: [v2.2](./PRD_Tach_Vat_The_Xoa_Phong_Local_Windows_1660Super_MacBook_M4_VI_v2.2.txt).

# xoa-nen
