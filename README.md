# Local POD Cutout Editor

Ứng dụng desktop local-first để tách nền artwork/ảnh sản phẩm cho DTG/DTF. V3 giữ engine V1 pixel-exact, bổ sung hybrid spatial/graph-cut, Smart Wand, chọn nhiều vật thể và model-pack AI local có xác minh.

## Chức năng hiện có

- Tauri v2 + React/TypeScript; không mở HTTP/FastAPI localhost.
- Coordinator Python giao tiếp line-delimited JSON qua stdin/stdout.
- Processing chạy trong worker subprocess tuần tự; worker chết được restart mà không làm sập UI.
- Canonical decode PNG/JPEG/static WebP, EXIF 1–8, source alpha và CMYK → sRGB có cảnh báo.
- Ba profile độc lập: `V3 Cân bằng` mặc định, `V3 AI Local` tùy chọn và `V1 nền phẳng` đúng output V1.
- V3 classical: robust constant/affine/quadratic background field, consensus trimap, OpenCV GrabCut và refine native-resolution chỉ trong vùng bất định; không còn hard edge barrier/dilate/fill-hole/blend toàn ảnh của v2.
- Brush Keep/Remove native-coordinate; Wand có `Legacy Color` đúng V1 và `Smart` dùng Lab seed patch + link-cost geodesic, luôn preview trước commit.
- Subject candidates, chọn nhiều vật thể, vùng Needs Review màu vàng và fallback bảo thủ khi proposal bất đồng.
- Alpha authoritative float32, tile 512 px nén zlib trong `.cutoutproj`; manifest/journal được ghi nguyên tử.
- Preview checker/white/black/garment, component inspector và effective-PPI preflight.
- Export `MASTER_SOURCE_FAITHFUL`, `POD_READY` và `ALPHA_ONLY` 16-bit.
- Release sidecar bằng PyInstaller; máy người dùng không cần Python.

Model AI không bundle sẵn. App chỉ cài `.cutout-modelpack` ONNX có Ed25519 signature, SHA-256, size, commercial/redistribution policy và backend đã qualification; runtime ưu tiên CoreML rồi CPU, không dùng `trust_remote_code`. BiRefNet Lite-matting/ViTMatte-small/SAM 2.1 vẫn ở catalog qualification cho tới khi có artifact được ký và corpus 100 ảnh hợp pháp.

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

Test sidecar bao phủ V1 pixel-exact, hard edge/noise/corner/source-alpha, background gradient/weak edge, Smart Wand, preview/commit, subject selection, AI fallback, model-pack signature/corruption, worker crash recovery, master RGB delta 0 và alpha 16-bit.

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

- `V3 Cân bằng` vẫn là segmentation classical: tóc/lông/glow/kính hoặc vật thể thật sự không có tín hiệu thị giác có thể cần Keep/Remove hoặc model-pack đã qualification.
- Không model nào được đánh dấu `READY` trong catalog mặc định; `V3 AI Local` chỉ bật sau khi người dùng cài pack hợp lệ và sẽ fallback có thông báo nếu thiếu/OOM/backend lỗi.
- Chưa có batch, provider preset, mockup/listing, shadow chuyển nền, glass/refraction, upscaler hay vectorizer.
- Chưa công bố SLA/quality claim cho GTX 1660 Super và M4 cho đến khi chạy đủ dataset POD hợp pháp theo PRD.
- Viewer hiện dùng preview mip 2K và alpha native-resolution ở sidecar; WebGL tile renderer đầy đủ là hạng mục tiếp theo cho ảnh cực lớn.

PRD nguồn: [v2.2](./PRD_Tach_Vat_The_Xoa_Phong_Local_Windows_1660Super_MacBook_M4_VI_v2.2.txt).

Thiết kế và kết quả gate: [Segmentation engine V3](./docs/SEGMENTATION_ENGINE_V3.md).

# xoa-nen
