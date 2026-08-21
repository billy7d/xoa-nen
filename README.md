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

Git không lưu model weights để repository luôn gọn. Script provision tải đúng revision ONNX đã pin, kiểm tra size, SHA-256 và chữ ký Ed25519 rồi mới đặt vào `models/`; Tauri bundle thư mục local này vào bản desktop và seed một lần sang AppData. Runtime không dùng `trust_remote_code`. BiRefNet Lite 512 tạo proposal, ViTMatte Small refine unknown ROI và OpenCV LaMa phục hồi watermark bằng AI local; SAM2 vẫn là topology gate tùy chọn. Các pack hiện ở mức `runtime_ready_quality_pending`, chưa phải công bố chất lượng thương mại.

## Chạy development

Yêu cầu Node.js, Rust stable và Python có Pillow + NumPy.

```sh
npm install
python3 -m venv .venv
.venv/bin/pip install -r sidecar/requirements-build.txt
# Tải ba model ONNX đã pin; file weights nằm local và bị Git ignore.
npm run models:provision
./scripts/dev-desktop.sh
```

`npm run tauri dev` và `npm run tauri build` cũng tự gọi provision trước khi chạy. Nếu Python cần dùng không nằm ở `.venv`, đặt `CUTOUT_PYTHON` khi chạy script dev hoặc `CUTOUT_BUILD_PYTHON` khi build.

### Đồng bộ sang macOS

Sau khi pull source trên macOS:

```sh
npm install
python3 -m venv .venv
.venv/bin/pip install -r sidecar/requirements-build.txt
npm run models:provision
npm run tauri dev
```

Ba pack được tái tạo local, không tải qua Git:

- `studioludens-birefnet-lite-512`: proposal cho `V3_AI_LOCAL`.
- `xenova-vitmatte-small-composition-1k`: matte biên/chi tiết trong suốt.
- `local-opencv-lama-watermark-512`: inpaint AI local cho watermark.

Muốn chỉ kiểm tra pack đã cài mà không tải lại, chạy `.venv/bin/python scripts/provision-models.py --verify-only`.

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
