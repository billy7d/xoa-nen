# Model-pack AI local

`V3_AI_LOCAL` hiện chạy theo chuỗi:

1. BiRefNet tạo `base_alpha_proposal` ở kích thước inference cố định.
2. SAM2-compatible tùy chọn chỉ làm membership/topology gate, không được ghi alpha fractional.
3. V3 Balanced dùng proposal để tạo foreground/background seed và giữ các vùng chắc chắn.
4. ViTMatte chạy trên bounding ROI của unknown band và chỉ được sửa pixel chưa chắc chắn.

Mọi model phải được đóng gói thành `.cutout-modelpack` có manifest, SHA-256 và chữ ký Ed25519 hợp lệ. Runtime không tải hoặc thực thi Python code của model; checkpoint PyTorch thô, đặc biệt `.pt` của SAM2, không được cài trực tiếp.

Git chỉ lưu manifest đã ký trong `model-manifests/`, không lưu ONNX weights. `npm run models:provision` tải đúng URL/revision đã pin, kiểm tra size, SHA-256, chữ ký và policy rồi mới cài vào `models/`. Thư mục weights này bị Git ignore nhưng được Tauri bundle vào artifact desktop local; vì vậy Windows và macOS dùng cùng contract mà repository không bị phình lớn.

`runtime_ready` chỉ xác nhận artifact, chữ ký, policy và backend đủ để chạy. `quality_qualified` chỉ được bật khi manifest đã qua corpus/metric Phase 0. UI và release tooling không được suy diễn quality từ `installed=true`.

## Contract manifest

Các trường runtime quan trọng:

```json
{
  "schema_version": "1.0.0",
  "model_id": "birefnet-qualified-revision",
  "revision": "pin-exact-revision",
  "role": "base_alpha_proposal",
  "adapter": "birefnet-v1",
  "input_size": [1024, 1024],
  "input_layout": "NCHW",
  "normalization": "imagenet",
  "output_layout": "NCHW",
  "output_activation": "auto",
  "output_semantics": "foreground",
  "qualified_backends": ["CPUExecutionProvider"],
  "runtime_remote_code_allowed": false,
  "commercial_pod_allowed": true,
  "redistribution_allowed": true,
  "artifacts": [{
    "filename": "model.onnx",
    "role": "base_alpha_proposal",
    "sha256": "...",
    "size": 123
  }]
}
```

`output_activation: "auto"` giữ nguyên output đã nằm trong `[0, 1]` và áp dụng sigmoid khi model trả logits. Có thể dùng `"sigmoid"` hoặc `"identity"` khi artifact đã được qualification với contract cụ thể.

## BiRefNet

BiRefNet là adapter chính cho `base_alpha_proposal`. Artifact cần nhận ảnh RGB và trả một mask 2D hoặc tensor dạng `[1, 1, H, W]`. Manifest có thể đặt `input_name`, `output_name`, `output_index`, `output_channel`, `mean`, `std`, `color_order` và `input_layout` khi ONNX export không dùng tên/mặc định chuẩn.

## ViTMatte

ViTMatte dùng ảnh RGB và trimap. Runtime hỗ trợ hai dạng ONNX export:

- một input 4 kênh ảnh + trimap nối theo channel; để trống `trimap_input_name`;
- hai input riêng; đặt `trimap_input_name` đúng tên input trong ONNX.

Output được resize về ROI gốc rồi chỉ ghi vào unknown band. Vùng background/foreground chắc chắn và giới hạn `SOURCE_ALPHA` luôn được giữ lại.

## SAM2 tùy chọn

SAM2 chỉ được dùng như lớp membership/topology. Model-pack phải là ONNX export có adapter `sam2-conditional-v1` hoặc `sam2-mask-prompt-v1`; nếu có mask prompt thì khai báo `prompt_input_name`. Runtime không dùng output SAM2 để tạo alpha fractional và sẽ bỏ qua topology nếu membership rỗng.

Việc chuyển checkpoint SAM2 sang ONNX, xác định input/output names và kiểm tra parity phải hoàn thành trước khi ký pack. Không đổi candidate thành `READY` chỉ vì ONNX load được.

## Watermark inpainting

Watermark Removal v2 dùng cùng `LocalModelRuntime`, không có runtime AI riêng. Role hiện được hỗ trợ:

- `watermark_inpaint_fast`: adapter `lama-v1`, ưu tiên cho ROI watermark vừa/lớn khi model-pack đã cài.
- `watermark_inpaint_quality`: adapter `lama-v1` hoặc `generic-inpaint`, tùy chọn cho Maximum Quality sau qualification riêng.

Manifest phải khai báo hoặc để runtime tự dò:

```json
{
  "role": "watermark_inpaint_fast",
  "adapter": "lama-v1",
  "image_input_name": "image",
  "mask_input_name": "mask",
  "output_name": "output",
  "input_size": [1024, 1024],
  "input_layout": "NCHW",
  "mask_input_layout": "NCHW",
  "normalization": "zero_one",
  "output_layout": "NCHW",
  "output_range": "zero_one",
  "qualified_backends": ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
}
```

Runtime chỉ gửi ROI đã mở rộng theo mask, rồi composite về ảnh native bằng soft mask. Pixel ngoài ROI/blend ring phải giữ nguyên byte-for-byte. Watermark v3 chỉ chấp nhận output từ model AI local: nếu role chưa cài hoặc inference lỗi, preview dừng và báo rõ để người dùng cài/sửa model-pack; tuyệt đối không fallback sang Deblend, Patch, Telea hay thuật toán lấp nền khác.

## Trạng thái hiện tại

Checkout local đã cài hai pack ONNX trong thư mục bị bỏ qua bởi Git `models/`:

- `studioludens-birefnet-lite-512`, revision `4a3c40c36c94093cc1e724d9ea428b8fa4b57dc7`, license MIT, tạo proposal bằng CPU.
- `xenova-vitmatte-small-composition-1k`, revision `6bc1297f6140f055a227b6d2cfe8c093281f35d2`, model gốc Apache-2.0, refine unknown ROI bằng CPU.

Smoke test qua worker đã gọi cả hai model, output float32 hữu hạn trong `[0, 1]`. SAM2 vẫn tùy chọn. Watermark v3 có thể dùng pack cục bộ `local-opencv-lama-watermark-512` (LaMa ONNX từ OpenCV, `watermark_inpaint_fast`, BGR, 512×512); pack nằm trong `models/` bị Git ignore, có SHA-256/chữ ký hợp lệ và đang ở `runtime_ready_quality_pending`. Bản desktop bundle pack này rồi seed một lần vào AppData khi khởi động, vì pipeline không có fallback thuật toán. Các pack chưa được Phase 0 vẫn cần corpus riêng trước khi được gắn `quality_qualified` cho POD thương mại.

Watermark v3.1 tuân thủ đúng contract OpenCV LaMa: mask được resize rồi nhị phân hóa trước inference. Auto detect ưu tiên nhận diện hình học Gemini sparkle ở góc phải dưới và chặn mọi mask classical vượt 6% diện tích ảnh. Cả auto lẫn brush dùng chung pipeline AI local và chỉ cho phép preview/commit khi output vượt quality gate; kết quả bị từ chối không thay đổi working RGB.
