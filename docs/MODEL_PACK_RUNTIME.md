# Model-pack AI local

`V3_AI_LOCAL` hiện chạy theo chuỗi:

1. BiRefNet tạo `base_alpha_proposal` ở kích thước inference cố định.
2. SAM2-compatible tùy chọn chỉ làm membership/topology gate, không được ghi alpha fractional.
3. V3 Balanced dùng proposal để tạo foreground/background seed và giữ các vùng chắc chắn.
4. ViTMatte chạy trên bounding ROI của unknown band và chỉ được sửa pixel chưa chắc chắn.

Mọi model phải được đóng gói thành `.cutout-modelpack` có manifest, SHA-256 và chữ ký Ed25519 hợp lệ. Runtime không tải hoặc thực thi Python code của model; checkpoint PyTorch thô, đặc biệt `.pt` của SAM2, không được cài trực tiếp.

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

## LaMa-compatible watermark inpainting

Watermark V2 nhận pack có `role: "image_inpainting"` và `adapter: "lama-inpaint-v1"`. Runtime chỉ chạy trên ROI bao quanh mask, sau đó chỉ ghi lại pixel có mask; pixel còn lại trong ROI và toàn ảnh giữ byte-exact.

- Một-input: tensor 4 kênh `RGB + mask`, để trống `mask_input_name`.
- Hai-input: đặt `mask_input_name` cho tensor mask một kênh.
- `mask_semantics` mặc định là `"masked_one"`; dùng `"valid_one"` khi ONNX export đảo nghĩa mask.
- `output_range` là `"zero_one"` hoặc `"minus_one_one"`.

Pack LaMa chỉ được ký/phát hành sau khi quyền thương mại và phân phối của checkpoint cụ thể đã được xác minh; catalog hiện chỉ là candidate, không tự tải artifact chưa qualification.

## SAM2 tùy chọn

SAM2 chỉ được dùng như lớp membership/topology. Model-pack phải là ONNX export có adapter `sam2-conditional-v1` hoặc `sam2-mask-prompt-v1`; nếu có mask prompt thì khai báo `prompt_input_name`. Runtime không dùng output SAM2 để tạo alpha fractional và sẽ bỏ qua topology nếu membership rỗng.

Việc chuyển checkpoint SAM2 sang ONNX, xác định input/output names và kiểm tra parity phải hoàn thành trước khi ký pack. Không đổi candidate thành `READY` chỉ vì ONNX load được.

## Trạng thái hiện tại

Checkout local đã cài hai pack ONNX trong thư mục bị bỏ qua bởi Git `models/`:

- `studioludens-birefnet-lite-512`, revision `4a3c40c36c94093cc1e724d9ea428b8fa4b57dc7`, license MIT, tạo proposal bằng CPU.
- `xenova-vitmatte-small-composition-1k`, revision `6bc1297f6140f055a227b6d2cfe8c093281f35d2`, model gốc Apache-2.0, refine unknown ROI bằng CPU.

Smoke test qua worker đã gọi cả hai model, output float32 hữu hạn trong `[0, 1]`; latency mẫu trên CPU lần lượt khoảng 5,8 giây và 2,2 giây cho ảnh 128×96. SAM2 chưa cài artifact ONNX nên topology giữ trạng thái tùy chọn và không chặn pipeline. Đây là qualification ở mức runtime, chưa phải release qualification: cần chạy Phase 0 trên corpus riêng trước khi dùng cho POD thương mại.
