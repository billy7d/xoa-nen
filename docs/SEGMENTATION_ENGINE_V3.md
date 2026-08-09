# V3 Hybrid Cutout

## Contract phát hành

- `LEGACY_V1` gọi implementation đóng băng từ Git HEAD `ed9be77`; tolerance/softness và alpha phải pixel-exact.
- `V3_BALANCED` dùng V1 evidence, robust spatial background field, GrabCut proxy và native unknown-band refinement.
- `V3_AI_LOCAL` chỉ nhận model-pack ONNX đã xác minh; mọi lỗi model fallback V3 Balanced và ghi `fallback_reason`.
- Suy luận màu dùng sRGB copy theo ICC. Canonical RGB không bị ghi ngược.
- Alpha cuối luôn là `SOURCE_ALPHA × CUTOUT_ALPHA`.
- POD decontamination chỉ sửa RGB ở pixel có `0 < alpha < 1`; Master giữ RGB canonical.

## Các regression v2 đã loại bỏ

- Không dùng pixel-edge làm hard veto.
- Không dilate sure-foreground ra nền, không fill hole mù và không blend matte cố định toàn ảnh.
- Palette/field không được tự xóa vật thể trung tâm khi bao phủ gần toàn ảnh; graph-cut và V1 support tạo guard, vùng bất đồng được đánh dấu Needs Review.
- Smart Wand dùng chi phí trên liên kết pixel, seed patch cố định và không nở selection ngoài membership.

## Gate đã chạy ngày 2026-08-09

- Synthetic unit/integration: 20/20 pass; navigation: 3/3 pass; TypeScript và Rust `cargo check` pass.
- Ảnh navy 3661×2953: nền navy liên thông xóa 99,837%; far-color core giữ 100%.
- Ảnh ly 1122×1402 QUALITY: body alpha mean 0,991; 98,20% body >0,95; wall alpha xấp xỉ 0; wall speckle 0%; kết quả có Needs Review cho slab/shadow.
- Benchmark cup 512² median: FAST khoảng 1,95× V1; QUALITY khoảng 5,77× V1.

Hai ảnh thật thuộc corpus private của người dùng và không được commit/redistribute. Catalog AI chưa có artifact `READY`: qualification 100 ảnh matte 16-bit và parity CoreML/CPU vẫn là điều kiện để ký model-pack phát hành.
