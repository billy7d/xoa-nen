# V3 Hybrid Cutout

## Contract phát hành

- `LEGACY_V1` gọi implementation đóng băng từ Git HEAD `ed9be77`; tolerance/softness và alpha phải pixel-exact.
- `V3_BALANCED` dùng V1 evidence, robust spatial background field, GrabCut proxy và native unknown-band refinement.
- `V3_AI_LOCAL` chỉ nhận model-pack ONNX đã xác minh; proposal lỗi mới fallback V3 Balanced, còn matte lỗi giữ proposal nhưng trả `AI_LOCAL_DEGRADED`/`NEEDS_PROTECTION` cùng `fallback_reason`.
- Suy luận màu dùng sRGB copy theo ICC. Canonical RGB không bị ghi ngược.
- `LEGACY_V1` và `V3_BALANCED` luôn là `SOURCE_ALPHA × CUTOUT_ALPHA`.
- `V3_AI_LOCAL` mặc định `source_alpha_mode=PRESERVE`, nên output không vượt alpha nguồn. Chỉ `RECOVER_PRIOR_CUTOUT` mới khôi phục lỗ alpha cũ bên trong `object_candidate`; chế độ này phải được chọn có chủ ý cho cutout cũ bị hỏng.
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

Hai ảnh thật thuộc corpus private của người dùng và không được commit/redistribute. Catalog phát hành chưa có artifact `READY`; hai pack local hiện đã qua kiểm tra chữ ký/SHA-256/backend CPU và smoke test, nhưng qualification 100 ảnh matte 16-bit và parity CoreML/CPU vẫn là điều kiện để phát hành.

## Runtime AI local đã áp dụng

- BiRefNet-compatible ONNX adapter tạo base alpha proposal, hỗ trợ logits/alpha, layout NCHW/NHWC và cache session.
- ViTMatte-compatible adapter nhận ảnh + trimap nối channel hoặc hai input riêng, chỉ refine unknown band trong ROI.
- SAM2-compatible topology gate là tùy chọn; chỉ gate membership, không ghi alpha fractional và không được phép xóa sạch proposal khi mask rỗng.
- Provider được chọn theo manifest và khả năng runtime theo thứ tự TensorRT, CUDA, DirectML, CoreML, CPU.
- Hai pack local BiRefNet Lite-matting và ViTMatte đã được cài trong `models/` bị bỏ qua bởi Git; SAM2 chưa có ONNX artifact nên chưa tham gia inference.

Chi tiết manifest và trạng thái model weights xem [MODEL_PACK_RUNTIME.md](./MODEL_PACK_RUNTIME.md).

## Bảo toàn vật thể v4

- `object_candidate` là proposal recall cao, tách biệt với `edge_matte`. Graph-cut chỉ là evidence/seed và không được quyền xóa candidate AI.
- Khi dùng BiRefNet Lite 512, QUALITY tự chạy full-context kèm tile phần đầu từng component; foreground click vẫn được ưu tiên. Component detail phải nối với component cha, nằm trong growth cap và được feather mép tile.
- Tool **Khóa vật thể** lưu `foreground_points`, `background_points`, `protection_mode=CONSERVATIVE` và `shadow_policy=REMOVE` vào processing manifest. Shift-click bổ sung chi tiết rời.
- ViTMatte chỉ nhận trimap có `sure_foreground`, `sure_background` và unknown band tăng theo độ phân giải. Semantic cao chứng minh membership chứ không tự động khóa alpha=1; pixel trong core `sure_foreground` mới bị clamp.
- Khi proposal AI và graph-cut bất đồng lớn mà chưa có foreground click, result trả `NEEDS_PROTECTION`; UI yêu cầu khóa vật thể thay vì coi kết quả tự động là đạt.
- SAM2 chỉ được dùng qua ONNX export đã qualification; raw checkpoint PyTorch không được nạp bởi app.

## QA nhựa trong 2026-08-16

- Ảnh private ba ly 1254² xác nhận RGB output cũ giống nguồn tuyệt đối; lỗi nắp nằm hoàn toàn ở alpha gần nhị phân.
- Bản nâng cấp tăng detail proxy nắp trái từ 59,8% lên 74,5% và nắp giữa từ 69,2% lên 89,9%; moon-tip false positive giảm từ 525 xuống 0 pixel `alpha>=128`.
- Ống giữa/phải đã chuyển từ gần đục sang alpha mean 157/165; thân/vành giữ 100% detail, lỗ quai giữ IoU 99,79–99,85%.
- Pack 512 hiện tại và ViTMatte QUInt8 chỉ có `runtime_ready`; chưa có `quality_qualified`. Matte stage lỗi sẽ trả `AI_LOCAL_DEGRADED` thay vì báo READY.

Phân tích, regression gate và nguồn dữ liệu xem [TRANSPARENT_DETAIL_QUALIFICATION.md](./TRANSPARENT_DETAIL_QUALIFICATION.md).

## QA bảo toàn vật thể 2026-08-11

- Regression synthetic kiểm tra foreground seed, Shift-click detail rời, lỗ quai, phục hồi alpha đã bị xóa, ViTMatte unknown-only, prompt input SAM2 ONNX và persistence manifest.
- Ảnh ly private được chạy ngoài Git với một foreground click: unknown band 1,8651%, 284.577 pixel `sure_foreground` và 0 pixel bị matte hạ dưới 0,95; lỗ quai có alpha trung bình 0,025689.
- Các chỉ số recall/F1 tuyệt đối chỉ được công bố sau khi matte ground-truth được người dùng duyệt riêng; trước đó, candidate bất đồng vẫn phải trả `NEEDS_PROTECTION`.
