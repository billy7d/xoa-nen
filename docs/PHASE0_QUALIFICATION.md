# Phase 0 qualification checklist

Không đổi candidate thành default chỉ vì load được model. Mỗi artifact phải có manifest pin revision + SHA-256 + license code/weights + preprocess version, rồi qua toàn bộ gate dưới đây.

## Dataset

- Tối thiểu 100 ảnh sở hữu hợp pháp, báo cáo riêng hard edge, text, line art, distressed, detached component, glow/gradient, white/black background, low contrast và anti-alias.
- Ground-truth matte 16-bit; split cố định và hash danh sách file.
- Composite QA trên trắng, đen và ít nhất ba màu garment.

## Gate chất lượng

- Hard-edge Boundary F1 ±2 px ≥ 0,98.
- Soft/complex Boundary F1 ±4 px ≥ 0,93.
- Recall component có kích thước ≥4 px ≥98%.
- Backend tăng tốc giảm Boundary F1 không quá 0,5 điểm phần trăm so với CPU FP32.
- Kiểm riêng source alpha, hidden RGB và topology/negative space.

## Gate máy đích

- Windows 11 + GTX 1660 Super 6 GB: native, không WSL/nvcc/compiler.
- macOS 15+ arm64 + M4: MPS/CPU recovery an toàn và parity đã đo.
- FAST p95 ≤30 giây; QUALITY p95 ≤90 giây trên ít nhất 30 ảnh 20–30 MP.
- Local refine ROI p95 ≤10 giây; crash/OOM phải restart worker và báo profile fallback.

## Candidate order

1. BiRefNet dynamic/standard 1024 FP16; Lite fallback.
2. SAM 2.1 Base+ chỉ cho membership/topology có prompt, không ghi fractional alpha.
3. ViTMatte-B ROI 512; ViTMatte-S fallback.
4. BiRefNet HR chỉ cho MAX/profile đã benchmark.
5. BEN2 Base và FeyNoBg là challenger; BRIA RMBG không bundle cho commercial POD.

File report Phase 0 phải lưu runtime/driver/OS, peak memory, latency từng stage, metric từng category, model manifest và mọi fallback. Profile bị invalidate khi model, preprocess, runtime, driver hoặc OS đổi.

