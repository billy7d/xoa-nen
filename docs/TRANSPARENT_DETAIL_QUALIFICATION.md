# Qualification chi tiết nhựa trong

Tài liệu này khóa các lỗi đã tìm thấy ngày 2026-08-16 trên ảnh private 1254×1254 có ba ly tím, nắp và ống hút trong. Ảnh private và matte không được commit hoặc phân phối.

## Kết luận pháp y ảnh

- PNG V3 cũ giữ RGB giống ảnh nguồn ở 100% pixel. Mọi chi tiết nhìn như bị mất đều do alpha, không phải do blur hay đổi màu RGB.
- Alpha cũ có 64,195% pixel bằng 0, 33,737% bằng 255 và chỉ 2,068% alpha trung gian. Đây là mask gần nhị phân, không phải matte tốt cho nhựa trong.
- BiRefNet Lite hiện cài nhận ảnh 1254×1254 qua input cố định 512. [Model card của pack](https://huggingface.co/studioludens/birefnet-lite-512) cũng nêu giới hạn mất chi tiết biên ở ảnh lớn và khuyến nghị crop.
- Candidate `<0,05` từng bị khóa thành background; semantic `>=0,92` từng bị khóa thành foreground đục. ViTMatte chỉ được sửa dải unknown 4 px nên không có quyền cứu rail nắp bị loại hoặc làm mềm highlight trong suốt.
- Ba lõi ống hút cũ gần như alpha 255; mảng trắng hình trăng cạnh ống trái bị nhận nhầm là foreground. Vì vậy lỗi là semantic/trimap/membership, không phải thiếu sharpening.

`detail retention` dưới đây là proxy `sum(Sobel(source) × alpha) / sum(Sobel(source))` trong ROI. Nó đo lượng gradient nguồn còn đóng góp khi composite, không phải recall so với ground-truth; texture nền nhìn xuyên nhựa cũng góp vào chỉ số.

| Gate trên ảnh private | V3 cũ | V3 nâng cấp |
|---|---:|---:|
| Detail nắp trái | 59,8% | 74,5% |
| Detail nắp giữa | 69,2% | 89,9% |
| Detail nắp phải | 95,9% | 95,0% |
| Pixel `alpha < 16` nắp trái | 40,52% | 0,27% |
| Pixel `alpha < 16` nắp giữa | 22,07% | 1,12% |
| Moon-tip false positive `alpha >= 128` | 525 px | 0 px |
| Alpha trung bình lõi ống giữa/phải | gần đục | 157 / 165 |
| RGB khác nguồn | 0 px | 0 px |

Nắp phải giảm nhẹ proxy vì alpha đã mềm hơn; thân, vành kim loại vẫn giữ 100% detail và lỗ quai giữ IoU 99,79–99,85% so với bản cũ.

## Thay đổi pipeline đã áp dụng

1. QUALITY tự tìm component lớn và chạy tối đa bốn tile phần đầu vật thể. Tile dùng đúng input 512 nên tăng mật độ pixel thật thay vì resize lại 80% ảnh.
2. Detail tile chỉ được nhập nếu nối với component cha, nằm trong giới hạn tăng trưởng 8–32 px và được feather ở mép tile. Guard này chặn décor nền kéo dài thành false positive.
3. Trimap tăng theo độ phân giải, từ 2 px đến tối đa 24 px. Semantic cao chỉ chứng minh membership, không chứng minh opacity; rail, nhựa trong và highlight mảnh được để unknown cho ViTMatte.
4. `ALL_DETECTED` không còn làm mất các ly chưa được click. Chỉ `SELECTED` mới giới hạn candidate vào component có prompt.
5. Pipeline AI thiếu hoặc lỗi matte trả `AI_LOCAL_DEGRADED`/`NEEDS_PROTECTION`; `no_unknown_roi` là no-op hợp lệ, không còn báo thành công chỉ vì proposal đã chạy.
6. PNG RGBA mặc định `source_alpha_mode=PRESERVE`; output không được lớn hơn alpha nguồn. `RECOVER_PRIOR_CUTOUT` chỉ dùng có chủ ý khi phục hồi một cutout cũ đã bị xóa nhầm.
7. Trạng thái model tách `runtime_ready` khỏi `quality_qualified`; chữ ký/SHA/backend hợp lệ không còn đồng nghĩa đã qua benchmark chất lượng.

Script đo tái lập: `scripts/measure-transparent-detail.py`. Ví dụ:

```powershell
.venv\Scripts\python.exe scripts\measure-transparent-detail.py `
  --source 'C:\Users\billy\OneDrive\Desktop\nắp mẫu.png' `
  --candidate 'artifacts\lid-analysis\v3-upgraded.png' `
  --baseline 'C:\Users\billy\OneDrive\Desktop\nắp mẫu-codex.png' `
  --roi 'lid_left:210,354,440,370' `
  --roi 'lid_center:510,354,745,370' `
  --roi 'lid_right:815,354,1050,370'
```

## Gate regression riêng cho mẫu nắp

- Contract: 1254×1254 RGBA, RGB exact nguồn, alpha hữu hạn `[0,1]`, PNG alpha bằng `round(NPY×255)`.
- Detail nắp trái/giữa/phải tối thiểu 72% / 87% / 94%; `alpha<16` tối đa 2% / 3% / 5%.
- Alpha trung gian 16–239 ở nắp trái/giữa/phải tối thiểu 35% / 18% / 5%; gradient top-25% bị xóa dưới 16 tối đa 0,5%.
- ROI moon-tip `x=320..359, y=300..344`: alpha-equivalent tối đa 25 px, alpha tối đa 80, không có pixel `alpha>=128`.
- Mỗi lõi ống có mean alpha 120–253,5 và alpha 255 tối đa 70%; ít nhất hai trong ba ống có tỷ lệ `alpha>=240` không quá 60%.
- Lỗ quai IoU tối thiểu 99,5%, core lỗ không có alpha từ 16 trở lên; phần thân dưới IoU tối thiểu 99,5%; thân/vành detail tối thiểu 99,9%; quai đặc tối thiểu 94%.
- Đúng ba component chính có diện tích ít nhất 16 px; tổng rác component nhỏ hơn 16 px phải dưới 16 px.

Đây là regression gate cho đúng cảnh này, không thay thế bộ ground-truth đa cảnh.

## Nguồn benchmark và training

### Benchmark/chỉ nghiên cứu đến khi quyền dữ liệu được audit

| Nguồn | Dùng để bắt lỗi | Ràng buộc |
|---|---|---|
| [AIM-500](https://github.com/JizhiziLi/AIM) | 500 matte thật high-resolution, có 34 vật thể trong suốt | Dùng benchmark theo agreement của repo; không nhập mù vào training thương mại. |
| [Trans10K](https://github.com/xieenze/Segment_Transparent_Objects) | hơn 10 nghìn vật thể trong suốt, tốt cho membership/topology | Repo yêu cầu liên hệ tác giả cho mục đích thương mại. |
| [DIS5K](https://xuebinqin.github.io/dis/) TE3/TE4 | vật thể rất phức tạp, chi tiết mảnh và lỗ âm | Quyền ảnh không đủ rõ để dùng làm training thương mại; benchmark nội bộ. |
| [ZIM/MicroMat-3K](https://github.com/naver-ai/ZIM) | matte zero-shot, tóc/chi tiết/translucency | CC BY-NC 4.0; chỉ nghiên cứu phi thương mại. |
| [Transparent-460](https://github.com/AceCHQ/TransMatting) | vật thể trong suốt có GT | Repo ghi non-commercial và yêu cầu liên hệ nếu thương mại. |

### Training thương mại sạch

- Dùng CAD/SKU thuộc sở hữu, render nhiều IOR, roughness, thickness, tint, blur, góc nhìn và ánh sáng. [TOM-Net Rendering](https://github.com/guanyingc/TOM-Net_Rendering) cung cấp code MIT và công thức object mask + attenuation + refractive flow; không mặc nhiên tái sử dụng bộ 178K gốc vì paper cho biết background lấy từ MS COCO.
- Render bằng Blender; GPL của Blender không áp lên artwork/render tạo ra. Chỉ dùng asset đầu vào tự sở hữu hoặc có license phù hợp.
- [Poly Haven](https://polyhaven.com/license) cung cấp HDRI, texture và model chính thức theo CC0, cho phép commercial/AI training; không lấy logo, nội dung web hoặc user/example render.
- Dùng quy trình [PolarMatte](https://openaccess.thecvf.com/content/CVPR2024/html/Enomoto_PolarMatte_Fully_Computational_Ground-Truth-Quality_Alpha_Matte_Extraction_for_Images_and_CVPR_2024_paper.html) làm tham chiếu để tự chụp alpha GT bằng camera phân cực + LCD. Paper không phải một dataset có license commercial-ready.
- Bộ thật tối thiểu nên có 150–300 ảnh sở hữu, trong đó ít nhất 50 cảnh nắp/ống trong; split theo SKU và phiên chụp để không rò cùng vật thể vào train/test.

## Model challenger theo thứ tự an toàn

1. Giữ BiRefNet Lite ONNX làm proposal nhanh nhưng luôn tile ở QUALITY. A/B [BiRefNet chính thức](https://github.com/ZhengPeng7/BiRefNet) dynamic/1024 và [BiRefNet HR-matting](https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting) ở ROI/offline, không đặt HR làm mặc định GTX 1660 Super trước benchmark VRAM.
2. A/B [BEN2 Base](https://github.com/PramaLLC/BEN2) MIT như vote thứ hai; chỉ nhận nếu tăng recall nắp mà không tăng décor/halo.
3. [SAM 2.1](https://github.com/facebookresearch/sam2) Apache-2.0 chỉ làm membership/topology. ONNX export nội bộ phải có parity test; SAM không được ghi alpha fractional.
4. A/B ViTMatte-S FP32/FP16 với pack QUInt8 hiện tại trước khi chọn quality default; fine-tune ROI chỉ bằng dữ liệu có quyền rõ.
5. withoutBG/BiRefNet HR chỉ là challenger. Không bundle RMBG-2.0 cho POD thương mại khi license/checkpoint không cho phép.

## Metric release bắt buộc

- SAD, MSE, Gradient và Connectivity trên alpha 16-bit.
- Boundary F1 ±2/±4 px; recall component ≥4 px; hole IoU ở lỗ 3/7/15/25/45 px.
- Composite error trên nền trắng, đen, checker và ít nhất ba màu/texture gần màu vật thể.
- Báo riêng opaque, transparent, translucent, specular, motion blur, tóc/dây, detached décor, 1/3/4+ vật thể và chi tiết ở đầu/bên/dưới.
- CPU FP32 là reference; FP16/quantized/DirectML phải lệch Boundary F1 không quá 0,5 điểm phần trăm.

PNG RGBA chỉ biểu diễn RGB + một alpha. Nó không thể tái tạo chính xác khúc xạ phụ thuộc nền. Nếu cần ly kính vật lý hơn, format nội bộ phải lưu thêm attenuation/tint/refractive flow; PNG xuất ra vẫn là approximation và không được quảng bá “pixel-perfect”.
