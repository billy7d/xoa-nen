# Segmentation engine v2

## Mục tiêu và phạm vi

Engine v2 cải thiện hai đường xử lý classical/offline: đề xuất alpha ARTWORK tự động và Magic Wand. Thiết kế giữ nguyên các contract của project: source alpha là upper bound, RGB canonical không đổi, edit có undo/redo và vùng mơ hồ được giữ lại thay vì tự xóa quá tay.

Đây không phải semantic segmentation. Khi foreground và background thực sự không có khác biệt màu, texture, cạnh hoặc topology quan sát được, classical vision không thể suy ra đúng vật thể. Những ca hair/fur/glass hoặc cần hiểu ngữ nghĩa vẫn phải đi qua model stack sau Phase 0 theo PRD.

## Cơ sở kỹ thuật đã chọn

- **CIE Lab và CIEDE2000:** khoảng cách Euclid trong sRGB cũ không đồng đều theo cảm nhận. QUALITY và Wand dùng CIEDE2000; FAST dùng Delta E 76 để giảm latency. Formula được kiểm tra bằng reference pair `2.0425` của Sharma, Wu và Dalal.
- **Background palette đa mode:** lấy mẫu dải viền, robust k-medians tối đa 3/4/5 tâm theo FAST/QUALITY/MAX. Nền gradient hoặc nhiều mảng màu không còn bị ép thành một RGB median duy nhất.
- **Region growing có edge barrier:** màu hợp lệ chỉ được flood qua khi gradient Lab cục bộ không vượt ngưỡng noise-adaptive. Điều này kết hợp region evidence và boundary evidence, thay vì chỉ threshold màu.
- **Bảo vệ topology bảo thủ:** sure-foreground được close một pixel, fill vùng khép kín và dilate nhẹ. Chi tiết/lỗ đồng màu nền hoặc khe anti-alias nhỏ không bị flood tự động phá hủy.
- **Edge-aware alpha refinement:** QUALITY/MAX tạo alpha mềm trên proxy bằng guided filter rồi hợp nhất với phép đo màu native-resolution. Pixel source alpha vẫn luôn là upper bound.
- **Wand native confirmation:** màu seed là median của patch nhỏ có noise allowance. Contiguous membership được tìm trên proxy có edge barrier, sau đó flood lại trên evidence màu native-resolution. Global mode cố ý chọn mọi pixel cùng range và không áp topology barrier.
- **POD edge decontamination đa nền:** mỗi pixel alpha bán trong suốt dùng màu gần nhất trong background palette, tránh halo sai màu khi nền có nhiều mode.

Các nguồn nền tảng:

- G. Sharma, W. Wu, E. Dalal, [The CIEDE2000 Color-Difference Formula](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/).
- K. He, J. Sun, X. Tang, [Guided Image Filtering](https://doi.org/10.1109/TPAMI.2012.213).
- C. Rother, V. Kolmogorov, A. Blake, [GrabCut — Interactive Foreground Extraction using Iterated Graph Cuts](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf). Engine v2 áp dụng nguyên lý kết hợp region/boundary và trimap bảo thủ, không nhúng GrabCut/OpenCV để giữ sidecar gọn.
- A. Levin, D. Lischinski, Y. Weiss, [A Closed-Form Solution to Natural Image Matting](https://people.csail.mit.edu/alevin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf). Đây là cơ sở cho việc giữ fractional alpha thay vì hard-fill toàn bộ foreground.

## Profile thực thi

| Profile | Proxy tối đa | Color metric | Edge-aware refine | Native refine |
| --- | ---: | --- | --- | --- |
| FAST | 768 px | Delta E 76 | Không | Có |
| QUALITY | 1536 px | CIEDE2000 | Có, radius 5 | Có |
| MAX | 2048 px | CIEDE2000 | Có, radius 7 | Có |

UI hiện expose FAST và QUALITY. Worker nhận và ghi đúng profile vào diagnostics; không còn tình trạng dropdown chỉ đổi metadata.

## Diagnostics và regression fixtures

Manifest processing ghi engine, metric, palette Lab và tỷ trọng, Delta E threshold, edge-stop threshold, proxy edge, background/protected/uncertain fractions và cờ native/edge-aware refinement.

Regression tests bao phủ:

- CIEDE2000 reference pair;
- source-alpha upper bound và component rời;
- nền gradient nhiều tông + foreground low-contrast nằm trong color tolerance;
- foreground chạm một cạnh không làm nhiễm background palette;
- lỗ đồng màu nền với khe hở anti-alias nhỏ;
- seed patch của Wand giữ đúng phía pixel được click sát boundary;
- Wand contiguous chặn tại weak edge trong khi global vẫn chọn theo color range;
- fractional coverage tại cạnh anti-aliased;
- POD edge decontamination chọn đúng background mode gần nhất;
- coordinator process/edit/undo/redo/export hiện hữu.

## Giới hạn có chủ ý

- Edge barrier ưu tiên tránh false removal, vì vậy background texture rất mạnh có thể cần click Wand bổ sung hoặc Remove Brush.
- Global Wand không bảo vệ vật thể cùng màu; đây là semantics mong đợi của global color selection. Bật contiguous để có edge/topology protection.
- Không tự tải model hoặc chạy code từ model repository. BiRefNet/SAM2/ViTMatte vẫn phải qua revision/checksum/license/hardware-quality gate của `PHASE0_QUALIFICATION.md`.
