# Kế hoạch tuần 4 — Thành viên 1

TV1 phụ trách dữ liệu và thống kê. Theo lịch bù **đề xuất** trong [PLAN.md](../PLAN.md), giai đoạn S2 là 21–27/11/2026; chi tiết và mốc CP6 ở [WEEK4.md](../docs/WEEK4.md). Đánh dấu hoàn thành khi có bằng chứng hoặc đường dẫn kết quả.

## 1.16 — Chuẩn bị dữ liệu rò rỉ E10 (CP6: 27/11)

- [x] Kiểm tra SHA-256 split cố định trước khi xử lý; dùng thiết kế Phương án A đã chốt ngày 23/11 trong [decisions.md](../context/decisions.md).
- [x] Hoàn thiện `scripts/make_leakage.py` và test: giữ train gốc, thêm donor từ validation; giữ nguyên 1.791 ID validation để đánh giá và không thay đổi split gốc hay tập test.
- [x] Với seed cố định, chọn donor ưu tiên cụm có ít nhất hai ID validation, sau đó dùng bản sao chính xác cho ID đơn lẻ; bảo đảm ba mức lồng nhau 5% ⊂ 10% ⊂ 20%.
- [x] Kiểm tra số ID validation bị lộ là **90 / 180 / 359**; tính phơi nhiễm khi chính ID hoặc bản gần trùng trực tiếp theo cosine 0,92 nằm trong train. Tách thống kê bản sao chính xác và bản gần trùng.
- [x] Tạo manifest chỉ chứa ID cho từng mức và SHA-256, kiểm tra khả năng tái lập; hướng dẫn bàn giao tại `results/E10/README.md` (chưa xác nhận nhóm đã nhận).
- [x] Chuẩn bị cách báo cáo Macro-F1 trên toàn validation, nhóm bị lộ và nhóm sạch, so ghép cặp với E2/E5 sạch; không dùng test: `results/E10/README.md`.

## 1.17 — Chạy lại baseline E0 (CP6)

- [x] Sau khi commit các thay đổi cần thiết, chạy E0 từ commit sạch; xác minh hash split và chỉ fit TF-IDF trên train.
- [x] Lưu kết quả validation và metadata tại `results/E0/seed1/`: commit, config, seed, phiên bản thư viện, GPU, thời gian và `working_tree_dirty: false`. Metadata cũ giữ tại `run_previous_20261122.json`; dự đoán và Macro-F1 không đổi.

## 1.18 — Sửa ghi chép gán nhãn (CP6a)

- [x] Rà soát `plan/plan_tv1_week2.md` và ghi đúng rằng nhãn gán thử do LLM tạo, không phải gold; dẫn tới `labeling/llm_annotations/` và quyết định ngày 23/11.
- [x] Đối chiếu ghi chép TV1 tuần 2–3 và đính chính; trạng thái thảo luận GUIDE và gán tay đợt 2 cần nhóm xác nhận.

## 1.19 — Gán đôi và phối hợp gold (CP6)

- [x] Kiểm tra bộ nhãn chung theo yêu cầu mới: CSV có 100 ID hợp lệ, mỗi ID có một nhãn. Bản cục bộ `double100_group_labels.csv` lưu ID và nhãn đã chuẩn hóa chữ thường. Đối chiếu ID với các file gold trong repo cho thấy tổng độ phủ 135/300 ID, trong đó 33 ID xuất hiện ở cả hai nguồn. Độ phủ này vẫn dưới mốc 180 và chưa phải gold đã khóa. Chưa xác minh hai lượt gán độc lập trên cùng ID nên chưa báo cáo kappa.
- [x] Nhập bộ nhãn chung đã được nhóm duyệt vào `labeling/gold/group_approved_week4.csv`. Giữ gold cũ cho cả 33 ID trùng, gồm 14 ID khác nhãn, và thêm 112 ID mới.
- [x] Lập danh sách 165 ID chưa có nhãn trong hai nguồn tại `data/raw/headlines_2026/labeling_packets/group_pending_ids.txt`. Lấy 45 dòng đầu cho lượt gán tay để đạt độ phủ 180/300.
- [x] Chuẩn bị gói cục bộ `group_next45_read.csv` và `group_next45_blank.csv`. Hai file khớp 45 ID theo thứ tự và không trùng 135 ID đã có nhãn. Mẫu điền có cột `id,label,reason,issue`.
- [x] Nhận và kiểm tra đủ 45 dòng mới có nhãn hợp lệ và lý do. Gộp theo ID vào `group145_labels.csv` cục bộ. Kết hợp với ID gold trong repo, độ phủ là 180/300.
- [x] Ghi nhận nhóm đã duyệt nhãn. Giữ lý do và ghi chú ở file nguồn cục bộ; file nhập gold chỉ có ID và nhãn để tránh đưa văn bản tiêu đề vào repo.
- [x] Kiểm tra trùng ID giữa hai nguồn trước khi nhập: 14 trong 33 ID trùng có nhãn khác nhau. Nhóm đã duyệt giữ nhãn gold cũ cho cả 14 ID.
- [x] Xác minh mốc gold 180/300 theo phương án bộ nhãn chung đã được nhóm duyệt.
- [x] Ghi giới hạn: chưa xác minh được hai lượt gán độc lập trên cùng ID để tính kappa. Khóa gold thuộc tuần 5.

## 1.20 — Nháp chương 3 (CP6)

- [x] Viết `report/ch3_data.md`: nguồn dữ liệu, khử trùng lặp, split cụm, kiểm tra hash, thiết kế E10 và giới hạn của bản sao chính xác.
- [x] Mô tả quy trình gán nhãn và sự cố nhãn LLM trung thực; phân biệt nhãn gold của người với nhãn LLM, không đưa văn bản tiêu đề 2026 vào báo cáo.

## Kiểm tra trước bàn giao

- [x] Các lần chạy mô hình của TV1 chỉ dùng validation; ba tập test chỉ đánh giá một lần ở tuần 6 sau khi mọi lựa chọn và nhãn đã khóa.
- [x] Rà danh sách file bàn giao: chỉ đưa vào Git mã nguồn, tài liệu, kết quả validation, manifest và dữ liệu ID/URL hoặc ID/nhãn. Đã thêm kiểm thử và cập nhật `MAP.md`.
- [x] Cập nhật tiến độ trong `PLAN.md` và tập hợp đường dẫn E10, E0, chương 3 tại [mục bàn giao kỹ thuật trong WEEK4.md](../docs/WEEK4.md#bàn-giao-kỹ-thuật-tv1).
- [ ] TV2/TV3 xác nhận đã nhận manifest, kết quả E0 và bản nháp chương 3 khi duyệt PR.

## Chốt tuần 4 của TV1

Mục 1.16–1.20 đã có đầu ra. Gold đạt 180/300 ID theo phương án nhóm duyệt. Tuần 5, TV2 điều phối gán tay 120 ID còn lại và khóa gold. Nhóm chỉ tính kappa khi xác minh được hai lượt gán tay độc lập trên cùng ID.

TV2 và TV3 tích hợp manifest E10 vào loader huấn luyện cho E2 và E5, giữ đủ 1.791 ID validation và ghép cặp theo ID, seed với baseline tương ứng. Dùng validation cho E10. Ba tập test chờ tuần 6 sau khi khóa cấu hình và nhãn.
