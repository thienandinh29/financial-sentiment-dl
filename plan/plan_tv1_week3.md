# Checklist tuần 3 — Thành viên 1

Nguồn: [việc chi tiết tuần 3](../docs/WEEK3.md) và [lịch bù](../PLAN.md). TV1 phụ trách dữ liệu và thống kê. CP5a là 18/11, CP5 là 20/11; việc E10 tiếp tục trong S2 (21–27/11).

## 1.13 — Chọn và phân công 300 tiêu đề 2026 (CP5a)

- [x] Chọn ngẫu nhiên có seed 300 ID từ kho RSS `test_candidate`, loại 30 ID trong `labeling/trial30_ids.csv`.
- [x] Lưu ID và URL vào `labeling/test300_ids.csv`; lưu ID và người gán vào `labeling/assignment.csv`.
- [x] Phân công 100 tiêu đề gán đôi, chia đều giữa ba cặp; mỗi người khoảng 135 nhãn, chia hai đợt khoảng 70 và 65.
- [ ] Chia sẻ danh sách và phân công cho nhóm để bắt đầu gán nhãn. Bạn đã nhận làm việc này; chưa xác nhận đã gửi.

## 1.12 — Baseline E0 (CP5)

- [x] Kiểm tra SHA-256 của split đã chốt trước khi chạy; fit TF-IDF chỉ trên train.
- [x] Chạy E0 trên validation; lưu `results/E0/seed1/run.json` có Macro-F1 và `preds_validation.jsonl` đúng định dạng chung.
- [x] Ghi commit, config, seed, phiên bản thư viện, GPU và thời gian theo quy tắc dự án; báo kết quả cho bảng RQ2 validation.

## 1.14 — Bootstrap và McNemar (CP5)

- [x] Viết `evaluate/stats.py`: paired bootstrap 1.000 lần, tính CI 95% của chênh lệch Macro-F1 giữa hai file dự đoán.
- [x] Tính McNemar trên các dự đoán cùng ID; bổ sung test với ví dụ tính tay và dữ liệu giả.

## 1.15 — Mô phỏng mức chênh phát hiện được (CP5)

- [x] Mô phỏng cho cỡ mẫu 1.791 và 300, xác định mức chênh Macro-F1 nhỏ nhất có CI 95% loại được 0 trong các điều kiện mô phỏng.
- [x] Lưu bảng, phương pháp và giả định vào `results/power/`.

## 1.16 — Dữ liệu rò rỉ E10 (S2, 21–27/11)

**Trạng thái: chờ nhóm thảo luận thiết kế và mức rò rỉ; bạn sẽ chuyển việc chốt mức cho TV2.** Bạn chọn dùng validation để nghiên cứu rò rỉ, giữ nguyên test; đây là hướng xử lý, chưa phải quyết định cuối của cả nhóm về thiết kế và các mức. Với split cố định, validation có trần lý thuyết 76/1.791 (4,24%) theo cách lấy ít nhất một ID mỗi cụm làm nguồn và chỉ tính các ID còn lại; trần này không chứng minh có thể thực hiện, và việc dùng ID validation để train cần làm rõ cách giữ tập đánh giá hợp lệ. Ba mức 5/10/20% hiện chưa được xác nhận khả thi. Không tự đổi mức, split hoặc dựng dữ liệu trước khi nhóm chốt; xem `results/feasibility/E10_frozen_split.md`.

- [ ] Hoàn thiện `scripts/make_leakage.py` cho thiết kế và mức nhóm chốt, tích hợp huấn luyện, xác nhận tính khả thi và báo nhóm nếu không đạt. Đã chuẩn bị script tạo manifest ID từ donor chỉ định và test giả; chưa chạy dữ liệu thật.
- [ ] Chỉ thay đổi train; giữ nguyên validation và test, không tạo hay sửa split đã chốt.

## Gán nhãn và kiểm tra chung

- [ ] Đính chính theo quyết định 23/11: nhãn đợt 1 TV1 do LLM tạo, không phải nhãn thủ công và chưa từng commit. Trạng thái thực hiện thủ công đợt 2 cần TV1 xác nhận; không suy đoán từ ghi chép cũ.
- [ ] Chỉ đánh giá validation trong giai đoạn này; ba tập test chỉ chạy một lần ở tuần 6 sau khi mọi lựa chọn đã chốt.
- [ ] Không commit dữ liệu thô, văn bản tiêu đề 2026, checkpoint hoặc file ONNX; giữ secrets trong `.env`.
- [ ] Cập nhật `MAP.md` khi tạo hoặc di chuyển file, bổ sung test cho thay đổi hành vi và cập nhật tiến độ trong `PLAN.md`.

TV1 xác nhận tự gán cả 134 nhãn; mã người gán ban đầu ghi nhầm và đã được sửa thành `tv1` theo xác nhận. Có 23 dòng có ghi chú `issue`, cần nhóm xử lý theo GUIDE; file nhãn gốc vẫn ở `data/raw/headlines_2026/labeling_packets/`.

E10 đã có khung `scripts/make_leakage.py`: kiểm tra hash, donor phải thuộc validation, tập đo loại donor, chỉ đếm cặp trực tiếp và từ chối nếu số ID phơi nhiễm khác yêu cầu. Manifest chưa được vòng huấn luyện đọc; chưa tạo dữ liệu thật. Nhóm cần chốt donor, mẫu số, cách so sánh trên cùng tập đo và mức trước khi dùng.
