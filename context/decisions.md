# Quyết định đã chốt

Mỗi dòng: ngày · quyết định · lý do. Không đổi lại quyết định cũ mà không thêm dòng mới giải thích.

| Ngày | Quyết định | Lý do |
| --- | --- | --- |
| 2026-10-22 | Đề tài: cảm xúc tiêu đề tin tài chính, 3 lớp, PyTorch | Giảng viên chấp nhận đề tài tự chọn, ưu tiên độ phức tạp |
| 2026-10-22 | Huấn luyện trên dữ liệu công khai (Twitter FNS), không tự gán nhãn tập train | Giảm khối lượng gán nhãn |
| 2026-10-22 | Không dùng PhraseBank để train; FinBERT không được đánh giá trên PhraseBank | FinBERT đã học PhraseBank → rò rỉ |
| 2026-10-22 | Tập test 2026: 300 tiêu đề, 100 gán đôi (~135 nhãn/người) | Giữ gán nhãn nhẹ; chấp nhận CI rộng và báo cáo mức chênh phát hiện được |
| 2026-10-22 | LLM không gán nhãn tập test; chỉ tạo nhãn cho huấn luyện | Test phải đo khớp với phán đoán con người |
| 2026-10-22 | RQ3 đổi thành "ngoài phân phối" (lệch kết hợp nguồn + thời gian) | Train tweet, test báo chí: không tách được nguồn và thời gian |
| 2026-10-22 | Bỏ DAPT; thay bằng chưng cất tri thức từ nhãn mềm LLM (E6) | Kho không nhãn vài nghìn tiêu đề quá nhỏ cho DAPT; chưng cất sâu hơn về DL |
| 2026-10-22 | Thêm focal loss (E13) vào RQ2 | Tăng độ sâu DL, xử lý mất cân bằng lớp |
| 2026-10-22 | Kết luận chính bằng CI 95% của chênh lệch (paired bootstrap); McNemar phụ | Test 2026 nhỏ, McNemar thiếu lực thống kê |
| 2026-10-22 | Ngưỡng cụm: Jaccard ≥ 0.75, cosine MiniLM ≥ 0.85, kiểm tra trên 100 cặp | Tách trùng từ vựng và trùng ngữ nghĩa cho RQ1 |
| 2026-10-22 | Xác suất LLM lấy từ logits token A/B/C (gộp biến thể có/không khoảng trắng) | Văn bản LLM sinh ra không phải phân phối thật |
| 2026-10-22 | Mỗi batch trộn FNS (CE) và 2026 (T²·KL); KLDivLoss nhận log-softmax học trò, softmax thầy | Không phải chạy LLM trên ~12k tweet; tránh sai gradient |
| 2026-10-22 | `<ticker>` chỉ cho mã có `$` hoặc danh sách mã lớn đã loại từ viết tắt thông dụng | Tránh biến AI, IT, US… thành mã |
| 2026-10-22 | Tuần 7: xuất ONNX (FP32, INT8) và đo latency CPU; FastAPI + Docker nếu kịp | Điểm thưởng hiệu năng, kỹ năng triển khai |
| 2026-10-22 | Thành viên 2 phụ trách RNN/CNN và vòng huấn luyện chung | Phần trọng tâm học phần |
| 2026-10-22 | Repo private đến khi chấm xong | Tránh nhóm khác sao chép |
| 2026-10-26 | Tokenizer: chữ thường (GloVe 6B không phân biệt hoa thường); `<url>`, `<num>`; `<ticker>` cho mọi mã có `$` và ~90 mã lớn viết hoa đứng riêng (`models/tokenizer.py`, `LARGE_CAPS`); bỏ MA, BA và mọi mã dính `&`/`.` (PG&E, GGAL.BA) | Kiểm tra trên FNS: MA = đường trung bình, BA = hậu tố sàn, PG khớp nhầm PG&E 16 lần; tiêu đề dài nhất 49 token nên `max_len: 64` đủ |
| 2026-10-27 | **ĐỀ XUẤT, chờ duyệt CP2:** mỗi người một nhánh mỗi tuần (`<ten>/week<N>`), một PR mỗi tuần; chỉ tách nhánh khi người khác cần một phần sớm. Thay cho "mỗi việc một nhánh" | Tuần 1 có 7 nhánh/PR cho một người, khó theo dõi và tốn lượt duyệt; đã gộp còn 4. Nhánh sống một tuần nên ít xung đột với `main` |
| 2026-11-02 | **ĐỀ XUẤT, chờ duyệt CP3:** thứ tự nhãn duy nhất `["bearish", "neutral", "bullish"]` (0, 1, 2); split chỉ chứa ID; mọi mô hình ghi dự đoán `{"id", "probs"}` theo thứ tự đó | FNS gốc dùng 0 = bearish, 1 = bullish, 2 = neutral và LLM dùng A/B/C khác thứ tự: một thứ tự chung tránh lỗi lệch nhãn khi so sánh mô hình |
| 2026-11-02 | **ĐỀ XUẤT, chờ duyệt CP3:** 30 tiêu đề gán thử bị loại khỏi 300 tiêu đề test 2026 | Nhóm sửa GUIDE dựa trên các tiêu đề này; giữ chúng trong test là chỉnh quy tắc theo chính tập test |
| 2026-11-02 | **ĐỀ XUẤT, chờ duyệt CP2 (bù):** dời mốc ◆ tuần 2 sang 07/11 | Việc tuần 1 bàn giao 01/11, CP2 chưa họp |
| 2026-11-10 | **ĐỀ XUẤT, chờ cả nhóm duyệt:** lịch bù giữ nguyên toàn bộ phạm vi (E0–E13, robustness, ONNX, Gradio, FastAPI + Docker), không cắt; bù bằng cách làm dày hơn (~10–12 giờ/tuần mỗi người, chạy GPU qua đêm, viết code trước khi có split/nhãn, viết chương song song). Mốc ◆ tuần 2 là 13/11, chạy ba tập test một lần 05–07/12, nộp mục tiêu 19/12 (hạn cứng 20/12). Thay đề xuất dời mốc sang 07/11 ghi ngày 02/11 | Việc tuần 1 bàn giao muộn và tuần 2 chưa xong đến 10/11; nhóm chọn tăng cường độ thay vì cắt phạm vi. Dự phòng cuối giảm từ 4 ngày còn 2 ngày; thứ tự cắt trong SPEC chỉ dùng khi trượt mốc ◆ quá 2 ngày |
| 2026-11-14 | Kiến trúc RNN/CNN: TextCNN (E1) theo Kim (2014), bộ lọc 3/4/5 × 100, ReLU, max-over-time; BiLSTM-Attention (E2) 2 lớp × 128 mỗi chiều, attention cộng v·tanh(W h) có mask theo độ dài thật; E3 thay attention bằng max-pooling trên cùng mô hình. Dropout 0,3 (embedding và trước lớp ra), weighted CE, Adam 1e-3, batch 64 giống nhau ở E1–E3 | Chỉ đổi một yếu tố giữa E1/E2/E3 (AGENTS.md quy tắc 4); mask để padding không làm đổi kết quả (đã có test). Chạy thử trên CPU: E1 ~6 s/epoch, E2 ~37 s/epoch, nên E1/E2 chạy được trên máy, không cần GPU |
