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
