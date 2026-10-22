# SPEC — chúng ta muốn đạt được gì

Bản đầy đủ: [docs/KeHoachChiTiet.pdf](docs/KeHoachChiTiet.pdf). Thay đổi file này = thay đổi
thiết kế → ghi lý do vào [context/decisions.md](context/decisions.md).

## Bài toán

Input: một tiêu đề/câu tin tài chính tiếng Anh. Output: bearish / neutral / bullish + xác suất.
Metric chính: Macro-F1. Kết luận bằng CI 95% của chênh lệch (paired bootstrap, 1.000 lần);
McNemar bổ sung; báo cáo mức chênh nhỏ nhất phát hiện được với test 2026.

## Câu hỏi nghiên cứu

| RQ | Giả thuyết | Bác bỏ khi |
| --- | --- | --- |
| RQ1 rò rỉ | Macro-F1 tăng đơn điệu theo tỷ lệ bản gần trùng của test nằm trong train | Đường phẳng hoặc CI của chênh 0% → 50% chứa 0 |
| RQ2 kiến trúc | Bỏ attention hoặc GloVe làm giảm Macro-F1; focal loss tăng F1 lớp thiểu số | CI của chênh so với E2 chứa 0 |
| RQ3 ngoài phân phối | Mô hình học trên tweet suy giảm trên tiêu đề báo 2026; nhãn mềm (chưng cất) giúp hơn nhãn cứng | CI suy giảm chứa 0, hoặc E6 không hơn E12 |

RQ3 đo lệch **kết hợp** nguồn/văn phong và thời gian; không kết luận riêng về thời gian.

## Dữ liệu

| Tập | Vai trò |
| --- | --- |
| Twitter Financial News Sentiment (~11,9k) | train/val/test trong miền, chia 70/15/15 theo cụm |
| Tiêu đề 2026 nhóm gán (300; 100 gán đôi, kappa ≥ 0.6) | test ngoài phân phối |
| Tiêu đề 2026 không nhãn (~3k) | nhãn mềm/cứng từ LLM, chỉ để train |
| Financial PhraseBank AllAgree (~2,3k) | test ngoài miền (không đánh giá FinBERT) |

Cụm gần trùng: Jaccard shingle từ ≥ 0.75 (MinHash LSH) hoặc cosine `all-MiniLM-L6-v2` ≥ 0.85.
`<ticker>` chỉ cho mã có `$` hoặc thuộc danh sách mã lớn (đã loại AI, IT, US, CEO, GDP, ON, ALL).

## Ma trận thí nghiệm

| ID | Thay đổi duy nhất | Mốc | RQ | Seed |
| --- | --- | --- | --- | --- |
| E0 | TF-IDF + LR | — | tham chiếu | 1 |
| E1 | TextCNN | E2 | RQ2 | 3 |
| E2 | BiLSTM-Attention đầy đủ (GloVe, attention, weighted CE) | — | RQ2 | 3 |
| E3 | Attention → max-pooling | E2 | RQ2 | 3 |
| E4 | GloVe → embedding ngẫu nhiên | E2 | RQ2 | 3 |
| E13 | Weighted CE → focal loss (γ = 2) | E2 | RQ2 | 3 |
| E5 | DistilBERT fine-tune | — | RQ3 | 3 |
| E6 | + chưng cất nhãn mềm LLM (α, T chọn trên val) | E5, E12 | RQ3 | 3 |
| E7 | Encoder đóng băng | E5 | RQ3 | 3 |
| E8 | FinBERT | E5 | RQ3 | 3 |
| E9 | Split ngẫu nhiên (E2, E5) | E2, E5 | RQ1 | 3 |
| E10 | Rò rỉ 10/25/50% (E2, E5) | E2, E5 | RQ1 | 3 |
| E11 | LLM zero-shot (Qwen2.5-7B 4-bit) | — | tham chiếu | 1 |
| E12 | + nhãn cứng LLM | E5 | RQ3 | 3 |

Thứ tự cắt khi trễ: E13 → E12 → FastAPI/Docker. Luôn giữ E0–E4, E9, E10 và test 2026.

## Hoàn thành khi

- Mỗi RQ có câu trả lời bằng số liệu (CI 95%), kể cả khi bác bỏ giả thuyết.
- README tái lập được toàn bộ bảng kết quả trên máy sạch.
- Nộp: báo cáo 5 chương, slide, code, demo Gradio, bảng latency ONNX, bảng đóng góp.
