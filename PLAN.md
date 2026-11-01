# PLAN — các bước thực hiện

Đánh dấu `[x]` khi xong, kèm commit hoặc kết quả. Mốc ◆ phải đạt mới sang tuần sau.
TV1 = dữ liệu & thống kê · TV2 = RNN/CNN · TV3 = Transformer & demo.

## Tuần 1 (22–28/10) ◆ RQ và ma trận chốt trước mọi lần huấn luyện

Chi tiết từng việc và checkpoint CP0–CP2: [docs/WEEK1.md](docs/WEEK1.md).

- [x] TV1: tải Twitter FNS, PhraseBank, GloVe; kiểm tra giấy phép; EDA — [kết quả 01/11](results/TV1_WEEK1.md), hash và hình trong `results/`
- [ ] TV1: chốt hướng dẫn gán nhãn (`labeling/GUIDE.md`) — đã bổ sung quy tắc, 9 ví dụ FNS và ca khó ngày 01/11; chờ cả 3 người duyệt CP2
- [ ] TV2: xuất tiêu đề 2026 từ pipeline; đóng gói script khử trùng lặp
- [ ] TV2: tokenizer, nạp GloVe
- [ ] TV3: chạy thử DistilBERT nhỏ và Qwen 4-bit trên Colab
- [ ] Cả nhóm: kiểm tra khả thi — tỷ lệ trùng thực tế trong Twitter FNS (quyết định mức rò rỉ E10) — TV1 đã đo: 12,72% trong cụm trùng, trần thô 50,28%; chờ nhóm quyết định
- [ ] Cả nhóm: nộp proposal — đã rà nội dung; còn placeholder thông tin nhóm/MSSV và chưa có xác nhận nộp

## Tuần 2 (29/10–4/11) ◆ split chốt kèm hash; gán thử 30 tiêu đề

- [ ] TV1: phân cụm, kiểm tra 100 cặp, split + SHA-256; baseline E0
- [ ] TV2: vòng huấn luyện chung (seed, config, log, early stopping)
- [ ] TV3: prompt A/B/C, kiểm tra top-10 logits; khung đánh giá chung
- [ ] Cả nhóm: gán thử 30 tiêu đề, sửa hướng dẫn

## Tuần 3 (5–11/11)

- [ ] TV1: bootstrap, McNemar, mô phỏng mức chênh phát hiện được
- [ ] TV2: TextCNN (E1), BiLSTM-Attention (E2) trên validation
- [ ] TV3: DistilBERT (E5), đóng băng (E7)
- [ ] Cả nhóm: gán nhãn đợt 1 (~70 nhãn/người)

## Tuần 4 (12–18/11)

- [ ] TV1: tạo dữ liệu rò rỉ 10/25/50%
- [ ] TV2: ablation E3, E4 (3 seed)
- [ ] TV3: LLM nhãn mềm ~3k tiêu đề; E6, E12
- [ ] Cả nhóm: gán nhãn đợt 2 (~65 nhãn/người)

## Tuần 5 (19–25/11) ◆ kappa xong, nhãn test 2026 chốt

- [ ] TV1: kappa, phân xử, chốt nhãn; tổng hợp RQ1
- [ ] TV2: E9, E10 cho BiLSTM; E13
- [ ] TV3: E9, E10 cho DistilBERT; E8; E11

## Tuần 6 (26/11–2/12) ◆ chạy test một lần, kết quả cố định

- [ ] TV1: chạy ba tập test; bảng kết quả, CI, McNemar
- [ ] TV2: tạo bộ robustness; robustness BiLSTM
- [ ] TV3: robustness Transformer, LLM

## Tuần 7 (3–9/12)

- [ ] TV1: phân tích lỗi theo nhóm
- [ ] TV2: trực quan attention; FastAPI + Dockerfile
- [ ] TV3: xuất ONNX, đo latency CPU; demo Gradio

## Tuần 8 (10–16/12)

- [ ] TV1: chương 3, kết quả RQ1; ghép báo cáo
- [ ] TV2: chương 2 (RNN/CNN/attention), kết quả RQ2
- [ ] TV3: chương 2 (Transformer, chưng cất), kết quả RQ3, demo
- [ ] Cả nhóm: slide

## Dự phòng (17–20/12) ◆ nộp trước 20/12

- [ ] TV2: chạy lại toàn bộ từ README trên máy sạch
- [ ] TV3: rà soát slide và demo
- [ ] TV1: nộp bài trên Google Classroom
