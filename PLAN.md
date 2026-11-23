# PLAN — các bước thực hiện

Đánh dấu `[x]` khi xong, kèm commit hoặc kết quả. Mốc ◆ phải đạt mới sang tuần sau.
TV1 = dữ liệu & thống kê · TV2 = RNN/CNN · TV3 = Transformer & demo.

## Lịch bù (ĐỀ XUẤT 10/11/2026, chờ cả nhóm duyệt) — giữ nguyên phạm vi, làm dày hơn

Việc tuần 1 bàn giao muộn (01/11) và tuần 2 chưa xong, nên các ngày ghi ở từng tuần bên dưới không còn
đúng. **Không cắt phạm vi**: E0–E13, robustness, ONNX, Gradio, FastAPI + Docker vẫn giữ. Bù bằng cách
làm dày hơn: ~10–12 giờ/tuần mỗi người (chưa kể gán nhãn), chạy GPU qua đêm trên Colab/Kaggle, viết code
trước khi có split/nhãn, viết các chương báo cáo song song với thí nghiệm. Nộp **mục tiêu Thứ Bảy 19/12**,
hạn cứng 20/12 (dự phòng cuối còn 2 ngày thay vì 4).

| Giai đoạn | Ngày | Việc (tương ứng tuần trong PLAN) | Mốc ◆ |
| --- | --- | --- | --- |
| S0 | Thứ Ba 10/11 – Thứ Sáu 13/11 | Xong tuần 2: gộp PR, **chốt split + SHA-256**, E0, khung đánh giá, prompt A/B/C, gán thử 30 + kappa + sửa GUIDE (ngoài split, phần còn lại xong trước Chủ Nhật 15/11). TV2 viết sẵn TextCNN và BiLSTM-Attention | Thứ Sáu 13/11 tối: split chốt trên `main` |
| S1 | Thứ Bảy 14/11 – Thứ Sáu 20/11 | Tuần 3 và đầu tuần 4: E1, E2 (TV2); E5, E7 và bắt đầu nhãn mềm LLM chạy qua đêm (TV3); bootstrap, McNemar, mô phỏng mức chênh, dữ liệu rò rỉ 5/10/20% (TV1); gán nhãn đợt 1 (~70/người) | Thứ Sáu 20/11: E1, E2, E5, E7 trên validation; nhãn đợt 1 xong |
| S2 | Thứ Bảy 21/11 – Thứ Sáu 27/11 | Cuối tuần 4: E3, E4 (TV2); nhãn mềm xong, E6, E12 (TV3); gán nhãn đợt 2 (~65/người) | Thứ Sáu 27/11: ablation, E6, E12 trên validation; nhãn đợt 2 xong |
| S3 | Thứ Bảy 28/11 – Thứ Sáu 04/12 | Tuần 5: kappa, phân xử, **chốt nhãn test 2026** (TV1); E9, E10 cho BiLSTM và E13 (TV2); E9, E10 cho DistilBERT, E8, E11 (TV3). Viết sẵn code robustness, ONNX, Gradio, FastAPI trên validation | **Thứ Sáu 04/12: mọi lựa chọn đã chốt, nhãn test đã khóa** |
| S4 | Thứ Bảy 05/12 – Thứ Hai 07/12 | Tuần 6: **chạy ba tập test đúng một lần**, bảng kết quả, CI 95%, McNemar (TV1) | Thứ Hai 07/12: kết quả test cố định |
| S5 | Thứ Ba 08/12 – Chủ Nhật 13/12 | Tuần 6 còn lại và tuần 7: robustness (TV2, TV3); phân tích lỗi (TV1); trực quan attention, FastAPI + Dockerfile (TV2); ONNX, đo latency CPU, demo Gradio (TV3) | Chủ Nhật 13/12: mọi bảng, hình, demo đã xong |
| S6 | Thứ Hai 14/12 – Thứ Năm 17/12 | Tuần 8: hoàn thiện báo cáo 5 chương (phần phương pháp viết sẵn từ S2), kết quả RQ1–RQ3, slide | Thứ Năm 17/12: báo cáo và slide bản đầy đủ |
| Dự phòng | Thứ Sáu 18/12 – Thứ Bảy 19/12 | TV2 chạy lại từ README trên máy sạch; TV3 rà soát slide và demo; TV1 nộp bài trên Google Classroom | **Nộp Thứ Bảy 19/12** (hạn cứng 20/12) |

Phần của con người không nén được: gán nhãn (~135/người) theo hai đợt 20/11 và 27/11, kappa và phân xử
trước 04/12. Nếu trượt mốc ◆ quá 2 ngày thì họp ngay và áp dụng thứ tự cắt đã có trong `SPEC.md`
(E13 → E12 → FastAPI/Docker); chưa cắt gì từ bây giờ.

## Tuần 1 (22–28/10) ◆ RQ và ma trận chốt trước mọi lần huấn luyện

Chi tiết từng việc và checkpoint CP0–CP2: [docs/WEEK1.md](docs/WEEK1.md).

- [x] TV1: tải Twitter FNS, PhraseBank, GloVe; kiểm tra giấy phép; EDA — [kết quả 01/11](results/TV1_WEEK1.md), hash và hình trong `results/`
- [ ] TV1: chốt hướng dẫn gán nhãn (`labeling/GUIDE.md`) — đã bổ sung quy tắc, 9 ví dụ FNS và ca khó ngày 01/11; chờ cả 3 người duyệt CP2
- [ ] TV2: xuất tiêu đề 2026 từ pipeline; đóng gói script khử trùng lặp
- [ ] TV2: tokenizer, nạp GloVe
- [x] TV3: chạy thử DistilBERT nhỏ và Qwen 4-bit trên Colab — `notebooks/colab_setup.ipynb`, log và ngân sách GPU trong `results/smoke/` (`bc36845`)
- [ ] Cả nhóm: kiểm tra khả thi — tỷ lệ trùng thực tế trong Twitter FNS (quyết định mức rò rỉ E10) — TV1 đã đo: 12,72% trong cụm trùng, trần thô 50,28%; chờ nhóm quyết định
- [ ] Cả nhóm: nộp proposal — đã rà nội dung; còn placeholder thông tin nhóm/MSSV và chưa có xác nhận nộp

## Tuần 2 (29/10–4/11) ◆ split chốt kèm hash; gán thử 30 tiêu đề

Chi tiết, hợp đồng chung và checkpoint CP2 (bù)–CP4: [docs/WEEK2.md](docs/WEEK2.md) (đề xuất dời mốc ◆ sang 07/11).

- [ ] TV1: phân cụm, kiểm tra 100 cặp, split + SHA-256; baseline E0
- [ ] TV2: vòng huấn luyện chung (seed, config, log, early stopping)
- [x] TV3: prompt A/B/C, kiểm tra top-10 logits; khung đánh giá chung — `evaluate/metrics.py`, `tests/test_metrics.py`, `configs/llm_prompts.yaml`, `notebooks/llm_prompts.ipynb`, `results/E11/{P1,P2,P3}/preds_validation.jsonl`, `results/llm_prompts/top10_check.json` (chốt `P2` với validation `macro_f1 = 0.7934`)
- [ ] Cả nhóm: gán thử 30 tiêu đề, sửa hướng dẫn — bỏ đánh dấu 23/11: nhãn gán thử do LLM tạo, chuyển sang `labeling/llm_annotations/`

- [x] TV1: selected 30 trial IDs (seed 2026); see `labeling/trial30_ids.csv`.
- [ ] TV1: share the IDs and exclude them when selecting the final 2026 test set.

## Tuần 3 (5–11/11)

Chi tiết theo lịch bù (S1, đến 20/11): [docs/WEEK3.md](docs/WEEK3.md).

- [ ] TV1: bootstrap, McNemar, mô phỏng mức chênh phát hiện được
- [x] TV2: TextCNN (E1), BiLSTM-Attention (E2) trên validation — Macro-F1 validation E1 0,755 ± 0,005, E2 0,758 ± 0,010 (3 seed; `results/rq2_validation.md`)
- [ ] TV3: DistilBERT (E5), đóng băng (E7)
- [ ] Cả nhóm: gán nhãn đợt 1 (~70 nhãn/người) — 23/11: nhãn của TV1, TV3 do LLM tạo, không dùng; TV2 gán tay toàn bộ 300 tiêu đề vào `labeling/gold/`

## Tuần 4 (12–18/11)

- [ ] TV1: tạo dữ liệu rò rỉ 5/10/20%
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

### Tiến độ TV1 S1/S2 (22/11)

- E0 validation: Macro-F1 0,7096, seed 1; log ghi cây làm việc có thay đổi.
- Bootstrap paired và McNemar exact đã có code và test; chưa đánh giá test.
- Đã chọn 300 ID RSS seed 2026 và phân công hai đợt; chờ gửi nhóm và từng người tự gán nhãn.
- E10: bạn chọn nghiên cứu trên validation, giữ test nguyên vẹn; chờ nhóm thống nhất thiết kế và mức. Trần k−1 validation 76/1.791 (4,24%) theo giả định trong `results/feasibility/E10_frozen_split.md`, chưa chứng minh 5/10/20% khả thi.

- TV1 xác nhận đã tự gán 69 + 65 tiêu đề; 23 dòng có ghi chú `issue` cần nhóm xem xét.
