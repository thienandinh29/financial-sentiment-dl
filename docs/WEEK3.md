# Tuần 3 (giai đoạn S1 của lịch bù) — việc chi tiết và checkpoint

Theo **lịch bù** trong `PLAN.md`: S1 kéo dài đến **Thứ Sáu 20/11**, gồm việc tuần 3 và đầu tuần 4.
Mốc ◆ S1: **E1, E2, E5, E7 có kết quả trên validation (3 seed); gán nhãn đợt 1 xong.**
TV1 = dữ liệu & thống kê · TV2 = RNN/CNN · TV3 = Transformer & demo.
Nhánh: mỗi người một nhánh cho tuần này (`<ten>/week3`), một PR.

Quy tắc nhắc lại: chỉ chấm **validation**; không ai chạy test trước tuần 6. Mọi lần chạy đọc split
`data/splits/cluster_split.jsonl` (SHA-256 `8c2c35ea…`) và ghi `run.json` + `preds_validation.jsonl`
đúng định dạng chung. Lý do gán nhãn viết **bằng lời của mình**, không chép tiêu đề, không chép của người
khác, không dùng AI.

## Lịch checkpoint

| CP | Thời điểm | Phải có |
| --- | --- | --- |
| CP5a | **Thứ Tư 18/11**, tối, 30 phút | PR tuần 1–2 đã duyệt và gộp vào `main` (thứ tự dưới); 300 ID tiêu đề test đã chọn và phân công gán; mức E10 mới đã chốt; `SPEC.md` sửa cosine 0,92 |
| CP5 ◆ | **Thứ Sáu 20/11**, tối | E0, E1, E2, E5, E7 trên validation; gán nhãn đợt 1 xong (~70/người); code bootstrap + McNemar có test |

## Việc chung trước tiên (hạn CP5a)

1. **Gộp PR** theo thứ tự đã chạy thử (mọi xung đột chỉ là dòng bảng, giữ cả hai bên):
   #7 → #6 → #2 → #5 → #9 → #10 → #12 → #8 → #13 → #11. Mỗi PR một người khác duyệt.
   PR #2 dùng **"Squash and merge"** (lịch sử cũ của nhánh có tiêu đề 2026 thật trong test).
2. **Chốt mức rò rỉ E10.** Với ngưỡng 0,92, trần thô chỉ còn ≈ 27% (488/1.791 tweet test), nên 50% không
   làm được. Đề xuất **5 / 10 / 20%**; ghi vào `context/decisions.md` và `SPEC.md`.
3. **`SPEC.md` dòng 31:** cosine `0.85` → `0.92` (đã thống nhất ở CP3).
4. Sau khi #10 vào `main`: điền `split_sha256` cho `configs/E0.yaml`; bỏ chữ "proposed" ở `configs/E2.yaml`.

## TV1 — dữ liệu & thống kê

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 1.12 | Chạy E0 trên split đã chốt | `results/E0/seed1/{run.json,preds_validation.jsonl}` | Macro-F1 validation ghi trong `run.json`; TF-IDF fit chỉ trên train | CP5 |
| 1.13 | Chọn 300 tiêu đề test 2026 và phân công | `labeling/test300_ids.csv` (id, url), `labeling/assignment.csv` (id, người gán) | Lấy ngẫu nhiên có seed từ kho RSS `test_candidate`, **loại 30 ID gán thử**; 100 tiêu đề gán đôi, chia đều 3 cặp người; mỗi người ~135 nhãn, chia 2 đợt (~70 rồi ~65) | CP5a |
| 1.14 | Bootstrap + McNemar | `evaluate/stats.py` + test | Paired bootstrap 1.000 lần → CI 95% của chênh Macro-F1 giữa 2 file dự đoán; McNemar trên cùng ID; test với ví dụ tính tay | CP5 |
| 1.15 | Mô phỏng mức chênh phát hiện được | `results/power/` (bảng + mô tả) | Với cỡ test 1.791 và 300: chênh Macro-F1 nhỏ nhất mà CI 95% loại được 0 | CP5 |
| 1.16 | Dữ liệu rò rỉ E10 (bắt đầu) | `scripts/make_leakage.py` + test | Theo mức đã chốt ở CP5a; chỉ đổi train, không đổi validation/test | S2 (21–27/11) |

## TV2 — RNN/CNN

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 2.11 | GloVe 6B 100d + điền hash split | `data/raw/glove/` (không commit), `configs/E1.yaml` | SHA-256 GloVe khớp báo cáo TV1; E1/E2 cùng hash split | ✅ 17/11 |
| 2.12 | E1 TextCNN, 3 seed | `results/E1/seed{1,2,3}/` | Mỗi seed có `run.json`, `preds_validation.jsonl`; chạy từ commit sạch | CP5 |
| 2.13 | E2 BiLSTM-Attention, 3 seed | `results/E2/seed{1,2,3}/` | Như trên | CP5 |
| 2.14 | Bảng RQ2 tạm (validation) | `results/rq2_validation.md` | Trung bình ± độ lệch chuẩn qua 3 seed cho E1, E2 (và E0 khi TV1 có) | CP5 |
| 2.15 | Chuẩn bị E3, E4 | `configs/E3.yaml`, `configs/E4.yaml` | Mỗi file copy E2, đổi đúng một dòng (pooling `max`; embedding `random`) | S2 |
| 2.16 | Gán nhãn đợt 1 | `labeling/round1/tv2.csv` | ~70 nhãn, tự gán, không AI | CP5 |

## TV3 — Transformer & LLM

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 3.9 | E5 DistilBERT fine-tune, 3 seed | `configs/E5.yaml`, `results/E5/seed{1,2,3}/` | Đọc split bằng `train/data.py`; early stopping theo Macro-F1 validation; dự đoán đúng định dạng chung | CP5 |
| 3.10 | E7 encoder đóng băng, 3 seed | `configs/E7.yaml`, `results/E7/seed{1,2,3}/` | Copy E5, chỉ đổi `freeze_encoder: true` | CP5 |
| 3.11 | Nhãn mềm LLM với prompt P2 (bắt đầu) | `results/llm_soft_labels/` (ID + xác suất, không văn bản) | Chạy qua đêm trên kho `train_unlabeled`; không bao giờ trên 300 tiêu đề test hay 30 tiêu đề gán thử | S2 |
| 3.12 | Gán nhãn đợt 1 | `labeling/round1/tv3.csv` | ~70 nhãn, tự gán, không AI | CP5 |

## Phụ thuộc

- PR #10 (split) vào `main` → E0 chính thức và mọi kết quả khác được tính là kết quả thật.
- 1.13 (300 ID + phân công) → gán nhãn đợt 1 của cả nhóm. TV1 đã có file tiêu đề 2026.
- 1.14 (bootstrap) cần file dự đoán E0/E1/E2 để thử, nhưng viết và test được ngay với dữ liệu giả.

## Biên bản checkpoint

### CP5a — Thứ Tư 18/11
- [ ] Các PR đã gộp theo thứ tự; `main` chạy được toàn bộ test
- [ ] 300 ID test + phân công đã gửi
- [ ] Mức E10 mới chốt, `SPEC.md` đã sửa

### CP5 ◆ — Thứ Sáu 20/11
- [ ] 1.12, 1.14, 1.15
- [ ] 2.12, 2.13, 2.14, 2.16
- [ ] 3.9, 3.10, 3.12
- [ ] PLAN.md tuần 3 đã đánh dấu `[x]`
