# Tuần 4 (giai đoạn S2 của lịch bù) — việc chi tiết và checkpoint

Theo **lịch bù** trong `PLAN.md`: S2 kéo dài **Thứ Bảy 21/11 – Thứ Sáu 27/11**.
Mốc ◆ S2: **ablation E3, E4 (3 seed), E6, E12 có kết quả trên validation; thiết kế E10 chốt lại và có dữ liệu;
gán nhãn gold tiến độ ≥ 180/300.**
TV1 = dữ liệu & thống kê · TV2 = RNN/CNN · TV3 = Transformer & demo.
Nhánh: mỗi người một nhánh cho tuần này (`<ten>/week4`, ví dụ `Kaorichan/week4`), một PR.

Quy tắc nhắc lại: chỉ chấm **validation**; không ai chạy test trước tuần 6. Mọi lần chạy đọc split
`data/splits/cluster_split.jsonl` (SHA-256 `8c2c35ea…`) và ghi `run.json` + `preds_validation.jsonl`.
**Chạy từ commit sạch** (`working_tree_dirty: false`); commit config trước rồi mới chạy.

## Hai thay đổi lớn so với kế hoạch cũ

1. **Nhãn gold (23/11).** Nhãn gán thử và nhãn đợt 1 của TV1, TV3 do LLM tạo, đã chuyển sang
   `labeling/llm_annotations/` (PR #19). TV2 tự gán tay toàn bộ 300 tiêu đề vào `labeling/gold/`
   (68 xong, PR #20). Không ai khác gán nhãn gold, trừ phần gán đôi bên dưới.
2. **E10 phải thiết kế lại.** TV1 đo trên split đã khóa (`results/feasibility/E10_frozen_split.md`):
   trần rò rỉ bằng bản gần trùng chỉ **4,13% test / 4,24% validation**, không phải 27% như quyết định 20/11.
   Mức 5/10/20% bằng bản gần trùng là không làm được. Phương án A đã chốt ngày 23/11 và bổ sung bản sao chính xác.

## Lịch checkpoint

| CP | Thời điểm | Phải có |
| --- | --- | --- |
| CP6a | **Thứ Tư 25/11**, tối, 30 phút | Thiết kế E10 chốt và ghi `context/decisions.md`; buổi gán đôi trực tiếp đã hẹn; TV1, TV3 sửa báo cáo đóng góp ghi sai "gán tay"; config E3, E4, E6, E12 đã commit |
| CP6 ◆ | **Thứ Sáu 27/11**, tối | E3, E4, E6, E12 trên validation (3 seed); dữ liệu E10 + test; E0 chạy lại từ commit sạch; gold ≥ 180/300; bản nháp phần phương pháp (chương 2–3) |

## Quyết định thiết kế E10 — ✅ đã chốt Phương án A (23/11, xem `context/decisions.md`)

| Phương án | Cách làm | Ưu | Nhược |
| --- | --- | --- | --- |
| **A (đề xuất)** | Rò rỉ vào **validation**: đưa bản sao chính xác của 5/10/20% ID validation vào train (ưu tiên bản gần trùng cùng cụm nếu có, phần còn lại là bản sao chính xác). Đo Macro-F1 trên validation, so với E2/E5 sạch. Test không đổi. | Giữ mức 5/10/20%; không chạm test; đo đúng câu hỏi "rò rỉ làm phồng điểm bao nhiêu" | Bản sao chính xác là rò rỉ "nặng" hơn bản gần trùng; phải ghi rõ trong báo cáo |
| B | Chỉ dùng bản gần trùng, hạ mức xuống 1/2/4% | Giống rò rỉ thực tế hơn | Mức nhỏ, chênh lệch có thể dưới độ nhạy (power ~0,006 ở n = 1.791) |

E9 (split ngẫu nhiên) vẫn đo rò rỉ tự nhiên qua bản gần trùng, nên A + E9 phủ cả hai kiểu.

## TV1 — dữ liệu & thống kê

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 1.16 | Dữ liệu rò rỉ E10 theo thiết kế đã chốt | `scripts/make_leakage.py` + test, `results/E10/manifests/leakage_{5,10,20}.json` (ID) + SHA-256 | Chỉ đổi train; validation, test giữ nguyên; seed cố định; test kiểm tra đúng tỷ lệ | CP6 |
| 1.17 | Chạy lại E0 từ commit sạch | `results/E0/seed1/` | `working_tree_dirty: false` | CP6 |
| 1.18 | Sửa ghi chép gán nhãn | `plan/plan_tv1_week2.md` | Ghi đúng: nhãn gán thử do LLM tạo, tham chiếu `labeling/llm_annotations/` | CP6a |
| 1.19 | Gán đôi trực tiếp | ~50 trong 100 tiêu đề gán đôi | Gán tay trong buổi gọi có bật camera, không AI (xem "Gán nhãn") | CP6 |
| 1.20 | Nháp chương 3 (dữ liệu) | `report/ch3_data.md` | Nguồn dữ liệu, khử trùng lặp, split cụm, rò rỉ, quy trình gán nhãn (kể cả sự cố LLM) | CP6 |

## TV2 — RNN/CNN

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 2.15 | E3, E4 config + chạy 3 seed | `configs/E3.yaml`, `configs/E4.yaml`, `results/E3`, `results/E4` | Mỗi config copy E2, đổi đúng một dòng; chạy từ commit sạch | ✅ 23/11 (commit `277fc66`, sạch) |
| 2.17 | Bảng RQ2 tạm cập nhật | `results/rq2_validation.md` | Thêm E3, E4; bootstrap ghép cặp E2–E3, E2–E4 trên validation (chỉ theo dõi) | ✅ 23/11 |
| 2.18 | Gold: gán tay tiêu đề 69–300 | `labeling/gold/` | ≥ 180/300 trước 27/11, đủ 300 trước 04/12 | CP6 |
| 2.19 | Chuẩn bị E13 | `configs/E13.yaml` | Copy E2, chỉ đổi `loss: focal` (γ = 2); có test cho focal loss | ✅ 23/11 (8 test trong `tests/test_losses.py`); đã chạy 3 seed: 0,7538 ± 0,0088 |
| 2.20 | Nháp chương 2 (RNN/CNN/attention) | `report/ch2_rnn_cnn.md` | TextCNN, BiLSTM, attention, cách huấn luyện chung | ✅ 23/11 (bản nháp) |

## TV3 — Transformer & LLM

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 3.13 | E12: E5 + nhãn cứng LLM, 3 seed | `configs/E12.yaml`, `results/E12/` | Copy E5, chỉ thêm nhãn cứng (argmax P2) của `train_unlabeled` vào train | CP6 |
| 3.14 | E6: chưng cất nhãn mềm, 3 seed | `configs/E6.yaml`, `results/E6/` | α, T ∈ {1, 2, 4} chọn trên validation, ghi lưới đã thử | CP6 |
| 3.15 | Sửa báo cáo đóng góp | `results/rq3_week3_contribution_report.md`, `results/llm_prompts/contribution_report_week2.md` | Ghi đúng: nhãn gán thử và đợt 1 do LLM tạo | CP6a |
| 3.16 | Gán đôi trực tiếp | ~50 trong 100 tiêu đề gán đôi | Như 1.19 | CP6 |
| 3.17 | Nháp chương 2 (Transformer, chưng cất) | `report/ch2_transformer.md` | DistilBERT, đóng băng, nhãn mềm/cứng | CP6 |

## Gán nhãn

- Gold chính: TV2 gán tay 300 tiêu đề (`labeling/gold/`). Không ai mở `labeling/llm_annotations/` khi gán.
- **100 tiêu đề gán đôi cần người thứ hai gán tay** để có kappa. Đề xuất: TV1 và TV3 mỗi người ~50,
  gán **trực tiếp trong buổi gọi có bật camera** (hai buổi ~45 phút), TV2 có mặt, không dùng AI.
  Nếu không làm được: tìm một người ngoài đã đọc `labeling/GUIDE.md`; nếu vẫn không có, báo cáo ghi
  "một người gán, không có kappa".
- Tự kiểm tra của TV2: sau ~1 tuần gán lại 20 tiêu đề không xem nhãn cũ, ghi tỷ lệ tự đồng ý.

## Phụ thuộc

- Thiết kế E10 (CP6a) → 1.16 → E10 cho E2 và E5 ở S3.
- E12 trước E6 (E6 so với cả E5 và E12).
- Gold đủ 300 và kappa → khóa nhãn test **04/12** (S3).

## Biên bản checkpoint

### CP6a — Thứ Tư 25/11
- [x] Thiết kế E10 chốt (Phương án A, 23/11), ghi `context/decisions.md`
- [ ] Buổi gán đôi trực tiếp đã hẹn
- [ ] 1.18, 3.15 (sửa ghi chép gán nhãn)
- [ ] Config E3, E4, E6, E12 đã commit

### CP6 ◆ — Thứ Sáu 27/11
- [ ] 1.16, 1.17, 1.19, 1.20
- [ ] 2.15, 2.17, 2.18, 2.19, 2.20
- [ ] 3.13, 3.14, 3.16, 3.17
- [ ] PLAN.md tuần 4 đã đánh dấu `[x]`

# Tiến độ TV1 ngày 26/11

- E10: manifest ID và hướng dẫn tại `results/E10/`; đạt 90/180/359 ID lộ, giữ toàn bộ validation; loader huấn luyện chưa tích hợp overlay.
- E0: chạy validation từ commit `865ac25cb555211410cafb99046a6c3bd2a41435` trên checkout sạch, Macro-F1 0,709552; dự đoán trùng hash với lần trước. Metadata cũ được giữ trong `results/E0/seed1/run_previous_20261122.json`.
- Ghi chép nhãn thử đã đính chính; bản nháp dữ liệu tại `report/ch3_data.md`. Gold TV2 hiện có 68 dòng; gán đôi và đạt mục tiêu 180 cần nhóm thực hiện, không tính kappa khi chưa có người thứ hai.
- Người dùng xác nhận sẽ tự gán tay 100 tiêu đề gán đôi; lịch, danh sách ID và khả năng đã xem nhãn LLM trước đó còn cần xác nhận. Hướng dẫn TV1 tuần 4 nêu khoảng 33 tiêu đề chưa từng đưa qua LLM, nên chưa thể mặc định cả 100 ID đều độc lập.
- Kiểm tra chỉ theo ID trên `labeling/assignment.csv`: trong 100 ID gán đôi ban đầu có 67 ID thuộc phân công TV1 và 33 ID không thuộc phân công TV1. Đây chỉ là thông tin phân công, không chứng minh ai đã xem nhãn nào; file nhãn LLM TV1 chưa từng commit. Người dùng chưa chắc đã xem nhãn LLM, nên cần kiểm tra với TV2 và chốt ID trước khi gán; không mở nhãn LLM để kiểm tra khi gán.
- Người dùng chọn đợi chốt đủ 100 ID với TV2 trước khi chuẩn bị gói gán đôi; không mặc định sử dụng riêng 33 ID trên. Lịch và tính độc lập của các ID vẫn cần xác nhận.
- Kiểm tra cấu trúc các gói cục bộ: `tv1_round1_blank.csv` và `tv1_round2_blank.csv` tuy có tên `blank` nhưng lần lượt có đủ nhãn và lý do trên 69 và 65 dòng; nguồn nhãn và việc người dùng đã xem chúng chưa được xác minh. Không dùng hai file này làm mẫu trống cho lượt gán độc lập; không commit hoặc hiển thị nội dung của chúng. Các file vẫn được giữ nguyên tại `data/raw/headlines_2026/labeling_packets/`.
- Kiểm thử toàn bộ: 104 test đạt khi bổ sung tạm `datasketch==2.0.0`, `duckdb==2.0.0` và `pytz==2026.5` từ lock vào thư mục phụ; môi trường Python gốc thiếu các gói đó.
