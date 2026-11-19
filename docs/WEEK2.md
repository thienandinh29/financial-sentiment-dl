# Tuần 2 — việc chi tiết và checkpoint

Mốc ◆ của tuần (PLAN.md): **split chốt kèm SHA-256; gán thử 30 tiêu đề.**
TV1 = dữ liệu & thống kê · TV2 = RNN/CNN · TV3 = Transformer & demo.
Hướng dẫn từng bước kèm code đã chạy thử: [TV1](guides/TV1-tuan2.md), [TV3](guides/TV3-tuan2.md).
Sau CP4: họp "hiểu bài" 90 phút theo [meetings/week2-review.md](meetings/week2-review.md) (demo + hỏi chéo + biên bản).

> **Cập nhật 12/11:** các ngày CP cũ (02/11, 04/11, 07/11, 11/11) đã qua mà chưa có PR nào được gộp, chưa có split.
> Theo **lịch bù** trong `PLAN.md` (giữ nguyên phạm vi, làm dày hơn), CP2, CP3 và mốc ◆ gộp thành **một buổi làm
> việc Thứ Sáu 13/11 tối (~2 giờ)**; phần còn lại của tuần 2 xong trước **Chủ Nhật 15/11 tối** (CP4).
> Cả nhóm duyệt lịch này và ghi vào `context/decisions.md`.

Nhánh: theo đề xuất "mỗi người một nhánh mỗi tuần" — `<ten>/week2` (TV2: `Kaorichan/week2`).

## Lịch checkpoint (đề xuất)

| CP | Thời điểm | Phải có |
| --- | --- | --- |
| CP2 + CP3 ◆ (gộp) | **Thứ Sáu 13/11**, tối, ~2 giờ | **Mốc ◆: split chốt + SHA-256 trên `main`.** Cùng buổi: 6 quyết định (mục "Chương trình họp"); PR tuần 1 và PR của TV2 đã duyệt và gộp; **hợp đồng chung** (mục dưới) đã thống nhất; chọn xong 30 tiêu đề gán thử |
| CP4 (phần còn lại) | **Chủ Nhật 15/11**, tối | E0 trên validation; khung đánh giá chung; prompt A/B/C; mỗi người gán xong 30 tiêu đề; kappa; GUIDE sửa theo bất đồng; vòng huấn luyện chung có test (TV2: đã có trên PR #9) |

Trong các bảng việc bên dưới, cột CP: **CP3** = buổi họp Thứ Sáu 13/11; **CP4** = Chủ Nhật 15/11.
Không chốt được split thì không chạy E1/E2/E5 (tuần 3), vì mọi thí nghiệm đọc split đã chốt.

## Chương trình họp Thứ Sáu 13/11

Họp làm việc: chạy code thật tại chỗ, không chỉ nói.

| Phút | Việc | Ai |
| --- | --- | --- |
| Trước giờ họp | TV1 xong 1.7 (kiểm tay 100 cặp, ~1 giờ) **trước khi chốt split**, vì ngưỡng có thể làm đổi cụm và đổi split; TV1 mở PR cho `tv1/download-data`; TV2 gửi riêng file tiêu đề 2026 cho TV1 (không commit); mọi người clone repo, tạo venv Python 3.11, chạy `python data/download.py --only fns` | Cả nhóm |
| 0–20 | **Quyết định:** lịch bù trong `PLAN.md`; GUIDE; mức E10 (đề xuất giữ 10/25%, hạ mức cao nhất vì trần đo được ≈ 50,28%); quy tắc nhánh; hợp đồng chung (thứ tự nhãn, split chỉ chứa ID, định dạng dự đoán); proposal đã nộp chưa | Cả nhóm |
| 20–50 | **Duyệt và gộp PR** theo thứ tự #7 → #6 → #2 → #5 → PR của TV1 → #8 → #9. Mỗi PR do người khác duyệt trên màn hình; xung đột nhỏ ở `MAP.md`, `decisions.md` thì giữ mọi dòng | Cả nhóm |
| 50–65 | **Chốt split:** TV1 chạy `scripts/make_splits.py` (code trong `docs/guides/TV1-tuan2.md`), chạy lại lần hai kiểm hash, điền `split_sha256` vào `configs/E2.yaml`, commit | TV1 |
| 65–75 | **Chọn 30 tiêu đề gán thử** (`pick30.py`), gửi danh sách ID cho cả nhóm; từ lúc này mỗi người tự gán, không AI | TV1 |
| 75–85 | **Chia việc tới Chủ Nhật 15/11** (bảng CP4); chốt ngày họp "hiểu bài" 90 phút | Cả nhóm |

## Hợp đồng chung — chốt ở CP3 (ai cũng phụ thuộc)

Ba thứ này dùng chung giữa 3 người; đổi sau khi chốt phải ghi `context/decisions.md`.

1. **Thứ tự nhãn duy nhất:** `LABELS = ["bearish", "neutral", "bullish"]` → chỉ số 0, 1, 2 trong mọi
   mô hình, file dự đoán và ma trận nhầm lẫn. (FNS gốc dùng 0 = bearish, 1 = bullish, 2 = neutral;
   LLM dùng A = bearish, B = neutral, C = bullish — mọi bộ nạp phải đổi về thứ tự trên.)
2. **File split** `data/splits/<ten>.jsonl`: mỗi dòng `{"id": "fns-train-00000", "split": "train"}`
   với `split ∈ {train, validation, test}`, **chỉ ID, không văn bản**. Văn bản lấy theo ID từ
   `data/raw/fns/`. SHA-256 ghi trong `data/splits/SHA256SUMS` và `split_sha256` của config.
   Hai file tuần này: `cluster_split.jsonl` (chính) và `random_split.jsonl` (cho E9), cùng tỷ lệ 70/15/15.
3. **File dự đoán** `results/<E>/seed<k>/preds_<split>.jsonl`: mỗi dòng
   `{"id": ..., "probs": [p_bearish, p_neutral, p_bullish]}`. Mọi mô hình (E0–E13) ghi cùng dạng
   này để TV1 tính metric, bootstrap và McNemar bằng một code duy nhất. Tuần 2–5 chỉ ghi
   `preds_validation`; `preds_test` chỉ có ở tuần 6 (AGENTS.md quy tắc 1).

## Việc mang sang từ tuần 1 (làm trước tiên)

| Việc | Ai | Xong khi |
| --- | --- | --- |
| Nộp proposal (điền tên nhóm, MSSV, URL repo) | TV1 | Ảnh xác nhận nộp gửi nhóm |
| Mở PR `tv1/download-data` | TV1 | PR mở, có người duyệt |
| Duyệt và gộp PR tuần 1 theo thứ tự #7 → #6 → #2 → #5 → TV1 → #8 | Cả nhóm | `main` có đủ code tuần 1; mỗi PR có 1 lượt duyệt của người khác |
| Chạy lại ô cài đặt trên Colab sau khi #6 được gộp | TV3 | Báo `requirements.lock` cài được trên Python 3.13 hay không |

## TV1 — dữ liệu & thống kê

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 1.7 | Kiểm tra tay 100 cặp gần trùng | `results/feasibility/manual_check_100.csv` (id_a, id_b, loại cặp, đúng/sai, ghi chú) | 50 cặp từ vựng + 50 cặp ngữ nghĩa, chọn ngẫu nhiên có seed; báo tỷ lệ đúng từng loại; nếu ngữ nghĩa sai nhiều thì đề xuất đổi ngưỡng (ghi `decisions.md`) | CP3 |
| 1.8 | Split theo cụm 70/15/15 | `data/splits/cluster_split.jsonl`, `SHA256SUMS`, `scripts/make_splits.py` + test | Cả cụm nằm trọn một phía; phân bố nhãn mỗi phía lệch ≤ 2 điểm % so với toàn bộ; seed cố định; chạy lại ra cùng hash | CP3 |
| 1.9 | Split ngẫu nhiên cho E9 | `data/splits/random_split.jsonl` | Cùng tỷ lệ và phân tầng nhãn; báo số cặp trùng bị tách qua train/test (đây chính là rò rỉ E9 đo) | CP3 |
| 1.10 | Baseline E0 (TF-IDF + LR) | `configs/E0.yaml`, `models/tfidf_lr.py`, `results/E0/seed1/preds_validation.jsonl`, Macro-F1 validation | TF-IDF fit **chỉ trên train**; class_weight tính trên train; chỉ đánh giá validation | CP4 |
| 1.11 | Chọn 30 tiêu đề gán thử | `labeling/trial30_ids.csv` (id, url, chỉ ID) | Lấy ngẫu nhiên có seed từ kho `test_candidate` (RSS) của TV2; **30 tiêu đề này bị loại khỏi 300 tiêu đề test** | CP3 |

## TV2 — RNN/CNN và vòng huấn luyện chung

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 2.6 | Bộ nạp dữ liệu theo split | `train/data.py` + test | Đọc split, **kiểm tra SHA-256 trước khi đọc**, lấy văn bản theo ID, đổi nhãn về `LABELS`; từ chối split thiếu/sai hash | CP3 |
| 2.7 | Vòng huấn luyện chung | `train/loop.py`, `train/run.py` (đọc YAML config) | Seed cố định (Python/NumPy/torch); AMP trên GPU; early stopping theo `val_macro_f1`; lưu checkpoint tốt nhất vào thư mục gitignore; **không có lệnh nào chạy test** | CP4 |
| 2.8 | Ghi log mỗi lần chạy | `results/<E>/seed<k>/run.json` | Có commit, dirty flag, config, seed, phiên bản thư viện, GPU, thời gian, đường cong val theo epoch | CP4 |
| 2.9 | Loss: weighted CE và focal | `train/losses.py` + test | Trọng số lớp tính **chỉ trên train**; focal γ = 2 (cho E13); test trên batch nhỏ cho giá trị tính tay | CP4 |
| 2.10 | Kiểm tra end-to-end trên CPU | test chạy vòng huấn luyện 2 epoch với mô hình tí hon | Loss giảm, ghi đủ `run.json` và `preds_validation.jsonl` đúng định dạng | CP4 |

## TV3 — Transformer & đánh giá

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 3.5 | Ba biến thể prompt A/B/C | `configs/llm_prompts.yaml`, notebook `notebooks/llm_prompts.ipynb` | Mỗi prompt chạy trên **validation của split theo cụm**; báo Macro-F1 validation và `abc_mass`; chọn prompt trên validation, không trên test | CP4 |
| 3.6 | Kiểm tra top-10 logits | `results/llm_prompts/top10_check.json` (không văn bản) | Với mỗi prompt, tỷ lệ mẫu có chữ cái thắng nằm trong top-10 và `abc_mass` thấp nhất; nêu nếu prompt nào lệch | CP4 |
| 3.7 | Khung đánh giá chung | `evaluate/metrics.py` + test | Đọc `preds_<split>.jsonl` + nhãn thật → Macro-F1, F1 từng lớp, ma trận nhầm lẫn theo `LABELS`; test với ví dụ tính tay | CP4 |
| 3.8 | Sửa bộ đọc split trong notebook Colab | `notebooks/colab_setup.ipynb` | Đọc split dạng ID (hợp đồng mục 2) thay vì đọc `text` trong file split | CP4 |

Lưu ý: `evaluate/` là vùng TV1 trong MAP.md — TV3 làm 3.7 thì báo TV1 trước, TV1 duyệt PR.

## Cả nhóm — gán thử 30 tiêu đề (con người gán, không AI)

1. TV1 chọn 30 ID (việc 1.11). Mỗi người mở URL/tiêu đề trên máy mình, **gán độc lập**, không xem nhãn
   của người khác, không hỏi AI. Lưu `labeling/trial/<ten>.csv` với cột `id, label, reason`.
2. Sau khi cả 3 nộp: tính tỷ lệ đồng ý và kappa từng cặp (TV1), liệt kê các tiêu đề bất đồng.
3. Họp 30 phút: thảo luận bất đồng, sửa GUIDE (thêm quy tắc/ví dụ), ghi dòng vào "Lịch sử sửa đổi".
4. Không commit văn bản tiêu đề — chỉ ID, nhãn, lý do.

## Phụ thuộc

- Định dạng split (CP3) → 2.6, 3.5, 3.8, 1.10. Chốt định dạng trước, file split thật đến sau.
- 1.11 (30 ID) → gán thử của cả nhóm. Cần export RSS của TV2 (PR #2) trên máy TV1.
- Split chốt (1.8) → E0 chính thức (1.10) và prompt trên validation (3.5). Trước đó có thể chạy
  thử trên split tạm, nhưng **không báo số** từ split tạm.

## Biên bản checkpoint

### Họp gộp Thứ Sáu 13/11 — CP3, mốc ◆
Quyết định:
- [ ] Lịch bù trong `PLAN.md` (nộp mục tiêu 19/12): duyệt / sửa
- [ ] GUIDE: duyệt hoặc ghi điều cần sửa
- [ ] Mức E10: giữ 10/25/50% hay hạ mức cao nhất (trần đo được ≈ 50,28%)
- [ ] Quy tắc nhánh "mỗi người một nhánh mỗi tuần": duyệt / bác
- [ ] Hợp đồng chung (nhãn, split, dự đoán) đã thống nhất
- [ ] Proposal đã nộp

Việc làm tại chỗ:
- [ ] PR đã duyệt và gộp: #7 → #6 → #2 → #5 → PR của TV1 → #8 → #9
- [ ] 1.7 xong trước khi chốt split; 1.8, 1.9: split chốt, SHA-256 trên `main`, `split_sha256` đã điền
- [ ] 1.11: 30 tiêu đề gán thử đã chọn và gửi cả nhóm
- [ ] 2.6 (TV2): có trên PR #9

### CP4 — Chủ Nhật 15/11 (phần còn lại của tuần 2)
- [ ] 1.10 (E0)
- [ ] 2.7, 2.8, 2.9, 2.10 (TV2: có trên PR #9)
- [ ] 3.5, 3.6, 3.7, 3.8
- [ ] 3 file gán thử + kappa + GUIDE đã sửa
- [ ] PLAN.md tuần 2 đã đánh dấu `[x]`
