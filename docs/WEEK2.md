# Tuần 2 — việc chi tiết và checkpoint

Mốc ◆ của tuần (PLAN.md): **split chốt kèm SHA-256; gán thử 30 tiêu đề.**
TV1 = dữ liệu & thống kê · TV2 = RNN/CNN · TV3 = Transformer & demo.
Hướng dẫn từng bước kèm code đã chạy thử: [TV1](guides/TV1-tuan2.md), [TV3](guides/TV3-tuan2.md).

> **Lịch trễ ~5 ngày.** PLAN ghi tuần 2 là 29/10–4/11, nhưng việc tuần 1 bàn giao ngày 01/11 và
> CP2 chưa họp. Đề xuất bên dưới dời mốc ◆ tuần 2 sang **Thứ Bảy 07/11**; tuần 3 còn 08–11/11,
> nên tuần 3 phải dồn. Cả nhóm duyệt lịch này ở CP2 (bù) và ghi vào `context/decisions.md`.

Nhánh: theo đề xuất "mỗi người một nhánh mỗi tuần" — `<ten>/week2` (TV2: `Kaorichan/week2`).

## Lịch checkpoint (đề xuất)

| CP | Thời điểm | Phải có |
| --- | --- | --- |
| CP2 (bù) | **Thứ Hai 02/11**, tối, 30 phút | 3 quyết định tuần 1 (GUIDE, mức E10, quy tắc nhánh) + lịch này; proposal đã nộp |
| CP3 | **Thứ Tư 04/11**, tối | PR tuần 1 đã gộp; **định dạng split** và **hợp đồng chung** (mục dưới) đã thống nhất; chọn xong 30 tiêu đề gán thử |
| CP4 ◆ | **Thứ Bảy 07/11**, tối | Split chốt + SHA-256 trên `main`; mỗi người gán xong 30 tiêu đề; GUIDE sửa theo bất đồng; E0 trên validation; vòng huấn luyện chung có test; khung đánh giá chung; prompt A/B/C |

Không đạt CP4 thì không chạy E1/E2/E5 (tuần 3), vì mọi thí nghiệm đọc split đã chốt.

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
| 1.8 | Split theo cụm 70/15/15 | `data/splits/cluster_split.jsonl`, `SHA256SUMS`, `scripts/make_splits.py` + test | Cả cụm nằm trọn một phía; phân bố nhãn mỗi phía lệch ≤ 2 điểm % so với toàn bộ; seed cố định; chạy lại ra cùng hash | CP4 |
| 1.9 | Split ngẫu nhiên cho E9 | `data/splits/random_split.jsonl` | Cùng tỷ lệ và phân tầng nhãn; báo số cặp trùng bị tách qua train/test (đây chính là rò rỉ E9 đo) | CP4 |
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

### CP2 (bù) — 02/11
- [ ] GUIDE: duyệt hoặc ghi điều cần sửa
- [ ] Mức E10: giữ 10/25/50% hay hạ mức cao nhất (trần đo được ≈ 50,28%)
- [ ] Quy tắc nhánh "mỗi người một nhánh mỗi tuần": duyệt / bác
- [ ] Lịch tuần 2 dời mốc ◆ sang 07/11: duyệt / sửa
- [ ] Proposal đã nộp

### CP3 — 04/11
- [ ] PR tuần 1 đã gộp vào `main`
- [ ] Hợp đồng chung (nhãn, split, dự đoán) đã thống nhất
- [ ] 1.7, 1.11, 2.6

### CP4 ◆ — 07/11
- [ ] 1.8, 1.9, 1.10
- [ ] 2.7, 2.8, 2.9, 2.10
- [ ] 3.5, 3.6, 3.7, 3.8
- [ ] 3 file gán thử + kappa + GUIDE đã sửa
- [ ] PLAN.md tuần 2 đã đánh dấu `[x]`
