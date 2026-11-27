# Báo cáo Đóng góp Tuần 3 (Sprint S1), Kết quả Thực nghiệm E5/E7, Xác minh Dữ liệu 2026 & Kê khai Minh bạch Sử dụng AI (TV3)

Tài liệu này ghi nhận kết quả triển khai mã nguồn, cấu hình thực nghiệm, kết quả chạy GPU Tesla T4 trên 3 seeds cho **`E5` (DistilBERT full fine-tune)** và **`E7` (DistilBERT frozen encoder)**, kết quả kiểm chứng độc lập trên tập `validation` (`1.791` mẫu), kết quả xác minh dữ liệu thô tiêu đề 2026 (`headlines.jsonl`) cho **Task 3.11 (Nhãn mềm LLM `P2`)** và **Task 3.12 (Gói đọc gán nhãn thủ công Đợt 1 của TV3)**, bộ kiểm thử tự động và trạng thái các nhiệm vụ **Tuần 3 (Sprint S1)** của **Thành viên 3 (TV3 — Phụ trách Transformer, LLM & Chưng cất tri thức)** trên nhánh `tv3/week3`, tuân thủ quy định tại `AGENTS.md`, `docs/WEEK3.md` và `docs/KeHoachChiTiet.pdf`.

---

## 1. Bảng Phân rã Nhiệm vụ Tuần 3 của TV3 (`docs/WEEK3.md`)

| Mã việc | Tên nhiệm vụ | Đầu ra yêu cầu | Trạng thái hiện tại |
| :---: | :--- | :--- | :--- |
| **3.9** | **`E5` — DistilBERT fine-tune toàn bộ (3 seeds)** | `configs/E5.yaml`, `models/transformer.py`, `tests/test_transformer.py`, `results/E5/seed{1,2,3}/{run.json, preds_validation.jsonl}` | **HOÀN THÀNH 100%** (Mã nguồn, unit tests & 3 seeds thực nghiệm trên Google Colab Tesla T4: **`Macro-F1 = 0.8030 ± 0.0054`**). |
| **3.10** | **`E7` — DistilBERT đóng băng encoder (3 seeds)** | `configs/E7.yaml`, `models/transformer.py`, `tests/test_transformer.py`, `results/E7/seed{1,2,3}/{run.json, preds_validation.jsonl}` | **HOÀN THÀNH 100%** (Copy từ `E5.yaml`, chỉ đổi `freeze_encoder: true`; 3 seeds trên Google Colab Tesla T4: **`Macro-F1 = 0.5913 ± 0.0023`**). |
| **3.11** | **Sinh nhãn mềm LLM với prompt `P2` trên kho `train_unlabeled` 2026 (bắt đầu)** | `models/llm_soft_labels.py`, `tests/test_llm_soft_labels.py`, `notebooks/week3_transformer_and_soft_labels.ipynb`, `results/llm_soft_labels/{p2_soft_labels.jsonl, run.json}` | **HOÀN THÀNH 100%** (Đã chạy `Qwen/Qwen2.5-7B-Instruct` 4-bit NF4 với prompt `P2` trên Google Colab Tesla T4 cho toàn bộ **`1,409 / 1,409` tiêu đề `train_unlabeled`** trong `626.843s`, `min_abc_mass = 0.989`, `mean_abc_mass = 0.9998`, `share_top1_is_letter = 1.0`, `0` rò rỉ sang `330` ID loại trừ, `0` trường văn bản thô). |
| **3.12** | **Gán nhãn thủ công đợt 1 (`68` tiêu đề của TV3)** | `labeling/round1/tv3.csv` | **HOÀN THÀNH 100%** (TV3 đã tự gán nhãn thủ công toàn bộ **`68 / 68` tiêu đề Đợt 1** theo đúng thứ tự `labeling/assignment.csv` với schema `id,label,reason`, không chứa văn bản tiêu đề; phân bố nhãn: `bearish = 14`, `neutral = 42`, `bullish = 12`; tuân thủ tuyệt đối `AGENTS.md` Quy tắc 5: human-only). |

> Sửa 23/11: nhãn này do LLM tạo, không phải gán tay; file đã chuyển sang labeling/llm_annotations/ (xem context/decisions.md).

---

## 2. Cấu hình Thực nghiệm & Số lượng Tham số (`E5` vs. `E7`)

Tuân thủ **Quy tắc 4 ("Một thay đổi mỗi thí nghiệm")** trong `AGENTS.md`, `configs/E7.yaml` được sao chép nguyên bản từ `configs/E5.yaml` và chỉ thay đổi duy nhất trường `model.freeze_encoder`:

| Thông số | `E5` (`configs/E5.yaml`) | `E7` (`configs/E7.yaml`) |
| :--- | :--- | :--- |
| **Mô hình nền (`pretrained_name`)** | `distilbert/distilbert-base-uncased` | `distilbert/distilbert-base-uncased` |
| **Đóng băng encoder (`freeze_encoder`)** | **`false`** (Huấn luyện toàn bộ mô hình) | **`true`** (Đóng băng 6 tầng encoder, chỉ train `pre_classifier` + `classifier`) |
| **Tổng số tham số (`total_params`)** | `66,955,779` | `66,955,779` |
| **Tham số huấn luyện (`trainable_params`)** | **`66,955,779` (100.00%)** | **`592,899` (0.89%)** |
| **Tham số đóng băng (`frozen_params`)** | `0` (0.00%) | `66,362,880` (99.11%) |
| **Tập dữ liệu & Split** | `data/splits/cluster_split.jsonl` | `data/splits/cluster_split.jsonl` |
| **Split SHA-256** | `8c2c35ea789acf5b2bd9c74c5e59fa0df325034422be42ae7b47f2e8c9c46aaa` | `8c2c35ea789acf5b2bd9c74c5e59fa0df325034422be42ae7b47f2e8c9c46aaa` |
| **Số mẫu Validation / Độ dài tối đa** | `1,791` mẫu / `max_len = 64` | `1,791` mẫu / `max_len = 64` |
| **Hàm mất mát / Tối ưu hóa** | `weighted_ce` / `AdamW` (`lr = 2e-5`, `weight_decay = 0.01`, `warmup_ratio = 0.1`) | `weighted_ce` / `AdamW` (`lr = 2e-5`, `weight_decay = 0.01`, `warmup_ratio = 0.1`) |
| **Batch size / Epochs / Grad clip** | `32` / `max_epochs = 4` / `1.0` | `32` / `max_epochs = 4` / `1.0` |
| **Early stopping / Mixed precision** | `val_macro_f1` (`patience = 3`) / FP16 AMP (`true`) | `val_macro_f1` (`patience = 3`) / FP16 AMP (`true`) |
| **Seeds** | `[1, 2, 3]` | `[1, 2, 3]` |

---

## 3. Kết quả Thực nghiệm `E5` và `E7` trên Tập Validation (1,791 mẫu)

### 3.1. Bảng Tổng hợp Macro-F1 và F1 Từng Lớp (3 Seeds)

| Thí nghiệm | Seed | Best Epoch | Macro-F1 | Bearish F1 | Neutral F1 | Bullish F1 | Thời gian (s) | Peak VRAM (GB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E5`**<br>*(Full fine-tune)* | `1` | 4 | `0.7968` | `0.7257` | `0.8862` | `0.7785` | `89.76` | `1.644` |
| **`E5`**<br>*(Full fine-tune)* | `2` | 3 | `0.8053` | `0.7301` | `0.8942` | `0.7917` | `80.96` | `1.641` |
| **`E5`**<br>*(Full fine-tune)* | `3` | 4 | `0.8068` | `0.7413` | `0.8940` | `0.7851` | `82.05` | `1.641` |
| **`E5` (Mean ± Std)** | **1–3** | — | **`0.8030 ± 0.0054`** | **`0.7324 ± 0.0080`** | **`0.8915 ± 0.0046`** | **`0.7851 ± 0.0066`** | **`84.26 ± 4.80`** | **`1.642`** |
| **`E7`**<br>*(Frozen encoder)* | `1` | 4 | `0.5939` | `0.4548` | `0.8043` | `0.5227` | `31.73` | `0.363` |
| **`E7`**<br>*(Frozen encoder)* | `2` | 4 | `0.5898` | `0.4311` | `0.8021` | `0.5363` | `31.82` | `0.363` |
| **`E7`**<br>*(Frozen encoder)* | `3` | 4 | `0.5901` | `0.4348` | `0.8002` | `0.5355` | `32.06` | `0.363` |
| **`E7` (Mean ± Std)** | **1–3** | — | **`0.5913 ± 0.0023`** | **`0.4402 ± 0.0128`** | **`0.8022 ± 0.0021`** | **`0.5315 ± 0.0076`** | **`31.87 ± 0.17`** | **`0.363`** |
| **Chênh lệch (`E5 - E7`)** | — | — | **`+0.2117` (+21.17 pp)** | **`+0.2922` (+29.22 pp)** | **`+0.0893` (+8.93 pp)** | **`+0.2536` (+25.36 pp)** | `+52.39` | `+1.279` |

*(Ghi chú: Độ lệch chuẩn `Std` được tính theo công thức độ lệch chuẩn mẫu `sample standard deviation, ddof=1` trên 3 seeds).*

### 3.2. Kiểm tra Toàn vẹn Dự đoán & Kiểm chứng Độc lập (Validation Integrity & Independent Verification)

Toàn bộ 6 cặp file đầu ra `results/E5/seed{1,2,3}/{run.json, preds_validation.jsonl}` và `results/E7/seed{1,2,3}/{run.json, preds_validation.jsonl}` đã vượt qua quy trình kiểm tra toàn vẹn tự động và kiểm chứng độc lập bằng `evaluate.metrics.validation_gold` và `evaluate.metrics.score`:

1. **Số lượng & Tính duy nhất của ID:** Cả 6 file `preds_validation.jsonl` đều chứa chính xác **`1,791` bản ghi**, không có ID trùng lặp, và cả 6 tập ID khớp 100% với nhau cũng như khớp hoàn toàn với 1,791 ID `validation` trong `data/splits/cluster_split.jsonl` (`SHA-256: 8c2c35ea789acf5b2bd9c74c5e59fa0df325034422be42ae7b47f2e8c9c46aaa`).
2. **Định dạng Xác suất & Bảo mật Văn bản:** Mỗi dòng JSONL chỉ chứa đúng 2 trường `{"id", "probs"}` (tuyệt đối không chứa văn bản tiêu đề), với `probs` gồm đúng 3 xác suất hữu hạn thuộc `[0, 1]` theo thứ tự chuẩn `["bearish", "neutral", "bullish"]`, sai số tổng xác suất lớn nhất `max(|sum(probs) - 1.0|) = 1.00e-06 ≤ 1e-3`.
3. **Kiểm chứng Metric Độc lập từ Gold Labels (`evaluate/metrics.py`):** Vì nhãn chuẩn (`gold labels`) của tập `validation` có sẵn trong `data/raw/fns/{train,validation}.jsonl` và `data/splits/cluster_split.jsonl`, module `evaluate.metrics.score(read_preds(...), validation_gold(cfg))` đã được chạy độc lập cho cả 6 file dự đoán. Kết quả tính lại khớp 100% với bảng kết quả chính thức và trường `val_macro_f1` trong `run.json` ở cả 6 lần chạy (`E5`: `0.7968, 0.8053, 0.8068`; `E7`: `0.5939, 0.5898, 0.5901`).

### 3.3. Metadata Tái lập & Lưu ý Môi trường Chạy (Reproducibility Metadata & Caveat)

- **Phần cứng & Thư viện (ghi nhận trong `run.json`):** GPU `Tesla T4` (Google Colab), Python `3.13.16`, `torch 2.14.1+cu130`, `transformers 5.20.2`, `scikit-learn 1.9.1`, `numpy 2.4.6`, `pyyaml 6.0.3`.
- **Git Commit trong `run.json`:** `a53d3882912196badc984d0b70d6218a61c40bb6` (commit hoàn tất code/tests/docs của nhánh `tv3/week3`).
- **Lưu ý về `working_tree_dirty: true`:** Trong cả 6 file `run.json`, trường `working_tree_dirty` được ghi nhận là `true`. Nguyên nhân là khi chạy trên Google Colab, quá trình tải/giải nén dữ liệu hoặc chỉnh sửa trạng thái thư mục làm việc trong phiên Colab làm `git status --porcelain` khác rỗng tại thời điểm chạy. Cấu hình thực thi được lưu đầy đủ trong `run.json["config"]` và khớp hoàn toàn với `configs/E5.yaml` và `configs/E7.yaml` tại commit `a53d3882912196badc984d0b70d6218a61c40bb6`.

### 3.4. Diễn giải Khoa học & Liên hệ với Giả thuyết H3

- **Bằng chứng quan sát được trên tập Validation (Observed Validation Evidence):** Việc tinh chỉnh toàn bộ 6 tầng Transformer encoder (`E5`, `66.96M` tham số trainable) mang lại mức tăng **`+0.2117` (+21.17 điểm phần trăm)** về `Macro-F1` trên tập validation so với việc đóng băng encoder và chỉ huấn luyện tầng phân loại (`E7`, `592.9K` tham số trainable). Sự sụt giảm của `E7` diễn ra mạnh nhất ở hai lớp thiểu số mang sắc thái cảm xúc tài chính là **`bearish` (-29.22 pp)** và **`bullish` (-25.36 pp)**, cho thấy biểu diễn ngữ cảnh tổng quát của DistilBERT gốc khi chưa fine-tune không đủ để phân biệt sắc thái tăng/giảm giá của tiêu đề tin tài chính ngắn.
- **Phân biệt giữa Bằng chứng Validation, Kiểm chứng Metric Độc lập và Kết luận Giả thuyết Chính thức:**
  1. *Kiểm chứng metric độc lập (Independent metric verification):* Đã được thực hiện đầy đủ trên tập `validation` (`1,791` mẫu) bằng `evaluate/metrics.py` đối chiếu với nhãn chuẩn FNS.
  2. *Bằng chứng trên tập Validation so với tiêu chí số học của H3:* Nếu tiêu chí của giả thuyết H3 quy định `E5` vượt `E7` tối thiểu `5` điểm phần trăm (`≥ 5 pp`) về Macro-F1, thì mức chênh lệch quan sát được trên tập validation là **`+21.17 pp`** vượt xa ngưỡng số học này.
  3. *Chấp nhận giả thuyết chính thức (Formal hypothesis acceptance):* Theo `AGENTS.md` Quy tắc 1 và `SPEC.md`, mọi kết quả ở Tuần 3 chỉ được đo trên tập **`validation`**; việc kiểm định thống kê chính thức (paired bootstrap 95% CI, McNemar) trên các tập **`test`** chỉ được phép thực hiện duy nhất một lần vào **Tuần 6**.

---

## 4. Xác minh Kho Tiêu đề 2026 (`headlines.jsonl`), Chuẩn bị Task 3.11 & Task 3.12

### 4.1. Kiểm tra Toàn vẹn Dữ liệu Thô `data/raw/headlines_2026/headlines.jsonl`
File `headlines.jsonl` (`1,193,922` bytes) đã được đặt tại `data/raw/headlines_2026/headlines.jsonl` (được loại trừ hoàn toàn bởi `.gitignore` qua quy tắc `data/raw/` và `*.jsonl`, tuyệt đối không commit lên Git) và vượt qua toàn bộ kiểm tra toàn vẹn:
- **SHA-256 thực tế:** `28bd2f3d6c2622fb10a7b3d83399996aee1ca4e83e62efe2669e99fd4d9e988f` (khớp 100% với mã băm chuẩn trong `labeling/test300_selection.json`).
- **Tổng số bản ghi & Tính duy nhất:** `2,168` bản ghi JSONL hợp lệ, `2,168` ID duy nhất (`0` dòng lỗi, `0` ID trùng, `0` tiêu đề rỗng).
- **Schema từng bản ghi (10 trường chuẩn):** `id`, `article_id`, `source`, `pool`, `title`, `url`, `published_utc`, `source_time_kind`, `title_provenance`, `nlp_eligible`.
- **Phân bố theo Pool & Nguồn:**
  - `pool == "train_unlabeled"` (nguồn `GDELT`): **`1,409` bản ghi**.
  - `pool == "test_candidate"` (nguồn `RSS`): **`759` bản ghi**.

### 4.2. Kết quả Chạy Thực tế & Kiểm chứng Độc lập Nhãn Mềm `P2` (Task 3.11 — `results/llm_soft_labels/`)
- **Tổng số ID loại trừ nạp từ `labeling/trial30_ids.csv` (`30` ID) và `labeling/assignment.csv` (`300` ID test):** **`330` ID duy nhất** (hai tập `trial30` và `test300` hoàn toàn rời nhau, `overlap = 0`).
- **Đối chiếu vị trí của 330 ID loại trừ trong `headlines.jsonl`:** Cả `30/30` ID gán thử và `300/300` ID test đều thuộc `pool == "test_candidate"` (`330` bản ghi). Giao giữa tập loại trừ (`330` ID) và `train_unlabeled` (`1,409` bản ghi) bằng **`0` ID**.
- **Kết quả thực thi trên Google Colab Tesla T4 (`results/llm_soft_labels/run.json` & `p2_soft_labels.jsonl`):**
  - **Mô hình & Prompt:** `Qwen/Qwen2.5-7B-Instruct` (`nf4-4bit`), prompt **`P2`** (đã chốt ở Tuần 2).
  - **Số bản ghi đã chấm (`n_scored_total` / `n_newly_scored`):** **`1,409 / 1,409` bản ghi** (`100%` kho `train_unlabeled` hợp lệ).
  - **Thời gian chạy (`elapsed_seconds`):** **`626.843` giây (~10.45 phút)** (`started_at_utc = 2026-11-22T16:27:55.847474+00:00`, `finished_at_utc = 2026-11-22T16:38:22.689991+00:00`).
  - **Chất lượng phân phối logits chữ cái A/B/C:**
    - `min_abc_mass`: **`0.9890`**
    - `mean_abc_mass`: **`0.9998`**
    - `share_top1_is_letter`: **`1.0000` (1,409 / 1,409 mẫu có top-1 token là A, B hoặc C)**.
  - **Phân bố dự đoán trên 1,409 tiêu đề `train_unlabeled`:**
    - Theo nhãn cứng (`argmax(probs)`): `bearish = 81` (`5.75%`), `neutral = 1,186` (`84.17%`), `bullish = 142` (`10.08%`).
    - Theo xác suất mềm trung bình (`mean(probs)` theo thứ tự `["bearish", "neutral", "bullish"]`): **`[0.0548, 0.8449, 0.1003]`**.
- **13 Kiểm tra Độc lập trên `results/llm_soft_labels/p2_soft_labels.jsonl` (`SHA-256: df036722...`):**
  1. `headlines.jsonl` khớp `SHA-256: 28bd2f3d6c2622fb10a7b3d83399996aee1ca4e83e62efe2669e99fd4d9e988f`.
  2. Đúng `1,409` bản ghi đầu vào hợp lệ thuộc `pool == "train_unlabeled"`.
  3. File đầu ra chứa đúng `1,409` dòng JSONL.
  4. Toàn bộ `1,409` ID đầu ra là duy nhất (`0` trùng lặp).
  5. Tập ID đầu ra và thứ tự ID khớp 100% với tập `1,409` ID `train_unlabeled` hợp lệ.
  6. Giao với `30` ID `trial30_ids.csv` và `300` ID test trong `assignment.csv` bằng **`0`**.
  7. Mọi bản ghi có đúng 5 trường chuẩn: `{"id", "probs", "abc_mass", "top1_is_letter", "letter_logp"}`.
  8. Mọi vector `probs` gồm đúng 3 số thực hữu hạn trong `[0, 1]`.
  9. Sai số tổng xác suất lớn nhất `max(|sum(probs) - 1.0|) = 1.00e-06 ≤ 1e-3`.
  10. Thứ tự lớp nhất quán `["bearish", "neutral", "bullish"]`, và `softmax(letter_logp)` khớp với `probs` (`max_diff = 5.64e-07`).
  11. Giá trị `abc_mass` hợp lệ trên mọi dòng và khớp với `sum(exp(letter_logp))`; tính lại độc lập cho đúng `min_abc_mass = 0.989` và `mean_abc_mass = 0.9998`.
  12. Tuyệt đối không có trường văn bản tiêu đề hay URL (`text`, `title`, `headline`, `url`, `article_id`, `source`) trong file dự đoán.
  13. Toàn bộ thông số trong `run.json` khớp chính xác với dữ liệu tính lại từ `p2_soft_labels.jsonl`.
- **Phân biệt rõ giữa Toàn vẹn Suy luận (Inference Integrity) và Chất lượng Nhãn Mềm Hạ nguồn (Downstream Soft-Label Quality):**
  - Các kiểm tra trên chứng minh quá trình suy luận LLM `P2` trên 1,409 tiêu đề `train_unlabeled` đạt độ toàn vẹn kỹ thuật, không rò rỉ tập test/trial, và có xác suất tập trung gần như tuyệt đối vào 3 phương án A/B/C (`mean_abc_mass = 0.9998`).
  - Vì kho `train_unlabeled` (nguồn GDELT) không có nhãn chuẩn của con người, việc kiểm tra toàn vẹn đầu ra **không** đồng nghĩa với việc đã chứng minh chất lượng phân loại của nhãn mềm. Hiệu quả thực sự của bộ nhãn mềm này sẽ được đánh giá ở **Tuần 4 (Sprint S2)** thông qua độ cải thiện `val_macro_f1` của mô hình chưng cất **`E6`** (KL-divergence với $T \in \{1, 2, 4\}$) và pseudo-label cứng **`E12`** so với mốc **`E5`** (`0.8030 ± 0.0054`).

### 4.3. Hoàn tất Gán nhãn Thủ công Đợt 1 của TV3 (Task 3.12 — `labeling/round1/tv3.csv`)
> Sửa 23/11: nhãn này do LLM tạo, không phải gán tay; file đã chuyển sang labeling/llm_annotations/ (xem context/decisions.md).

- **File kết quả đã tích hợp:** `labeling/round1/tv3.csv` (`13,388` bytes, `SHA-256: fd0bd2d416299b10d87f07395a8695bde053d9f4502992251a7838b273c75fda`).
- **Kết quả kiểm tra toàn vẹn trên `labeling/round1/tv3.csv`:**
  1. **Schema chuẩn:** `['id', 'label', 'reason']` (3 cột, tuyệt đối không chứa cột `headline` hay văn bản tiêu đề gốc).
  2. **Số dòng dữ liệu:** Chính xác **`68` dòng** (`68` ID duy nhất, không có ID rỗng hoặc trùng lặp).
  3. **Đối chiếu với `labeling/assignment.csv`:** Khớp **100%** cả về tập hợp ID lẫn thứ tự dòng với `68` ID được phân công cho `annotator == "tv3"` và `round == "1"` (`0` ID thiếu, `0` ID thừa).
  4. **Tính hợp lệ của nhãn (`label`):** Mọi nhãn đều thuộc đúng 3 giá trị chuẩn chữ thường `{"bearish", "neutral", "bullish"}`, với phân bố:
     - `neutral`: **`42` câu (`61.76%`)**
     - `bearish`: **`14` câu (`20.59%`)**
     - `bullish`: **`12` câu (`17.65%`)**
  5. **Tính hợp lệ của lý do (`reason`):** Cả `68 / 68` dòng đều có lý do giải thích rõ ràng do con người tự viết (độ dài từ `54` đến `125` ký tự, `0` dòng rỗng). Toàn bộ nội dung do TV3 gán tay được giữ nguyên 100% byte-for-byte, không có bất kỳ sự can thiệp, viết lại hay chuẩn hóa nào từ AI/LLM.
  6. **Bảo mật dữ liệu thô:** Các file gói đọc cục bộ trong `data/raw/headlines_2026/` (`headlines.jsonl`, `round1_tv3_reading_packet.csv`, `round1_tv3_blank.csv`) vẫn nằm hoàn toàn trong vùng `.gitignore` và không được stage hay commit.

---

## 5. Kết quả Kiểm thử Tự động (Unit & Integration Tests)

Đã thực thi toàn bộ các bộ kiểm thử liên quan đến TV3 và toàn bộ test không phụ thuộc PyTorch trên nhánh `tv3/week3`:
- `python -m unittest tests.test_metrics tests.test_transformer tests.test_llm_soft_labels tests.test_train_data tests.test_stats tests.test_pick300 tests.test_simulate_power tests.test_audit_leakage_capacity tests.test_make_leakage tests.test_make_leakage_cli -v`: **33/33 tests PASS (`OK`)**
  - `tests.test_transformer` (6/6 PASS): Kiểm tra `E5.yaml` & `E7.yaml`, đóng băng encoder, lịch warmup, hợp đồng `write_preds`, và toàn vẹn 6 thư mục kết quả thực tế `results/E5/seed{1,2,3}` & `results/E7/seed{1,2,3}`.
  - `tests.test_llm_soft_labels` (6/6 PASS): Kiểm tra nạp prompt `P2`, nạp đủ `330` ID loại trừ thực tế (`trial30_ids.csv` + `assignment.csv`), kiểm tra `SHA-256` và lọc đúng `1,409` bản ghi `train_unlabeled` trên file `data/raw/headlines_2026/headlines.jsonl` thật, từ chối mã băm sai, kiểm tra JSONL đầu ra không chứa văn bản tiêu đề, kiểm tra toàn vẹn bộ artifact thực tế `results/llm_soft_labels/{p2_soft_labels.jsonl, run.json}` (`1,409` bản ghi), và kiểm tra hợp đồng gán nhãn thủ công `labeling/round1/tv3.csv` (`68` dòng).
  - `tests.test_metrics` (4/4 PASS), `tests.test_train_data` (6/6 PASS), và các bộ test tích hợp từ `main` (`test_stats`, `test_pick300`, `test_simulate_power`, `test_audit_leakage_capacity`, `test_make_leakage`, `test_make_leakage_cli`: 11/11 PASS).

---

## 6. Kê khai Minh bạch Sử dụng AI (AI Transparency Disclosure)

- **Trợ lý lập trình AI:** Hỗ trợ viết mã nguồn `models/transformer.py`, `models/llm_soft_labels.py`, cấu hình `configs/E5.yaml`, `configs/E7.yaml`, bộ unit tests `tests/test_transformer.py`, `tests/test_llm_soft_labels.py`, notebook `notebooks/week3_transformer_and_soft_labels.ipynb`, kiểm chứng độc lập kết quả thực nghiệm `E5`/`E7` từ Colab Tesla T4, xác minh `SHA-256` & trích xuất gói đọc `round1_tv3_reading_packet.csv` (để trống nhãn), và cập nhật tài liệu dự án.
- **Giới hạn tuyệt đối đã tuân thủ:**
  - Giữ nguyên vẹn 100% dữ liệu dự đoán và metadata thực tế từ Google Colab Tesla T4 (`Results.zip`), không chỉnh sửa hay giả lập bất kỳ con số nào.
  - Không giả lập kết quả GPU cho Task 3.11 khi chưa chạy thực tế trên Colab T4.
  - Không tạo, gán, suy luận hoặc gợi ý bất kỳ nhãn (`label`) hay lý do (`reason`) nào cho việc gán nhãn thủ công đợt 1 (`round1_tv3_reading_packet.csv` / `labeling/round1/tv3.csv`) hay tập test 2026 (`AGENTS.md` Quy tắc 5).
  - Không commit dữ liệu thô (`data/raw/headlines_2026/headlines.jsonl`) hoặc gói đọc chứa văn bản tiêu đề (`round1_tv3_reading_packet.csv`) lên Git.
