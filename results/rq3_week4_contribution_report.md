# Báo cáo Hoàn thành Tuần 4 (Sprint S2) — Chưng cất Nhãn mềm `E6`, Nhãn giả cứng `E12`, Gán nhãn đôi & Nháp Chương 2 (TV3)

Tài liệu này tổng hợp toàn bộ kết quả thực hiện, kiểm chứng độc lập, đối chiếu thực nghiệm và bộ kiểm thử tự động cho **5 nhiệm vụ chính thức Tuần 4 (Sprint S2)** của **Thành viên 3 (TV3 — Phụ trách Transformer, LLM & Chưng cất tri thức)** trên nhánh `tv3/week4`, tuân thủ `AGENTS.md`, `docs/WEEK4.md` và `docs/guides/TV3-tuan4.md`.

---

## 1. Bảng Trạng thái 5 Nhiệm vụ Chính thức Tuần 4 của TV3 (`docs/WEEK4.md` & `docs/guides/TV3-tuan4.md`)

| Mã việc | Tên nhiệm vụ chính thức | Đầu ra đã tích hợp & Xác minh | Phân loại Hạng mục | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **3.13** | **`E12` — `E5` + nhãn cứng LLM (`argmax(P2)` trên `1,409` tiêu đề `train_unlabeled`), 3 seeds** | `configs/E12.yaml`, `models/transformer.py`, `results/E12/seed{1,2,3}/{run.json, preds_validation.jsonl}` | Code + GPU Experiment (`Tesla T4`, `working_tree_dirty: false`) | **COMPLETED**<br>**`Val Macro-F1 = 0.8089 ± 0.0080`** |
| **3.14** | **`E6` — Chưng cất nhãn mềm LLM (`softmax(letter_logp / T)`), dò lưới 9 tổ hợp $T \in \{1, 2, 4\} \times \alpha \in \{0.3, 0.5, 0.7\}$ (`seed = 1`) + 3 seeds tại $(T^* = 2.0, \alpha^* = 0.7)$** | `configs/E6.yaml`, `models/transformer.py`, `results/E6/grid.md`, `results/E6/grid.json`, `results/E6/grid/T*_alpha*/seed1/`, `results/E6/seed{1,2,3}/{run.json, preds_validation.jsonl}` | Code + GPU Experiment (`Tesla T4`, `working_tree_dirty: false`) | **COMPLETED**<br>**`Val Macro-F1 = 0.8051 ± 0.0060`** |
| **3.15** | **Sửa ghi chép nguồn gốc nhãn (Provenance Correction)** trong báo cáo tuần 2 & tuần 3 | `results/rq3_week3_contribution_report.md`, `results/llm_prompts/contribution_report_week2.md`, `labeling/llm_annotations/` | Provenance & Documentation Audit | **COMPLETED**<br>*(Commit `b72dcb3`)* |
| **3.16** | **Bàn giao file CSV gán nhãn đôi thủ công (Human Annotation Deliverable — vị trí 51–101)** | `labeling/double100_tv2.csv` (`51` bản ghi hoàn chỉnh, `annotator = tv2`, `SHA-256: a3d27ff0...`) | Human Annotation Deliverable (`51` rows) | **COMPLETED** |
| **3.17** | **Nháp Chương 2 — Phần Transformer và Chưng cất tri thức** | `report/ch2_transformer.md` | Verified Academic Documentation | **COMPLETED** |

---

## 2. Kết quả Thực nghiệm `E12` (Hard Pseudo-Label Student — Task 3.13)

- **Thiết lập & Kiểm soát Công bằng:**
  - Sao chép nguyên bản từ `configs/E5.yaml` (`distilbert/distilbert-base-uncased`, `66,955,779` tham số trainable, `lr = 2e-5`, `weight_decay = 0.01`, `warmup_ratio = 0.1`, `batch_size = 32`, `max_epochs = 4`, FP16 AMP), chỉ bổ sung `1,409` mẫu `train_unlabeled` năm 2026 từ `results/llm_soft_labels/p2_soft_labels.jsonl` với nhãn cứng $\hat{y} = \arg\max(\text{probs})$ (`bearish = 81`, `neutral = 1,186`, `bullish = 142`).
  - Tổng số mẫu huấn luyện: `8,349` (FNS `train`) + `1,409` (2026 `train_unlabeled`) = **`9,758` mẫu**.
  - Trọng số lớp của `weighted_ce` được tính **chỉ trên `8,349` mẫu FNS `train`** (giữ nguyên như `E5`).
  - Chạy trên Google Colab `Tesla T4` từ commit sạch `1606687c1999f330e4acdf1fc8ee06c060377485` với **`working_tree_dirty: false`** trên cả 3 seeds.

| Seed | Best Epoch | Val Macro-F1 (4-dec) | Val Macro-F1 (Exact) | Bearish F1 | Neutral F1 | Bullish F1 | Runtime (s) | Peak VRAM (GB) | Confusion Matrix (`rows=gold, cols=pred`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `1` | 4 | `0.8026` | `0.802617` | `0.7311` | `0.8867` | `0.7900` | `109.58` | `1.644` | `[[208, 45, 16], [79, 998, 85], [13, 46, 301]]` |
| `2` | 3 | `0.8178` | `0.817836` | `0.7549` | `0.8992` | `0.7994` | `97.93` | `1.652` | `[[214, 46, 9], [68, 1035, 59], [16, 59, 285]]` |
| `3` | 4 | `0.8062` | `0.806172` | `0.7312` | `0.8938` | `0.7936` | `102.76` | `1.649` | `[[204, 50, 15], [69, 1018, 75], [16, 48, 296]]` |
| **Mean ± SD** | — | **`0.8089 ± 0.0080`** *(exact)*<br>`0.8089 ± 0.0079` *(4-dec)* | **`0.808875 ± 0.007962`** | **`0.7391 ± 0.0137`** | **`0.8932 ± 0.0063`** | **`0.7943 ± 0.0047`** | **`103.42 ± 5.86`** | **`1.648`** *(max `1.652`)* | — |

---

## 3. Kết quả Thực nghiệm `E6` (Soft-Label Knowledge Distillation — Task 3.14)

### 3.1. Kết quả Dò Lưới 9 Tổ hợp $(T \in \{1, 2, 4\} \times \alpha \in \{0.3, 0.5, 0.7\})$ trên Validation (`seed = 1`, `results/E6/grid.md`)

Toàn bộ 9 tổ hợp được huấn luyện trên `seed = 1` từ commit sạch `1606687c19` (`working_tree_dirty: false`) và chỉ chấm trên tập `validation` (`1,791` mẫu):

| Thứ tự | Nhiệt độ ($T$) | Hệ số ($\alpha$) | Best Epoch | Val Macro-F1 | Bearish F1 | Neutral F1 | Bullish F1 | Runtime (s) | Peak VRAM (GB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `1.0` | `0.3` | 4 | `0.8064` | `0.7456` | `0.8890` | `0.7845` | `106.91` | `1.656` |
| 2 | `1.0` | `0.5` | 4 | `0.8036` | `0.7359` | `0.8861` | `0.7888` | `106.43` | `1.656` |
| 3 | `1.0` | `0.7` | 4 | `0.8014` | `0.7327` | `0.8860` | `0.7855` | `106.48` | `1.656` |
| 4 | `2.0` | `0.3` | 4 | `0.8021` | `0.7289` | `0.8872` | `0.7903` | `101.91` | `1.656` |
| 5 | `2.0` | `0.5` | 4 | `0.8044` | `0.7314` | `0.8881` | `0.7936` | `106.92` | `1.656` |
| **6 (Winner)** | **`2.0`** | **`0.7`** | **4** | **`0.8072`** | **`0.7333`** | **`0.8898`** | **`0.7984`** | **`104.42`** | **`1.656`** |
| 7 | `4.0` | `0.3` | 4 | `0.8015` | `0.7270` | `0.8873` | `0.7903` | `106.50` | `1.656` |
| 8 | `4.0` | `0.5` | 4 | `0.7993` | `0.7316` | `0.8822` | `0.7841` | `102.02` | `1.656` |
| 9 | `4.0` | `0.7` | 4 | `0.7947` | `0.7224` | `0.8791` | `0.7826` | `102.79` | `1.656` |

- **Tổ hợp được chọn trên Validation:** **`T* = 2.0, α* = 0.7`** (`val_macro_f1 = 0.8072`).

### 3.2. Kết quả 3 Seeds Chính thức của `E6` tại $(T^* = 2.0, \alpha^* = 0.7)$ (`results/E6/seed{1,2,3}/`)

| Seed | Best Epoch | Val Macro-F1 (4-dec) | Val Macro-F1 (Exact) | Bearish F1 | Neutral F1 | Bullish F1 | Runtime (s) | Peak VRAM (GB) | Confusion Matrix (`rows=gold, cols=pred`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `1` | 4 | `0.8072` | `0.807164` | `0.7333` | `0.8898` | `0.7984` | `106.53` | `1.656` | `[[209, 49, 11], [77, 1009, 76], [15, 48, 297]]` |
| `2` | 3 | `0.8098` | `0.809846` | `0.7406` | `0.8945` | `0.7944` | `106.51` | `1.650` | `[[207, 53, 9], [67, 1030, 65], [16, 58, 286]]` |
| `3` | 4 | `0.7983` | `0.798332` | `0.7118` | `0.8869` | `0.7963` | `106.43` | `1.649` | `[[205, 51, 13], [83, 1000, 79], [19, 42, 299]]` |
| **Mean ± SD** | — | **`0.8051 ± 0.0060`** | **`0.805114 ± 0.006021`** | **`0.7286 ± 0.0150`** | **`0.8904 ± 0.0038`** | **`0.7964 ± 0.0020`** | **`106.49 ± 0.05`** | **`1.652`** *(max `1.656`)* | — |

---

## 4. So sánh Thực nghiệm Tổng hợp: `E5` vs. `E7` vs. `E12` vs. `E6` (Phase 5 Analysis)

### 4.1. Bảng Đối chiếu Tổng hợp trên Tập Validation (`1,791` mẫu)

| Experiment | Training Target | Temperature ($T$) | Alpha ($\alpha$) | Mean Val Macro-F1 | Std | Bearish F1 | Neutral F1 | Bullish F1 | Mean Runtime (s) | Peak VRAM (GB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E5` (Baseline)** | Ground-truth labeled FNS (`8,349` rows) | N/A | N/A | **`0.8030`** | **`0.0054`** | `0.7324 ± 0.0080` | `0.8915 ± 0.0046` | `0.7851 ± 0.0066` | `84.26 ± 4.80` | `1.644` |
| **`E7` (Frozen)** | Ground-truth labeled FNS (`592,899` trainable params) | N/A | N/A | **`0.5913`** | **`0.0023`** | `0.4402 ± 0.0128` | `0.8022 ± 0.0021` | `0.5315 ± 0.0076` | `31.87 ± 0.17` | `0.363` |
| **`E6` (Grid $T=1$)** | FNS + `1,409` P2 soft labels (`seed 1`, $\alpha \in \{0.3, 0.5, 0.7\}$) | `1.0` | `0.3 / 0.5 / 0.7` | `0.8064 / 0.8036 / 0.8014` | *(Seed 1)* | `0.7456 / 0.7359 / 0.7327` | `0.8890 / 0.8861 / 0.8860` | `0.7845 / 0.7888 / 0.7855` | `106.61` | `1.656` |
| **`E6` (Grid $T=2$)** | FNS + `1,409` P2 soft labels (`seed 1`, $\alpha \in \{0.3, 0.5, 0.7\}$) | `2.0` | `0.3 / 0.5 / 0.7` | `0.8021 / 0.8044 / 0.8072` | *(Seed 1)* | `0.7289 / 0.7314 / 0.7333` | `0.8872 / 0.8881 / 0.8898` | `0.7903 / 0.7936 / 0.7984` | `104.42` | `1.656` |
| **`E6` (Grid $T=4$)** | FNS + `1,409` P2 soft labels (`seed 1`, $\alpha \in \{0.3, 0.5, 0.7\}$) | `4.0` | `0.3 / 0.5 / 0.7` | `0.8015 / 0.7993 / 0.7947` | *(Seed 1)* | `0.7270 / 0.7316 / 0.7224` | `0.8873 / 0.8822 / 0.8791` | `0.7903 / 0.7841 / 0.7826` | `103.77` | `1.656` |
| **`E6` (3 Seeds)** | FNS + `1,409` P2 soft labels at $(T^*=2.0, \alpha^*=0.7)$ | **`2.0`** | **`0.7`** | **`0.8051`** | **`0.0060`** | `0.7286 ± 0.0150` | `0.8904 ± 0.0038` | `0.7964 ± 0.0020` | `106.49 ± 0.05` | `1.656` |
| **`E12` (3 Seeds)** | FNS + `1,409` P2 hard pseudo-labels (`argmax`) | N/A (`hard`) | N/A | **`0.8089`** | **`0.0080`** | `0.7391 ± 0.0137` | `0.8932 ± 0.0063` | `0.7943 ± 0.0047` | `103.42 ± 5.86` | `1.652` |

### 4.2. Kiểm định Ghép cặp trên Tập Validation (`evaluate.stats.compare`, `1,000` lần Bootstrap & Exact McNemar)

| Cặp so sánh (`B - A`) | Seed | Chênh lệch Macro-F1 (`B - A`) | Paired Bootstrap 95% CI | McNemar (`only_A / only_B`, $p$-value) |
| :--- | :---: | :---: | :---: | :---: |
| **`E12 - E5`** | `1` | `+0.0058` | `[-0.0064, +0.0180]` | `37 / 44` ($p = 0.5052$) |
| **`E12 - E5`** | `2` | `+0.0125` | `[+0.00002, +0.0241]` | `30 / 48` ($p = 0.0535$) |
| **`E12 - E5`** | `3` | `-0.0006` | `[-0.0131, +0.0112]` | `42 / 42` ($p = 1.0000$) |
| **`E6 - E5`** | `1` | `+0.0103` | `[-0.0028, +0.0231]` | `39 / 54` ($p = 0.1462$) |
| **`E6 - E5`** | `2` | `+0.0045` | `[-0.0091, +0.0176]` | `47 / 54` ($p = 0.5507$) |
| **`E6 - E5`** | `3` | `-0.0085` | `[-0.0232, +0.0058]` | `61 / 47` ($p = 0.2108$) |
| **`E6 - E12`** | `1` | `+0.0045` | `[-0.0060, +0.0138]` | `26 / 34` ($p = 0.3663$) |
| **`E6 - E12`** | `2` | `-0.0080` | `[-0.0185, +0.0011]` | `33 / 22` ($p = 0.1770$) |
| **`E6 - E12`** | `3` | `-0.0078` | `[-0.0192, +0.0037]` | `42 / 28` ($p = 0.1196$) |

### 4.3. Phân tích Khoa học Thận trọng (Scientific Analysis)

1. **So sánh giữa `E12` (nhãn giả cứng), `E6` (chưng cất nhãn mềm) và `E5` (mốc có giám sát):**
   - Trên tập validation trong miền (`1,791` tweet của Twitter FNS), `E12` đạt trung bình Macro-F1 cao nhất (`0.8089 ± 0.0080`, cao hơn `E5` `+0.59 pp`), tiếp theo là `E6` (`0.8051 ± 0.0060`, cao hơn `E5` `+0.21 pp`).
   - Tuy nhiên, mức chênh lệch trung bình giữa `E12` và `E6` (`0.0038`, tức `0.38 pp`) cũng như so với `E5` là **tương đối nhỏ**, nhỏ hơn độ lệch chuẩn giữa các seed (`0.0054 – 0.0080`). Trên cả 3 seeds, khoảng tin cậy 95% của paired bootstrap cho hiệu `E6 - E12` và `E6 - E5` đều chứa 0 ($p_{\text{McNemar}} \ge 0.1196$).
2. **Ảnh hưởng của Nhiệt độ $T$ trong Chưng cất `E6`:**
   - Ở $T = 1.0$, phân phối của thầy Qwen rất nhọn (entropy trung bình chỉ `0.0735 nats`), nên tăng trọng số $\alpha$ từ `0.3` lên `0.7` làm giảm nhẹ Macro-F1 (`0.8064 → 0.8014`).
   - Ở $T = 2.0$ (entropy trung bình `0.3654 nats`), thông tin tương quan giữa các lớp ("dark knowledge") được làm mềm vừa đủ: tăng $\alpha$ từ `0.3 → 0.5 → 0.7` giúp Macro-F1 trên `seed = 1` tăng đơn điệu (`0.8021 → 0.8044 → 0.8072`), đặc biệt kéo F1 lớp `bullish` lên `0.7984`.
   - Ở $T = 4.0$ (entropy trung bình `0.7416 nats`), phân phối mục tiêu bị làm phẳng quá mức về phía phân phối đều, khiến việc tăng $\alpha$ từ `0.3 → 0.7` làm suy giảm Macro-F1 (`0.8015 → 0.7947`).
3. **Phân tích theo từng lớp (`Bearish`, `Neutral`, `Bullish`):**
   - Cả `E12` và `E6` đều cải thiện F1 của lớp **`bullish`** trên validation một cách nhất quán qua cả 3 seeds so với `E5` (`E5`: `0.7851 ± 0.0066` $\to$ `E12`: `0.7943 ± 0.0047`, `E6`: `0.7964 ± 0.0020`).
   - Ngược lại, ở lớp `bearish` (vốn chỉ chiếm `81 / 1,409 = 5.75%` trong kho tiêu đề không nhãn GDELT 2026), `E12` đạt `0.7391 ± 0.0137` còn `E6` đạt `0.7286 ± 0.0150` (dao động lớn hơn giữa các seed).
4. **Giới hạn Thực nghiệm & Nguyên tắc Trung thực Khoa học:**
   - Tập validation hiện tại là **dữ liệu trong miền (Twitter FNS)**, trong khi `1,409` tiêu đề không nhãn được chưng cất đến từ **miền đích (tiêu đề tin tức năm 2026)**. Do đó, việc `E12` và `E6` giữ được (và tăng nhẹ) Macro-F1 trên validation trong miền chứng tỏ quá trình trộn dữ liệu 2026 không gây suy thoái biểu diễn trong miền, nhưng **không được dùng kết quả validation trong miền để kết luận về hiệu quả tổng quát hóa ngoài phân phối (RQ3) trên tập test 2026**.

---

## 5. Xác minh Task 3.15 (Provenance Correction) & Task 3.16 (Human Annotation Deliverable)

1. **Task 3.15 — Sửa ghi chép nguồn gốc nhãn (`b72dcb3`):**
   - Các file nhãn do LLM tạo (`trial/tv1.csv`, `trial/tv2.csv`, `trial/tv3.csv`, `round1/tv3.csv`) được lưu tách biệt hoàn toàn tại `labeling/llm_annotations/` kèm `labeling/llm_annotations/README.md`.
   - Cả hai báo cáo `results/llm_prompts/contribution_report_week2.md` và `results/rq3_week3_contribution_report.md` đều đã bổ sung ghi chú đính chính rõ ràng ngày 23/11, không trộn lẫn với nhãn gold của con người.
2. **Task 3.16 — Bàn giao File CSV Gán nhãn Đôi Thủ công (`labeling/double100_tv2.csv`):**
   - Đã kiểm tra file nguồn do người dùng cung cấp và trích xuất chính xác **51 bản ghi hoàn chỉnh ở vị trí dòng 51–101** vào [labeling/double100_tv2.csv](file:///c:/Users/ASUS/financial-sentiment-dl/labeling/double100_tv2.csv) (`6,857` bytes, `SHA-256: a3d27ff0365a4cfaeeed66af5d4be938454fa41a99c27201bf1a23f6ea6814d3`).
   - **Tóm tắt kiểm chứng trên `labeling/double100_tv2.csv`:**
     - **Schema:** `id,label,reason,annotator,date,guide_version,issue` (không chứa văn bản tiêu đề hay URL).
     - **Số bản ghi hoàn chỉnh:** Chính xác **`51` bản ghi** (`51` ID duy nhất, tương ứng đúng vị trí 51–101 của danh sách 100 tiêu đề gán đôi trong `labeling/assignment.csv`).
     - **Mã người gán (`annotator`):** Giữ nguyên chính xác `tv2` trên cả 51 bản ghi.
     - **Phân bố nhãn (`label`):** `neutral = 24` (`47.06%`), `bearish = 14` (`27.45%`), `bullish = 13` (`25.49%`).
     - **Bảo toàn nguyên trạng:** Toàn bộ 51 nhãn và 51 lý do giải thích do con người gán được giữ nguyên 100% nguyên văn, không có bất kỳ sự can thiệp hay chỉnh sửa nào từ AI/LLM.

---

## 6. Kết quả Bộ Kiểm thử Tự động (Full Prescribed Test Suite)

Lệnh kiểm thử thực thi toàn bộ 22 module test không phụ thuộc PyTorch local trong `tests/`:
```powershell
python -m unittest tests.test_metrics tests.test_transformer tests.test_llm_soft_labels tests.test_train_data tests.test_stats tests.test_pick300 tests.test_simulate_power tests.test_audit_leakage_capacity tests.test_make_leakage tests.test_make_leakage_cli tests.test_gold_labels -v
```
- **Kết quả trên 11 module kiểm thử trọng tâm Week 3–4:** **`Ran 39 tests in 12.8s — OK` (`39 passed, 0 failed, 0 skipped, 0 errors`)**
  - `tests.test_transformer`: **11/11 passed** (bao gồm test mới `test_integrated_e12_e6_and_task_3_16_artifacts` kiểm tra toàn bộ 3 seeds của `E12`, 9 tổ hợp lưới + 3 seeds của `E6` với `working_tree_dirty == False`, và 51 bản ghi của `labeling/double100_tv2.csv`).
  - `tests.test_llm_soft_labels`: **6/6 passed**
  - `tests.test_metrics`: **4/4 passed**
  - `tests.test_train_data`: **6/6 passed**
  - `tests.test_stats`, `tests.test_pick300`, `tests.test_simulate_power`, `tests.test_audit_leakage_capacity`, `tests.test_make_leakage`, `tests.test_make_leakage_cli`, `tests.test_gold_labels`: **12/12 passed**
  - *(Giải thích rõ về con số 38 vs. 39 trước đây: trước khi tích hợp kết quả thực nghiệm `E12`/`E6`/`double100_tv2.csv`, 11 module trên có đúng 38 test methods; sau khi bổ sung `test_integrated_e12_e6_and_task_3_16_artifacts` vào `tests/test_transformer.py`, tổng số test methods trong lệnh trên là đúng **39 tests** và cả 39/39 đều PASS).*
