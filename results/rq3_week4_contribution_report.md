# Báo cáo Đóng góp Tuần 4 (Sprint S2) — Chưng cất Nhãn mềm `E6`, Nhãn giả cứng `E12` & Nháp Chương 2 (TV3)

Tài liệu này ghi nhận kết quả kiểm toán đặc tả Tuần 4 (`docs/WEEK4.md`, `docs/guides/TV3-tuan4.md`), kiểm chứng kho nhãn mềm `P2` (`results/llm_soft_labels/`), triển khai mã nguồn và cấu hình cho **`E12` (Hard Pseudo-Label Student)** và **`E6` (Soft-Label Knowledge Distillation với $T \in \{1, 2, 4\} \times \alpha \in \{0.3, 0.5, 0.7\}$)**, cập nhật ghi chép gán nhãn (**Task 3.15**), bản nháp Chương 2 phần Transformer & Chưng cất (**Task 3.17** — `report/ch2_transformer.md`), và trạng thái các nhiệm vụ của **Thành viên 3 (TV3)** trên nhánh `tv3/week4`.

---

## 1. Bảng Phân rã Nhiệm vụ Chính thức Tuần 4 của TV3 (`docs/WEEK4.md` & `docs/guides/TV3-tuan4.md`)

| Mã việc | Tên nhiệm vụ chính thức | Đầu ra yêu cầu | Trạng thái (`Status`) |
| :---: | :--- | :--- | :---: |
| **3.15** | **Sửa ghi chép gán nhãn trong báo cáo tuần 2 & tuần 3** (ghi rõ nhãn gán thử 30 và đợt 1 do LLM tạo, đã chuyển sang `labeling/llm_annotations/`) | `results/rq3_week3_contribution_report.md`, `results/llm_prompts/contribution_report_week2.md` | **COMPLETED**<br>*(Commit `b72dcb3`: `docs(tv3): correct labeling records (LLM-made)`)* |
| **3.13** | **`E12` — `E5` + nhãn cứng LLM (`argmax(P2)` trên `1,409` tiêu đề `train_unlabeled`), 3 seeds** (`working_tree_dirty: false`) | `configs/E12.yaml`, `models/transformer.py`, `tests/test_transformer.py`, `results/E12/seed{1,2,3}/{run.json, preds_validation.jsonl}` | **CODE & TESTS COMPLETED / GPU EXECUTION PENDING ON COLAB T4**<br>*(Đã commit & push config + code để chạy từ commit sạch `working_tree_dirty: false` qua `notebooks/week4_e12_and_e6_distillation.ipynb`)* |
| **3.14** | **`E6` — Chưng cất nhãn mềm LLM (`softmax(letter_logp / T)`), dò lưới $T \in \{1, 2, 4\} \times \alpha \in \{0.3, 0.5, 0.7\}$ trên validation (`seed = 1`) + chạy 3 seeds ở $(T^*, \alpha^*)$** | `configs/E6.yaml`, `models/transformer.py`, `tests/test_transformer.py`, `results/E6/grid.md`, `results/E6/seed{1,2,3}/{run.json, preds_validation.jsonl}` | **CODE & TESTS COMPLETED / GPU EXECUTION PENDING ON COLAB T4**<br>*(Đã cài đặt `softmax_temperature`, `distillation_kl_loss`, `run_e6_grid_search` có hỗ trợ resume từng ô lưới trên Colab T4)* |
| **3.16** | **Gán đôi trực tiếp (~50 tiêu đề trong 100 tiêu đề gán đôi chưa từng đưa qua LLM)** | Gán tay trực tiếp trong buổi gọi bật camera (~45 phút) cùng TV2, không dùng AI | **NOT STARTED (HUMAN-ONLY LIVE SESSION)**<br>*(Yêu cầu TV3 hẹn giờ và thực hiện trực tiếp với TV2 theo `docs/guides/TV3-tuan4.md` Mục 3.16)* |
| **3.17** | **Nháp Chương 2 — Phần Transformer và Chưng cất tri thức** (2–3 trang tiếng Việt) | `report/ch2_transformer.md` | **COMPLETED**<br>*(Đã hoàn thành bản nháp đầy đủ công thức, kiến trúc DistilBERT, so sánh `E5`/`E7`, logit `P2` và chưng cất `E12`/`E6`)* |

---

## 2. Kiểm toán Toàn vẹn Dữ liệu Đầu vào & Mốc So sánh (`Phase 1 Audit`)

1. **Mốc so sánh `E5` (DistilBERT full fine-tune trên `data/splits/cluster_split.jsonl`, `SHA-256: 8c2c35ea...`):**
   - `Validation Macro-F1` (3 seeds `[1, 2, 3]` trên `1,791` mẫu validation): **`0.8030 ± 0.0054`** (`seed1 = 0.7968`, `seed2 = 0.8053`, `seed3 = 0.8068`).
   - `Per-class F1`: `bearish = 0.7324 ± 0.0080`, `neutral = 0.8915 ± 0.0046`, `bullish = 0.7851 ± 0.0066`.
   - `Trainable parameters`: `66,955,779 / 66,955,779` (`100%`), `Runtime`: `84.26 ± 4.80 s`, `Peak VRAM`: `1.642 GB` trên `Tesla T4`.
2. **Kiểm chứng Artifact Nhãn mềm `P2` (`results/llm_soft_labels/p2_soft_labels.jsonl`, `SHA-256: df036722...`):**
   - Tổng số bản ghi: **`1,409` ID duy nhất**, khớp 100% với `1,409` bản ghi `pool == "train_unlabeled"` trong `data/raw/headlines_2026/headlines.jsonl` (`SHA-256: 28bd2f3d6c2622fb10a7b3d83399996aee1ca4e83e62efe2669e99fd4d9e988f`).
   - **Kiểm tra chống rò rỉ (Leakage Audit):**
     - Giao với `30` ID gán thử (`labeling/trial30_ids.csv`): **`0` ID**.
     - Giao với `300` ID tập test 2026 (`labeling/assignment.csv`): **`0` ID**.
     - Giao với `11,931` ID của Twitter FNS (`train`, `validation`, `test` trong `data/splits/cluster_split.jsonl`): **`0` ID**.
   - **Thống kê Nhãn giả cứng (`E12` — `argmax(probs)`) và Độ tự tin (`max(probs)`):**
     - Số mẫu giữ lại (`n_retained`): **`1,409`** (`n_excluded_by_threshold = 0`; đặc tả chính thức tại `docs/guides/TV3-tuan4.md` Mục 3.13 giữ toàn bộ `1,409` mẫu, không áp ngưỡng cắt confidence).
     - Phân bố lớp nhãn cứng (`hard_class_counts`): `bearish = 81` (`5.75%`), `neutral = 1,186` (`84.17%`), `bullish = 142` (`10.08%`).
     - Phân bố độ tự tin $\max_c p_c$ ở $T = 1$: `min = 0.4851`, `mean = 0.9693`, `median = 0.9993`, `max = 1.0000`.
   - **Ảnh hưởng của Nhiệt độ $T \in \{1, 2, 4\}$ lên Phân phối Mục tiêu $\text{target}_T = \mathrm{softmax}(\text{letter\_logp} / T)$ (`E6`):**
     - Ở $T = 1.0$: Entropy trung bình $= \mathbf{0.0735\text{ nats}}$ (phân phối gần như one-hot).
     - Ở $T = 2.0$: Entropy trung bình $= \mathbf{0.3654\text{ nats}}$ (làm nổi bật xác suất lớp thứ cấp ở các tiêu đề biên).
     - Ở $T = 4.0$: Entropy trung bình $= \mathbf{0.7416\text{ nats}}$ (tiệm cận phân phối đều $\ln 3 \approx 1.0986\text{ nats}$).

---

## 3. Thiết kế Kỹ thuật & Công thức Toán học cho `E12` và `E6` (`models/transformer.py`)

1. **Giữ nguyên trọng số lớp của `E5` (Fairness & Single-Factor Rule):**
   - Trong `run_transformer_seed()`, hàm mất mát `loss_fn = build_loss(tcfg["loss"], [y for _, _, y in fns_train_ex], ...)` luôn được khởi tạo **trước** khi ghép `1,409` mẫu nhãn giả 2026 vào tập train. Nhờ đó, vector trọng số lớp $w_c = N_{\text{FNS}} / (3 N_{\text{FNS}, c})$ của `E12` và `E6` giống hệt `E5` tới từng chữ số thập phân, không bị làm lệch bởi tỷ lệ `84.17% neutral` của kho GDELT 2026.
2. **Hàm mất mát Chưng cất `E6` (`compute_batch_loss` & `distillation_kl_loss`):**
   - Với mỗi mẫu nhãn giả $j$ có vector log-prob thầy $z_{t,j} = \text{letter\_logp}_j$ và vector logit học trò $z_{s,j}$, phân phối mục tiêu ở nhiệt độ $T$ là $p_{t,j}^{(T)} = \mathrm{softmax}(z_{t,j} / T)$ và phân phối học trò là $p_{s,j}^{(T)} = \mathrm{softmax}(z_{s,j} / T)$.
   - Độ phân kỳ KL trên các mẫu nhãn giả trong batch được tính bằng `F.kl_div(F.log_softmax(z_s / T, dim=-1), p_t^{(T)}, reduction="batchmean")` và nhân hệ số $\alpha \cdot T^2$:
     $$\mathcal{L}_{\text{pseudo}} = \alpha \cdot T^2 \cdot \mathrm{KL}\!\left(p_t^{(T)} \,\middle\|\, p_s^{(T)}\right).$$
   - Trong một mini-batch gồm $n_{\text{FNS}}$ mẫu FNS và $n_{\text{pseudo}}$ mẫu 2026 ($N_B = n_{\text{FNS}} + n_{\text{pseudo}}$):
     $$\mathcal{L}_{\text{batch}} = \frac{n_{\text{FNS}}}{N_B}\,\mathcal{L}_{\text{WCE}}(\text{FNS}) + \frac{n_{\text{pseudo}}}{N_B}\,\mathcal{L}_{\text{pseudo}}(\text{2026}).$$
     Khi $\alpha = 0$, thành phần $\mathcal{L}_{\text{pseudo}}$ bằng đúng $0.0$.

---

## 4. Bảng Tổng hợp So sánh `E5`, `E7`, `E12`, `E6` trên Validation (`1,791` mẫu)

*(Tuân thủ `AGENTS.md` Quy tắc 6: tuyệt đối không điền số giả lập cho các lần chạy GPU đang chờ thực thi trên Google Colab T4).*

| Experiment | Training Target | Temperature ($T$) | Alpha ($\alpha$) | Mean Val Macro-F1 | Std | Runtime (s) | Peak VRAM (GB) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E5`** | Ground-truth FNS train (`8,351` rows) | N/A | N/A | **`0.8030`** | **`0.0054`** | `84.26 ± 4.80` | `1.642` | **COMPLETED** |
| **`E7`** | Ground-truth FNS train (Frozen encoder) | N/A | N/A | **`0.5913`** | **`0.0023`** | `31.87 ± 0.17` | `0.363` | **COMPLETED** |
| **`E12`** | FNS train + `1,409` P2 hard pseudo-labels (`argmax`) | N/A (`hard`) | N/A | *(Pending Colab T4 run)* | *(Pending)* | *(Pending)* | *(Pending)* | **READY TO RUN** |
| **`E6` (Grid $T=1$)** | FNS train + `1,409` P2 soft labels | `1.0` | `{0.3, 0.5, 0.7}` | *(Pending Colab T4 grid)* | *(Seed 1 grid)* | *(Pending)* | *(Pending)* | **READY TO RUN** |
| **`E6` (Grid $T=2$)** | FNS train + `1,409` P2 soft labels | `2.0` | `{0.3, 0.5, 0.7}` | *(Pending Colab T4 grid)* | *(Seed 1 grid)* | *(Pending)* | *(Pending)* | **READY TO RUN** |
| **`E6` (Grid $T=4$)** | FNS train + `1,409` P2 soft labels | `4.0` | `{0.3, 0.5, 0.7}` | *(Pending Colab T4 grid)* | *(Seed 1 grid)* | *(Pending)* | *(Pending)* | **READY TO RUN** |
| **`E6` (3 Seeds)** | FNS train + `1,409` P2 soft labels at $(T^*, \alpha^*)$ | $T^*$ (from val grid) | $\alpha^*$ (from val grid) | *(Pending Colab T4 run)* | *(Pending)* | *(Pending)* | *(Pending)* | **READY TO RUN** |

---

## 5. Kết quả Kiểm thử Tự động (Unit & Integration Tests)

Đã thực thi bộ kiểm thử trên nhánh `tv3/week4`:
```powershell
python -m unittest tests.test_metrics tests.test_transformer tests.test_llm_soft_labels tests.test_train_data tests.test_stats tests.test_pick300 tests.test_simulate_power tests.test_audit_leakage_capacity tests.test_make_leakage tests.test_make_leakage_cli tests.test_gold_labels -v
```
- **Kết quả:** **`38/38 tests PASS (`OK`)`**
  - `tests.test_transformer` (**10/10 PASS**):
    - Kiểm tra `E5.yaml`, `E6.yaml`, `E7.yaml`, `E12.yaml` hợp lệ và tuân thủ quy tắc "một thay đổi mỗi thí nghiệm".
    - Kiểm tra tính toán tay `softmax_temperature(letter_logp, T=2.0)` và `distillation_kl_loss` ($\alpha = 0 \implies \text{loss} = 0$; $z_s = z_t \implies \text{KL} = 0$; kiểm tra hệ số $\alpha \cdot T^2$ ở $T = 2, \alpha = 0.5$).
    - Kiểm tra `load_pseudo_labeled_examples` ở cả hai chế độ `hard` (`argmax`) và `soft` (`target_T`), kiểm tra chốt chặn từ chối ID trùng, từ chối 330 ID loại trừ (`trial30` + `test300`), từ chối ID thuộc tập FNS, và tích hợp thực tế toàn bộ `1,409` bản ghi từ `results/llm_soft_labels/p2_soft_labels.jsonl`.
    - Kiểm tra xuất bảng Markdown `format_e6_grid_markdown` cho lưới 9 tổ hợp của `E6`.
    - Kiểm tra toàn vẹn 6 bộ kết quả `results/E5/seed{1,2,3}` và `results/E7/seed{1,2,3}`.
  - `tests.test_llm_soft_labels` (**6/6 PASS**): Kiểm tra toàn vẹn `results/llm_soft_labels/p2_soft_labels.jsonl` (`1,409` bản ghi) và `labeling/llm_annotations/round1/tv3.csv`.
  - Các bộ test dùng chung của nhóm (`test_metrics`, `test_train_data`, `test_stats`, `test_pick300`, `test_simulate_power`, `test_audit_leakage_capacity`, `test_make_leakage`, `test_make_leakage_cli`, `test_gold_labels`): **22/22 PASS**.
