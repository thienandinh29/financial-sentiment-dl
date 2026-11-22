# Báo cáo Đóng góp Tuần 3 (Sprint S1) & Kê khai Minh bạch Sử dụng AI (TV3)

Tài liệu này ghi nhận kết quả triển khai mã nguồn, cấu hình thực nghiệm, bộ kiểm thử tự động và trạng thái các nhiệm vụ **Tuần 3 (Sprint S1)** của **Thành viên 3 (TV3 — Phụ trách Transformer, LLM & Chưng cất tri thức)** trên nhánh `tv3/week3`, tuân thủ quy định tại `AGENTS.md`, `docs/WEEK3.md` và `docs/KeHoachChiTiet.pdf`.

---

## 1. Bảng Phân rã Nhiệm vụ Tuần 3 của TV3 (`docs/WEEK3.md`)

| Mã việc | Tên nhiệm vụ | Đầu ra yêu cầu | Trạng thái hiện tại |
| :---: | :--- | :--- | :--- |
| **3.9** | **`E5` — DistilBERT fine-tune toàn bộ (3 seeds)** | `configs/E5.yaml`, `models/transformer.py`, `tests/test_transformer.py`, `results/E5/seed{1,2,3}/{run.json, preds_validation.jsonl}` | **Đã hoàn thành 100% mã nguồn, cấu hình & unit tests.**<br>*(Chờ chạy GPU trên Google Colab T4 qua `notebooks/week3_transformer_and_soft_labels.ipynb` để sinh 3 file `run.json` và `preds_validation.jsonl` thực tế).* |
| **3.10** | **`E7` — DistilBERT đóng băng encoder (3 seeds)** | `configs/E7.yaml`, `models/transformer.py`, `tests/test_transformer.py`, `results/E7/seed{1,2,3}/{run.json, preds_validation.jsonl}` | **Đã hoàn thành 100% mã nguồn, cấu hình & unit tests** (copy nguyên bản từ `E5.yaml`, chỉ đổi duy nhất `freeze_encoder: true`).<br>*(Chờ chạy GPU trên Google Colab T4 để thu thập 3 seeds thực tế).* |
| **3.11** | **Sinh nhãn mềm LLM với prompt `P2` trên kho `train_unlabeled` 2026 (bắt đầu)** | `models/llm_soft_labels.py`, `tests/test_llm_soft_labels.py`, `results/llm_soft_labels/{p2_soft_labels.jsonl, run.json}` | **Đã hoàn thành 100% pipeline có chốt chặn chống rò rỉ (`pool == "train_unlabeled"`, loại trừ `trial30_ids.csv` và `test300_ids.csv`), hỗ trợ chạy tiếp (resume) và unit tests.**<br>*(Chờ file `data/raw/headlines_2026/headlines.jsonl` trên Colab T4 để chạy sinh nhãn mềm thực tế).* |
| **3.12** | **Gán nhãn thủ công đợt 1 (~70 tiêu đề)** | `labeling/round1/tv3.csv` | **Chờ phân công (`labeling/assignment.csv` từ PR #16 của TV1) & TV3 tự gán tay.**<br>*(Tuân thủ `AGENTS.md` Quy tắc 5: quy trình human-only, tuyệt đối không dùng AI/LLM để gán nhãn hay gợi ý nhãn).* |

---

## 2. Chi tiết Kỹ thuật & Lý do Thiết kế (Tasks 3.9, 3.10, 3.11)

1. **Thiết kế `models/transformer.py` tương thích hoàn toàn với `train/*` và `evaluate/*` mà không sửa code của TV1/TV2:**
   - Tái sử dụng trực tiếp `train.data.load_split_data` (kiểm tra bắt buộc `split_sha256 = 8c2c35ea789acf5b2bd9c74c5e59fa0df325034422be42ae7b47f2e8c9c46aaa` và thứ tự nhãn chuẩn `LABELS = ("bearish", "neutral", "bullish")`), `train.losses.build_loss` (`weighted_ce` tính trọng số lớp chỉ trên tập `train`), `train.loop.set_seed`, và `train.runlog.run_metadata`.
   - Cài đặt đầy đủ siêu tham số chuẩn cho Transformer theo Mục 7 của `docs/KeHoachChiTiet.pdf`: `distilbert/distilbert-base-uncased`, `AdamW` (`lr = 2e-5`, `weight_decay = 0.01` tách riêng bias/LayerNorm), lịch trình `warmup_ratio = 0.1` (tăng tuyến tính 10% số bước đầu rồi giảm tuyến tính về 0), `batch_size = 32`, `max_epochs = 4`, `mixed_precision = true` (FP16 AMP), early stopping theo `val_macro_f1` (`patience = 3`) và khôi phục trọng số của `best_epoch`.
2. **Đảm bảo Quy tắc "Một thay đổi mỗi thí nghiệm" (`AGENTS.md` Quy tắc 4) giữa `E5` và `E7`:**
   - File `configs/E7.yaml` được sao chép từ `configs/E5.yaml` và chỉ thay đổi đúng 1 tham số: `model.freeze_encoder: true` (so với `false` ở `E5`).
   - Khi `freeze_encoder: true`, hàm `configure_encoder_freezing()` đóng băng toàn bộ 6 tầng Transformer encoder (`requires_grad = False`), chỉ giữ lại các tham số thuộc tầng phân loại (`pre_classifier`, `classifier`) để huấn luyện, đồng thời ghi nhận `total_params`, `trainable_params`, `frozen_params` vào `run.json`.
3. **Chốt chặn Chống Rò rỉ Dữ liệu trong `models/llm_soft_labels.py` (Task 3.11):**
   - Sử dụng cố định prompt **`P2`** (đã chốt ở Tuần 2 với `val_macro_f1 = 0.7934`).
   - Hàm `load_unlabeled_2026_pool()` chỉ giữ các bản ghi có `pool == "train_unlabeled"` (loại bỏ hoàn toàn nguồn RSS `test_candidate`) và tự động đọc `labeling/trial30_ids.csv` cùng `labeling/test300_ids.csv` để loại bỏ mọi ID thuộc tập gán thử hoặc tập test 2026.
   - Hàm `validate_soft_label_row()` kiểm tra nghiêm ngặt không cho phép bất kỳ trường văn bản hay URL nào (`text`, `title`, `headline`, `url`) xuất hiện trong `results/llm_soft_labels/p2_soft_labels.jsonl`.
   - Lưu đồng thời `probs` (xác suất chuẩn hóa ở $T=1$, tương thích 100% với `evaluate.metrics.read_preds`) và `letter_logp` (log-prob của 3 phương án `A`, `B`, `C` trước khi chia nhiệt độ $T$) để phục vụ trực tiếp cho cả `E12` (nhãn cứng) và `E6` (chưng cất nhãn mềm với nhiệt độ $T \in \{1, 2, 4\}$ ở Tuần 4).

---

## 3. Kết quả Kiểm thử Tự động (Unit & Integration Tests)

Đã thực thi toàn bộ các bộ kiểm thử liên quan đến TV3 trên nhánh `tv3/week3`:
- `python -m unittest tests.test_transformer -v`: **5/5 tests PASS (`OK`)**
  - Kiểm tra `E5.yaml` và `E7.yaml` hợp lệ và chỉ khác nhau đúng 1 yếu tố `freeze_encoder`.
  - Kiểm tra đóng băng encoder (`E5`: 100% tham số trainable; `E7`: chỉ `592,899` tham số của `pre_classifier` + `classifier` trainable, đóng băng toàn bộ encoder).
  - Kiểm tra công thức `linear_warmup_lr_multiplier`.
  - Kiểm tra từ chối config thiếu `split_sha256` hoặc dùng metric trên tập test.
  - Kiểm tra tương thích đầu ra `write_preds` với `evaluate.metrics.read_preds` và `score`.
- `python -m unittest tests.test_llm_soft_labels -v`: **4/4 tests PASS (`OK`)**
  - Kiểm tra nạp đúng prompt `P2` và 30 ID loại trừ thật từ `labeling/trial30_ids.csv`.
  - Kiểm tra bộ lọc loại bỏ `test_candidate`, `trial30_ids.csv` và `test300_ids.csv`.
  - Kiểm tra ghi file JSONL không chứa văn bản tiêu đề, tương thích `read_preds()`, và tính năng chạy tiếp (resume) từ bản ghi dở dang.
  - Kiểm tra từ chối bản ghi có chứa văn bản tiêu đề hoặc xác suất không hợp lệ.
- `python -m unittest tests.test_metrics -v`: **4/4 tests PASS (`OK`)**

---

## 4. Phân biệt Rõ: Phần Đã Triển khai & Kiểm thử vs. Thí nghiệm GPU Chờ Chạy Thực tế

Tuân thủ nguyên tắc trung thực thực nghiệm (`AGENTS.md` Quy tắc 6 — không bao giờ giả lập kết quả chưa chạy):
- **Đã hoàn thành trên nhánh `tv3/week3`:** Toàn bộ mã nguồn (`models/transformer.py`, `models/llm_soft_labels.py`), cấu hình (`configs/E5.yaml`, `configs/E7.yaml`), bộ kiểm thử (`tests/test_transformer.py`, `tests/test_llm_soft_labels.py`) và notebook chạy trên Colab T4 (`notebooks/week3_transformer_and_soft_labels.ipynb`).
- **Chờ chạy thực tế trên Google Colab GPU T4 / Con người thực hiện:**
  1. Chạy `E5` (3 seeds) và `E7` (3 seeds) trên Google Colab T4 bằng `notebooks/week3_transformer_and_soft_labels.ipynb` để sinh `results/E5/seed{1,2,3}/{run.json, preds_validation.jsonl}` và `results/E7/seed{1,2,3}/{run.json, preds_validation.jsonl}`.
  2. Chạy `models.llm_soft_labels` trên Colab T4 với file `data/raw/headlines_2026/headlines.jsonl` để sinh `results/llm_soft_labels/p2_soft_labels.jsonl` và `results/llm_soft_labels/run.json`.
  3. TV3 tự gán nhãn thủ công đợt 1 (~70 tiêu đề) vào `labeling/round1/tv3.csv` sau khi nhận phân công từ TV1 (PR #16).

---

## 5. Kê khai Minh bạch Sử dụng AI (AI Transparency Disclosure)

- **Trợ lý lập trình AI:** Hỗ trợ viết mã nguồn `models/transformer.py`, `models/llm_soft_labels.py`, cấu hình `configs/E5.yaml`, `configs/E7.yaml`, bộ unit tests `tests/test_transformer.py`, `tests/test_llm_soft_labels.py`, notebook `notebooks/week3_transformer_and_soft_labels.ipynb` và cập nhật tài liệu dự án.
- **Giới hạn tuyệt đối đã tuân thủ:**
  - Không tự tạo, giả lập hay điền trước bất kỳ con số kết quả thực nghiệm nào cho `E5`, `E7` hay nhãn mềm LLM khi chưa chạy thực tế trên GPU.
  - Không tạo, gán hoặc gợi ý bất kỳ nhãn nào cho việc gán nhãn thủ công đợt 1 (`labeling/round1/tv3.csv`) hay tập test 2026.
  - Không truy cập hoặc đánh giá trên bất kỳ tập `test` nào.
