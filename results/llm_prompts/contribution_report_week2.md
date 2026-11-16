# Báo cáo Đóng góp Tuần 2 & Kê khai Minh bạch Sử dụng AI (TV3)

Tài liệu này tổng hợp kết quả hoàn thành các nhiệm vụ **Tuần 2** của **Thành viên 3 (TV3 — Phụ trách Transformer, LLM, Chưng cất tri thức & Khung đánh giá chung)** trên nhánh `tv3/week2`, kèm theo bản kê khai minh bạch việc sử dụng trợ lý AI theo quy định tại `AGENTS.md` và Mục 14 của `docs/KeHoachChiTiet.pdf`.

---

## 1. Thông tin chung & Trạng thái Hoàn thành Tuần 2

| Hạng mục | Nội dung |
| :--- | :--- |
| **Vai trò phụ trách** | Thành viên 3 (TV3) — Transformer, LLM (`Qwen2.5-7B-Instruct` 4-bit), Chưng cất tri thức (`RQ3`), Khung đánh giá chung (`evaluate/metrics.py`) |
| **Nhánh làm việc** | `tv3/week2` |
| **Các nhiệm vụ đã hoàn thành** | Task 3.5 (Đánh giá 3 biến thể prompt `P1`, `P2`, `P3`), Task 3.6 (Kiểm tra top-10 logits & chốt prompt `P2`), Task 3.7 (Khung đánh giá chung `evaluate/metrics.py` + `tests/test_metrics.py`), Task 3.8 (Cập nhật bộ đọc split dạng ID trong `notebooks/colab_setup.ipynb`), và gán nhãn thử 30 tiêu đề năm 2026 (`labeling/trial/tv3.csv`) |

---

## 2. Kết quả Thực nghiệm Đánh giá Prompt `P1`, `P2`, `P3` trên Tập Validation (Tasks 3.5 & 3.6)

### 2.1 Cấu hình thực nghiệm
- **Mô hình:** `Qwen/Qwen2.5-7B-Instruct`
- **Lượng tử hóa:** 4-bit NormalFloat (`nf4-4bit`, `bnb_4bit_compute_dtype = float16`) trên Google Colab GPU (`Tesla T4`).
- **Tập dữ liệu đánh giá:** Toàn bộ **`1.791` mẫu `validation`** từ file split cố định `data/splits/cluster_split.jsonl` (`SHA-256: 8c2c35ea789acf5b2bd9c74c5e59fa0df325034422be42ae7b47f2e8c9c46aaa`), thứ tự nhãn chuẩn `("bearish", "neutral", "bullish")` (`0, 1, 2`).
- **Seed chọn mẫu few-shot (`P3`):** `seed = 1` (lấy đúng 1 mẫu mỗi nhãn từ tập `train`, hoàn toàn tách biệt với `validation` và `test`).

### 2.2 Bảng đối chiếu kết quả (`results/llm_prompts/top10_check.json` & kiểm chứng độc lập qua `evaluate.metrics`)

| Chỉ số | `P1` (Zero-shot ngắn gọn) | **`P2` (Zero-shot có định nghĩa nhãn — ĐƯỢC CHỌN)** | `P3` (Few-shot 3 ví dụ từ `train`) |
| :--- | :---: | :---: | :---: |
| **Số mẫu Validation (`n`)** | `1,791` | **`1,791`** | `1,791` |
| **Validation Macro-F1** | `0.7311` | **`0.7934`** | `0.7057` |
| **F1 — `bearish` (Lớp 0)** | `0.7546` | **`0.7535`** | `0.7169` |
| **F1 — `neutral` (Lớp 1)** | `0.7789` | **`0.8696`** | `0.7680` |
| **F1 — `bullish` (Lớp 2)** | `0.6597` | **`0.7572`** | `0.6323` |
| **Min ABC Mass** | `0.9933` | **`0.7903`** | `0.9984` |
| **Mean ABC Mass** | `0.9999` | **`0.9995`** | `1.0000` |
| **Share Top-1 is Letter (`A`/`B`/`C`)** | `1.0000` | **`1.0000`** | `1.0000` |
| **Thời gian chạy (`elapsed_seconds`)** | `151.47 s` | **`238.92 s`** | `249.20 s` |

### 2.3 Quyết định lựa chọn Prompt (`context/decisions.md`)
- **Quy tắc chọn:** Chọn prompt có `Macro-F1` trên tập `validation` cao nhất; nếu hòa thì ưu tiên `abc_mass` cao hơn; tuyệt đối không sử dụng tập `test`.
- **Quyết định:** Chọn **`P2`** làm prompt cố định cho toàn bộ các thí nghiệm dùng LLM của dự án (`E11` zero-shot, `E6` chưng cất nhãn mềm, `E12` nhãn cứng).
- **Lý do:** `P2` đạt **`macro_f1 = 0.7934`**, vượt xa `P1` (`+0.0623`) và `P3` (`+0.0877`), đặc biệt cải thiện mạnh khả năng phân biệt giữa lớp `neutral` (`F1 = 0.8696`) và `bullish` (`F1 = 0.7572`) nhờ đưa trực tiếp định nghĩa chuẩn của `labeling/GUIDE.md` vào câu lệnh, đồng thời vẫn duy trì `share_top1_is_letter = 1.0` và `mean_abc_mass = 0.9995`.

---

## 3. Phân định Công việc: Thành viên (TV3) thực hiện vs. Trợ lý AI hỗ trợ vs. Kiểm chứng Độc lập

| Hạng mục | Do Thành viên (TV3) trực tiếp thực hiện | Do Trợ lý AI hỗ trợ (dưới sự kiểm soát của TV3) | Bằng chứng Kiểm chứng |
| :--- | :--- | :--- | :--- |
| **Gán nhãn thử 30 tiêu đề năm 2026** | **TV3 tự đọc và gán nhãn thủ công độc lập 100% cho 30 tiêu đề vòng thử (`labeling/trial/tv3.csv`) kèm lý do ngắn, không sử dụng AI/LLM.** | Tuyệt đối không tham gia gán nhãn, gợi ý hay suy đoán nhãn cho dữ liệu gán thử. | Cohen's kappa giữa TV3 và TV1 đạt `0.849`, giữa TV3 và TV2 đạt `0.747` (`labeling/trial_agreement.json`). |
| **Chạy thực nghiệm GPU trên Google Colab (`P1`, `P2`, `P3`)** | **TV3 trực tiếp chạy `notebooks/llm_prompts.ipynb` với `Qwen/Qwen2.5-7B-Instruct` 4-bit NF4 trên 1.791 mẫu validation trên Google Colab T4 và xuất 4 file kết quả.** | Không giả lập hay chỉnh sửa bất kỳ xác suất dự đoán nào; chỉ đối chiếu mã băm và chấm điểm độc lập qua `evaluate/metrics.py`. | Chạy `python -m evaluate.metrics` trên cả 3 file `results/E11/{P1,P2,P3}/preds_validation.jsonl` cho kết quả khớp 100% từng con số `macro_f1`, `f1` và ma trận nhầm lẫn với `top10_check.json`. |
| **Phát triển `evaluate/metrics.py`, `tests/test_metrics.py` & Notebooks** | Chỉ đạo quy trình tích hợp Git, kiểm duyệt từng commit và xác nhận quy tắc chọn prompt. | Viết mã nguồn `evaluate/metrics.py`, `tests/test_metrics.py`, `configs/llm_prompts.yaml`, `notebooks/llm_prompts.ipynb`, cập nhật `notebooks/colab_setup.ipynb`, `MAP.md`, `PLAN.md`, `context/decisions.md`. | `python -m unittest tests.test_metrics -v` đạt 4/4 tests (`OK`). |

---

## 4. Kê khai Minh bạch Sử dụng AI (AI Transparency Disclosure)

- **Trợ lý lập trình AI:** Được sử dụng để hỗ trợ cài đặt module đánh giá `evaluate/metrics.py`, viết unit tests `tests/test_metrics.py`, chuẩn bị notebook `notebooks/llm_prompts.ipynb`, kiểm tra chéo kết quả dự đoán từ Colab và cập nhật tài liệu dự án (`context/decisions.md`, `MAP.md`, `PLAN.md`).
- **Mô hình ngôn ngữ lớn (`Qwen/Qwen2.5-7B-Instruct` 4-bit):** Chỉ được chạy trên tập `validation` của Twitter FNS (với 3 ví dụ few-shot của `P3` lấy từ tập `train`) để chọn prompt cố định ở Tuần 2.
- **Cam kết tuân thủ `AGENTS.md`:** Không sử dụng AI hay LLM để gán nhãn, chỉnh sửa hoặc suy đoán nhãn thật cho bất kỳ tập đánh giá hay tập test nào; không truy cập hay chạy đánh giá trên bất kỳ tập `test` nào trước Tuần 6.
