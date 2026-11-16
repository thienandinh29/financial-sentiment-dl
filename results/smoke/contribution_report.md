# Báo cáo Đóng góp Tuần 1 & Kê khai Minh bạch Sử dụng AI (TV3)

Tài liệu này ghi nhận kết quả thực hiện công việc **Tuần 1** của **Thành viên 3 (TV3 — Phụ trách Transformer, LLM, Chưng cất tri thức & Demo)** trên nhánh `tv3/colab-smoke`, kèm theo bản kê khai minh bạch việc sử dụng trợ lý AI theo quy định tại `AGENTS.md`, `README.md` và Mục 14 của `docs/KeHoachChiTiet.pdf`.

---

## 1. Thông tin thành viên & Các mục cần xác nhận

*(Lưu ý: Theo nguyên tắc trung thực, các thông tin định danh cá nhân chưa có trong repository được đánh dấu `[CẦN XÁC NHẬN]` để bạn điền chính thức trước khi nộp báo cáo cuối kỳ).*

| Trường thông tin | Nội dung |
| :--- | :--- |
| **Vai trò trong nhóm** | Thành viên 3 (TV3) — Transformer (`DistilBERT`, `FinBERT`), LLM (`Qwen2.5-7B` 4-bit), Chưng cất tri thức (`RQ3`), ONNX & Demo Gradio |
| **Họ và tên / MSSV** | `[CẦN XÁC NHẬN: Điền Họ tên và MSSV của TV3]` |
| **Nhánh làm việc Tuần 1** | `tv3/colab-smoke` (tách từ `origin/main` tại commit `e4fd2cbc06ae0a78fc947cbed1a8fa3f0a46cb7b`) |
| **Tỷ lệ đóng góp trong nhóm** | `[CẦN XÁC NHẬN: Thống nhất bảng tỷ lệ đóng góp chung giữa TV1, TV2, TV3 tại buổi họp checkpoint]` |
| **Biên bản họp CP1 / CP2** | `[CẦN XÁC NHẬN: Ghi nhận danh sách thành viên có mặt và quyết định duyệt đề xuất tại CP2]` |

---

## 2. Các công việc Tuần 1 đã hoàn thành thực tế (Tasks 3.1 – 3.4)

Đối chiếu với `PLAN.md`, `docs/WEEK1.md` và `docs/guides/TV3-tuan1.md`, các hạng mục sau đã được hoàn thành và kiểm chứng:

1. **Task 3.1 — Thiết lập Notebook Google Colab & Kiểm tra môi trường (`notebooks/colab_setup.ipynb`, `results/smoke/env.json`):**
   - Xây dựng notebook thiết lập môi trường trên Google Colab (`Tesla T4` GPU) với cơ chế nạp `HF_TOKEN` và `GH_TOKEN` an toàn qua **Colab Secrets** (`google.colab.userdata`).
   - Sử dụng header HTTP xác thực tạm thời trong môi trường tiến trình con (`GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_0` / `GIT_CONFIG_VALUE_0`) khi `git clone`, đảm bảo `GH_TOKEN` không xuất hiện trong URL, `.git/config`, cell output hay log.
   - Kiểm tra sự tồn tại và khả năng cài đặt của `requirements.lock`; tự động ghi nhận trạng thái (`requirements_lock_exists: false` trên nhánh gốc, môi trường Colab chạy Python `3.13.16`) và fallback sang `requirements.txt`.
   - Ghi nhận đầy đủ thông tin phần cứng/phần mềm thực tế vào `results/smoke/env.json`: GPU `Tesla T4` (`15.6 GB` VRAM), CUDA `13.0`, PyTorch `2.11.0+cu130`, Transformers `5.21.0`, Datasets `4.8.5`, Scikit-learn `1.6.1`, NumPy `2.1.3`.

2. **Task 3.2 — Smoke Test huấn luyện `DistilBERT` (`results/smoke/distilbert.json`):**
   - Cài đặt cơ chế kiểm tra bắt buộc mã băm SHA-256 (`split_sha256` trong `configs/E2.yaml`) nếu file split chính thức tồn tại, và fallback sang nạp tập `train`/`validation` in-memory của `zeroshot/twitter-financial-news-sentiment` trong Tuần 1 (khi split chính thức của Tuần 2 chưa được tạo).
   - Chạy 1 epoch fine-tune `distilbert/distilbert-base-uncased` trên 1.000 mẫu `train` và đánh giá trên 500 mẫu `validation` (`seed = 1`, `batch_size = 32`, `max_len = 64`, `lr = 5e-5`, FP16 AMP).
   - Kết quả đo thực tế trên Colab `Tesla T4`:
     - Thời gian huấn luyện 1 epoch trên 1.000 mẫu: **`epoch_seconds = 3.3` giây**.
     - Bộ nhớ GPU đỉnh: **`peak_gpu_mem_gb = 1.64` GB**.
     - Chỉ số kiểm tra hội tụ: **`val_macro_f1_smoke = 0.689`** (chỉ dùng để xác nhận pipeline hoạt động đúng, không dùng làm kết quả thí nghiệm).

3. **Task 3.3 — Smoke Test `Qwen2.5-7B-Instruct` 4-bit & Đọc Logits A/B/C (`results/smoke/qwen.json`):**
   - Bổ sung kiểm tra bắt buộc CUDA GPU trước khi nạp mô hình lượng tử hóa 4-bit NormalFloat (`nf4-4bit`, `bnb_4bit_compute_dtype = float16`) và kiểm tra tính hợp lệ, tách biệt của các single-token ID ứng với các lựa chọn `A`/`B`/`C` (có và không có khoảng trắng đầu).
   - Chạy suy luận trên đúng 20 tiêu đề lấy từ tập `train` của Twitter FNS (`seed = 1`), trích xuất xác suất 3 lớp bằng `torch.logsumexp` và `softmax` từ logits ở vị trí token cuối cùng.
   - Kết quả đo thực tế trên Colab `Tesla T4`:
     - Token IDs xác nhận: `{"A": [32, 362], "B": [33, 425], "C": [34, 356]}`.
     - Thời gian suy luận trung bình trên 20 mẫu: **`seconds_per_headline = 0.35` giây/tiêu đề**.
     - Tổng xác suất rơi vào cụm token `A`/`B`/`C` trước chuẩn hóa: **`min_abc_mass = 1.0`** (vượt ngưỡng `0.5`, không có mẫu nào rơi xuống dưới ngưỡng).
     - File `results/smoke/qwen.json` không chứa văn bản tiêu đề gốc.

4. **Task 3.4 — Ước tính Thời gian Tính toán GPU T4 (`results/smoke/gpu_budget.md`):**
   - Phân loại rõ các số đo thực nghiệm (`[M] Measured`), ước tính ngoại suy tuyến tính (`[E] Estimated`) và các giả định chưa xác minh (`[A] Unverified Assumptions`).
   - Lập bảng tính chi tiết (theo giây, phút và giờ) cho hai phạm vi:
     - **Nhóm 1 (`E5–E8, E11, E12` + Teacher LLM — Phạm vi `docs/WEEK1.md`):** `1.446 – 2.008 giờ` (`86.7 – 120.5 phút`) với kịch bản $N=5$ epochs; `3.349 – 4.323 giờ` (`201.0 – 259.4 phút`) với kịch bản trần $N=20$ epochs.
     - **Nhóm 2 (`E5–E12` toàn bộ bảng 3.4 trong `docs/guides/TV3-tuan1.md`, gồm cả `E9, E10`):** `1.905 – 2.489 giờ` (`114.3 – 149.3 phút`) với $N=5$ epochs; `5.186 – 6.243 giờ` (`311.2 – 374.6 phút`) với $N=20$ epochs.
   - Phân biệt rõ thời gian tính toán GPU thuần túy (`Estimated GPU Compute Time`) với tổng thời gian thực thi phiên làm việc thực tế (`Total Wall-Clock Execution Time`) và các chi phí phụ (overheads).

5. **Cập nhật bản đồ thư mục (`MAP.md`):**
   - Bổ sung đường dẫn `notebooks/colab_setup.ipynb` và các file trong `results/smoke/` (`env.json`, `distilbert.json`, `qwen.json`, `gpu_budget.md`, `contribution_report.md`) theo đúng quy tắc tại `AGENTS.md:L34`.

---

## 3. Phân định rõ: Công việc do Thành viên thực hiện vs. AI hỗ trợ vs. Kết quả đã kiểm chứng

| Hạng mục | Do Thành viên (TV3) trực tiếp thực hiện | Do Trợ lý AI hỗ trợ (dưới sự kiểm duyệt của TV3) | Trạng thái & Bằng chứng kiểm chứng |
| :--- | :--- | :--- | :--- |
| **Quản lý nhánh & Quy trình Git** | Ra quyết định chọn nhánh gốc (`origin/main`), phê duyệt tên nhánh `tv3/colab-smoke`, kiểm soát từng bước không cho phép tự động commit/push. | Kiểm tra lịch sử commit, đối chiếu diff giữa các nhánh remote và thực thi lệnh tạo nhánh sau khi được phê duyệt. | Nhánh `tv3/colab-smoke` trỏ đúng vào commit `e4fd2cb`; working tree chỉ chứa đúng các file thuộc phạm vi TV3. |
| **Thiết kế & Kiểm duyệt `notebooks/colab_setup.ipynb`** | Đặt yêu cầu kỹ thuật, rà soát bảo mật (yêu cầu không lưu `GH_TOKEN` vào `.git/config`, bắt buộc raise lỗi nếu thiếu `split_sha256` khi đọc split, bắt buộc kiểm tra GPU guard cho Qwen). | Soạn thảo code Python và các ô Markdown giải thích trong `notebooks/colab_setup.ipynb`; chạy kiểm tra cú pháp tĩnh (`ast.parse`) và quét regex chống lộ secret. | Cả 4/4 code cells hợp lệ cú pháp Python, `outputs: []`, `execution_count: null`, không chứa bất kỳ secret hay dữ liệu tiêu đề nào. |
| **Chạy thực nghiệm trên Google Colab T4** | **TV3 trực tiếp cấu hình Colab Secrets (`HF_TOKEN`, `GH_TOKEN`), chạy toàn bộ notebook trên runtime Google Colab GPU `Tesla T4` và xuất 3 file kết quả JSON.** | Không truy cập tài khoản Colab của người dùng, không tự sinh hay chỉnh sửa số liệu trong các file JSON kết quả. | 3 file `env.json` (SHA-256: `EB35EA1A...`), `distilbert.json` (`23112AFC...`), `qwen.json` (`41891C0C...`) được copy nguyên bản từ kết quả chạy Colab thực tế. |
| **Phân tích & Ước tính ngân sách GPU (Task 3.4)** | Rà soát phản biện qua nhiều vòng (yêu cầu làm rõ giả định `E7`, `E8`, công thức mẫu của `E10`, chống tính trùng Validation ở `E11`, phân biệt GPU compute time với wall-clock time). | Tính toán chi tiết bảng giờ/phút/giây theo chỉ đạo rà soát của TV3 và soạn thảo tài liệu `results/smoke/gpu_budget.md`. | Toàn bộ công thức và bảng số liệu trong `results/smoke/gpu_budget.md` đã được kiểm tra chéo với `env.json`, `distilbert.json`, `qwen.json` và `SPEC.md`. |

---

## 4. Những công việc CHƯA thực hiện trong Tuần 1 (Không tự nhận)

Để đảm bảo tính trung thực tuyệt đối, báo cáo này xác nhận các công việc sau **chưa** được thực hiện ở Tuần 1 (do thuộc lộ trình các tuần tiếp theo trong `PLAN.md`):
- **Chưa** chạy huấn luyện chính thức 3 seeds cho các thí nghiệm `E5`, `E6`, `E7`, `E8`, `E9`, `E10`, `E12` (thuộc Tuần 3–5 sau khi chốt split ở Tuần 2).
- **Chưa** chạy mô hình `Qwen2.5-7B-Instruct` trên kho tiêu đề báo chí năm 2026 không nhãn (thuộc Tuần 4) hay đánh giá zero-shot `E11` trên tập validation/test chính thức (thuộc Tuần 2, 5, 6).
- **Chưa** thực hiện gán nhãn thủ công cho tập test 2026 (vòng thử 30 câu thuộc Tuần 2; gán chính thuộc Tuần 3–4).
- **Chưa** đụng chạm hay truy cập vào bất kỳ tập `test` nào (tuân thủ quy tắc chỉ chạy test đúng một lần ở Tuần 6).

---

## 5. Kê khai Minh bạch Sử dụng AI (AI Transparency Disclosure)

Theo quy định tại `AGENTS.md` và Mục 14 (*Liêm chính học thuật và hỗ trợ AI*) của `docs/KeHoachChiTiet.pdf`:

| Công cụ / Mô hình AI | Mục đích sử dụng được phép (Trong Tuần 1 & Dự án) | Giới hạn tuyệt đối — KHÔNG sử dụng cho |
| :--- | :--- | :--- |
| **Trợ lý lập trình AI** (*Coding Assistant*) | - Tóm tắt và đối chiếu các tài liệu quy hoạch trong repo (`AGENTS.md`, `SPEC.md`, `PLAN.md`, `docs/WEEK1.md`, `docs/guides/TV3-tuan1.md`).<br>- Hỗ trợ viết khung mã nguồn `notebooks/colab_setup.ipynb` theo hướng dẫn có sẵn và sửa lỗi theo yêu cầu kiểm duyệt của TV3.<br>- Hỗ trợ tính toán bảng ước tính thời gian GPU (`results/smoke/gpu_budget.md`), cập nhật `MAP.md` và định dạng báo cáo đóng góp. | - **Không** dùng để tự tạo, gán nhãn, chỉnh sửa hoặc suy đoán nhãn thật (*ground-truth labels*) của tập test 2026 hay bất kỳ tập đánh giá nào.<br>- **Không** dùng để giả lập hay bịa đặt kết quả chạy thực nghiệm.<br>- **Không** tự ý thay đổi quyết định thiết kế đã chốt trong `context/decisions.md` hoặc thực hiện thao tác Git (`commit`, `push`, `merge`) khi chưa có phê duyệt của thành viên. |
| **LLM mã nguồn mở** (`Qwen/Qwen2.5-7B-Instruct` 4-bit) | - Trong Tuần 1: Chạy thử nghiệm kỹ thuật (*smoke test*) trên 20 câu thuộc tập `train` của Twitter FNS để đo tốc độ (`seconds_per_headline`) và kiểm tra phân phối xác suất trên các token `A`/`B`/`C`.<br>- Trong các tuần sau: Làm baseline zero-shot tham chiếu (`E11`) và sinh nhãn mềm/cứng trên kho tiêu đề 2026 không nhãn chỉ dùng để huấn luyện mô hình học trò (`E6`, `E12`). | - **Tuyệt đối không** dùng nhãn của LLM làm nhãn thật (*ground-truth*) cho bất kỳ tập đánh giá hay tập test nào (`AGENTS.md` Quy tắc 5). |

