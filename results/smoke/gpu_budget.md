# Task 3.4 — Ước tính Thời gian Tính toán GPU T4 (TV3 — Tuần 1)

Tài liệu này tổng hợp kết quả đo thực nghiệm (smoke test) từ Mục 3.1–3.3 trên Google Colab (`Tesla T4`) và tính toán ngân sách thời gian tính toán GPU (GPU compute time) cho các thí nghiệm của **TV3** theo yêu cầu tại `docs/WEEK1.md` và `docs/guides/TV3-tuan1.md`.

---

## 1. Số liệu cơ sở từ Smoke Test thực tế (`results/smoke/*.json`)

Các số đo dưới đây được trích xuất trực tiếp từ phiên chạy thực nghiệm trên Google Colab tại commit `e4fd2cbc06ae0a78fc947cbed1a8fa3f0a46cb7b`:

1. **Môi trường phần cứng & phần mềm (`env.json`):**
   - **GPU thực tế:** `Tesla T4` (`15.6 GB` tổng VRAM), `cuda_available = true`, CUDA `13.0`.
   - **Phiên bản phần mềm:** Python `3.13.16`, PyTorch `2.11.0+cu130`, Transformers `5.21.0`, Datasets `4.8.5`, Scikit-learn `1.6.1`, NumPy `2.1.3`.
   - **Trạng thái `requirements.lock`:** Chưa có trên nhánh gốc (`requirements_lock_exists: false`), đã fallback sang `requirements.txt`. *(Ghi chú cho CP2: môi trường Colab hiện chạy Python `3.13.16`, cần lưu ý khi sử dụng `requirements.lock` khóa trên Python 3.11).*

2. **Smoke Test `DistilBERT` (`distilbert.json`):**
   - **Cấu hình đo:** `model = "distilbert/distilbert-base-uncased"`, `n_train = 1000`, `n_val = 500`, `batch_size = 32`, `max_len = 64`, `lr = 5e-5`, `seed = 1`, `mixed_precision_fp16 = true` (1 epoch trên tập `train`/`validation` in-memory của Twitter FNS).
   - **Thời gian huấn luyện 1 epoch trên 1.000 mẫu ($t$):** **`3.3 giây`** (`epoch_seconds = 3.3`).
   - **Bộ nhớ GPU đỉnh (`peak_gpu_mem_gb`):** **`1.64 GB`**.
   - **Chỉ số kiểm tra hội tụ (`val_macro_f1_smoke`):** **`0.689`** (chỉ dùng để xác nhận pipeline huấn luyện hoạt động đúng, không dùng làm kết quả thí nghiệm).

3. **Smoke Test `Qwen2.5-7B-Instruct` 4-bit (`qwen.json`):**
   - **Cấu hình đo:** `model = "Qwen/Qwen2.5-7B-Instruct"`, `quant = "nf4-4bit"`, `bnb_4bit_compute_dtype = "float16"`, `n_samples = 20`, `seed = 1` (trên 20 tiêu đề lấy từ tập `train` của Twitter FNS).
   - **Trạng thái phần cứng:** Đã nạp và chạy thành công trên GPU `Tesla T4` (`15.6 GB` VRAM). *(Lưu ý: smoke test của Qwen không ghi nhận `peak_gpu_mem_gb`, nên báo cáo này không khẳng định con số VRAM đỉnh cụ thể của Qwen).*
   - **Thời gian suy luận trung bình ($s$):** **`0.35 giây/tiêu đề`** (`seconds_per_headline = 0.35`, đo trên mẫu nhỏ 20 tiêu đề; việc nhân lên cho các tập dữ liệu lớn hơn là **ước tính ngoại suy tuyến tính**).
   - **Tổng xác suất rơi vào cụm token lựa chọn `A`/`B`/`C` (`min_abc_mass`):** **`1.0`** (vượt ngưỡng `0.5`, `low_mass_top10_diagnostics = []`, với `letter_ids = {"A": [32, 362], "B": [33, 425], "C": [34, 356]}`).

---

## 2. Phân loại cấp độ xác thực: Measured / Estimated / Assumptions

Để đảm bảo tính minh bạch thực nghiệm (`AGENTS.md`), mọi con số trong bảng tính được phân loại rõ vào 3 nhóm:

| Ký hiệu | Cấp độ xác thực | Nội dung chi tiết |
| :---: | :--- | :--- |
| **[M]** | **Đã đo thực nghiệm (Measured)** | - Thời gian 1 epoch `DistilBERT` trên $1.000$ mẫu train: $t = \mathbf{3.3\text{ s}}$, `peak_gpu_mem_gb` $= \mathbf{1.64\text{ GB}}$.<br>- Thời gian suy luận `Qwen2.5-7B` 4-bit trên 20 mẫu train: $s = \mathbf{0.35\text{ s/tiêu đề}}$. |
| **[E]** | **Ước tính ngoại suy tuyến tính (Estimated)** | - Tập `train` đầy đủ của Twitter FNS chiếm $70\% \times 11.931 \approx \mathbf{8.350\text{ mẫu}}$ (gấp $8.35\times$ quy mô smoke test).<br>- Thời gian tính toán 1 epoch `DistilBERT` trên $8.350$ mẫu: $T_{\text{1-seed, 1-ep}} = 8.35 \times 3.3 = \mathbf{27.555\text{ s}}$.<br>- Thời gian tính toán 3 seeds $\times$ 1 epoch của `E5`: $T_1 = 3 \times 27.555 = \mathbf{82.665\text{ s}} = \mathbf{1.378\text{ phút}} = \mathbf{0.0229625\text{ giờ}}$.<br>- Thời gian suy luận `Qwen2.5-7B` 4-bit ngoại suy tuyến tính cho $K$ tiêu đề: $T_{\text{Qwen}}(K) = K \times 0.35\text{ s}$. |
| **[A]** | **Giả định chưa xác minh thực nghiệm (Unverified Assumptions)** | - **Số epoch huấn luyện ($N$):** Chưa có `configs/E5.yaml`. Xét 2 kịch bản: $N = 5$ (kịch bản chuẩn khi fine-tune Transformer) và $N = 20$ (mức trần `max_epochs: 20` theo template `configs/E2.yaml` nếu không dừng sớm bởi early stopping).<br>- **Kích thước kho tiêu đề 2026 không nhãn ($M_{2026}$):** Xét khoảng từ $2.200$ câu (`docs/guides/TV3-tuan1.md`) đến $3.000$ câu (`SPEC.md`, `PLAN.md`).<br>- **`E7` (Đóng băng encoder):** Lấy ngân sách bảo thủ cố định bằng $1.000 \times \text{E5}$.<br>- **`E8` (FinBERT):** Xét hai kịch bản ngoại suy kiến trúc $1.000\times \text{ – } 2.000\times \text{E5}$.<br>- **`E10` (Rò rỉ 10/25/50%):** Xét từ kịch bản giữ cố định cỡ tập train ($3.000 \times \text{E5}$) đến kịch bản cận trên cộng gộp mẫu rò rỉ ($3.182 \times \text{E5}$).<br>- **`E11` (Zero-shot Qwen):** Xét từ kịch bản chạy 1 prompt chuẩn trên Validation + 3 tập Test ($6.144$ câu) đến kịch bản thử 3 prompts trên Validation ở Tuần 2 + 1 prompt thắng cuộc trên 3 tập Test ở Tuần 6 ($9.724$ câu). |

---

## 3. Giải trình phương pháp tính cho từng thí nghiệm đặc thù

### 3.1 `E7` (DistilBERT với Encoder đóng băng)
- **Ngân sách sử dụng:** Đặt bằng **`1.000 × E5`** (theo đúng hướng dẫn `≈ E5 hoặc ít hơn` tại `docs/guides/TV3-tuan1.md`).
- **Ghi chú lý thuyết:** Khi đóng băng 6 tầng Transformer encoder (`requires_grad = False`), PyTorch chỉ cần tính gradient và cập nhật trọng số cho tầng phân loại (classifier head), bỏ qua bước lan truyền ngược qua khối encoder nên về mặt lý thuyết thời gian mỗi epoch được kỳ vọng sẽ thấp hơn `E5`. Tuy nhiên, vì chưa chạy benchmark riêng cho `E7` ở Tuần 1, mọi hệ số giảm cụ thể (như $0.4\text{–}0.6\times$) chỉ là **nhận định lý thuyết chưa được benchmark** và không được dùng để cắt giảm ngân sách dự kiến.

### 3.2 `E8` (FinBERT — `ProsusAI/finbert`)
- **Hai kịch bản ngoại suy kiến trúc (`1.000× – 2.000× E5`):**
  - **Cận dưới (`1.000 × E5`):** Theo ký hiệu xấp xỉ cơ sở `≈ E5` trong bảng 3.4 của hướng dẫn.
  - **Cận trên (`2.000 × E5`):** Ngoại suy theo tỷ lệ số tầng Transformer encoder giữa `ProsusAI/finbert` (kiến trúc `bert-base` 12 tầng, ~110M tham số) và `distilbert-base-uncased` (6 tầng, ~66M tham số) như ghi chú trong hướng dẫn (`BERT-base lớn hơn DistilBERT ~2×`).
- **Lưu ý:** Khoảng `1.000× – 2.000× E5` là **hai kịch bản ngoại suy kiến trúc**, không phải giới hạn dưới/trên đã được xác nhận bằng benchmark thực nghiệm trên tập dữ liệu này.

### 3.3 `E10` (Rò rỉ dữ liệu 10% / 25% / 50% trên DistilBERT — RQ1)
- **Quy mô chạy:** 3 mức rò rỉ ($10\%, 25\%, 50\%$) $\times$ 3 seeds = **9 lần huấn luyện** (mỗi lần $N$ epochs).
- **Công thức xác định số mẫu mỗi mức:**
  - Tập `train` trong miền chuẩn có $N_{\text{train}} \approx 70\% \times 11.931 \approx \mathbf{8.350\text{ mẫu}}$; tập `test` trong miền có $N_{\text{test}} \approx 15\% \times 11.931 \approx \mathbf{1.790\text{ mẫu}}$.
  - Hiện tại script tạo dữ liệu rò rỉ (việc Tuần 4 của TV1 — `PLAN.md:L32`) và tỷ lệ cụm trùng chốt ở CP2 (Việc 1.5) chưa được cố định trong `configs/`, nên ta xét 2 kịch bản rõ ràng:
    1. **Kịch bản 1 — Giữ cố định kích thước tập train (`3.000 × E5`, theo công thức trong `TV3-tuan1.md`):**
       Nếu mỗi mức rò rỉ thay thế mẫu sao cho kích thước tập train giữ nguyên ở mức $8.350$ mẫu:
       $$\text{Tổng mẫu trên 1 seed qua 3 mức} = 8.350 + 8.350 + 8.350 = 25.050\text{ mẫu} = \mathbf{3.000 \times \text{E5}}$$
    2. **Kịch bản 2 — Giả định cận trên bảo thủ khi cộng gộp trực tiếp mẫu rò rỉ vào tập train (`3.182 × E5`, chưa xác minh):**
       Nếu ở mỗi mức $r \in \{10\%, 25\%, 50\%\}$, một lượng mẫu bằng $r \times N_{\text{test}}$ được **cộng thêm** trực tiếp vào $8.350$ mẫu train gốc:
       - Mức $10\%$ (3 seeds): $N_{\text{train}, 10\%} = 8.350 + 0.10 \times 1.790 = 8.350 + 179 = \mathbf{8.529\text{ mẫu/epoch}}$ ($1.0214 \times \text{E5}$)
       - Mức $25\%$ (3 seeds): $N_{\text{train}, 25\%} = 8.350 + 0.25 \times 1.790 = 8.350 + 448 = \mathbf{8.798\text{ mẫu/epoch}}$ ($1.0537 \times \text{E5}$)
       - Mức $50\%$ (3 seeds): $N_{\text{train}, 50\%} = 8.350 + 0.50 \times 1.790 = 8.350 + 895 = \mathbf{9.245\text{ mẫu/epoch}}$ ($1.1072 \times \text{E5}$)
       - Cộng tổng cả 3 mức (9 lần chạy):
         $$\text{Hệ số so với E5} = \frac{8.529 + 8.798 + 9.245}{8.350} = \frac{26.572}{8.350} = \mathbf{3.1823 \times \text{E5}}$$

### 3.4 `E11` (Qwen2.5-7B 4-bit Zero-Shot) — Kiểm soát chống tính trùng (No Double-Counting)
- **Danh sách các tập dữ liệu đánh giá (`SPEC.md:L26-L29`):**
  - `validation` trong miền (từ `data/splits/cluster_split.jsonl`, $15\% \times 11.931$): $\approx \mathbf{1.790\text{ câu}}$.
  - `test` trong miền (từ `data/splits/cluster_split.jsonl`, $15\% \times 11.931$, chỉ chạy 1 lần ở Tuần 6): $\approx \mathbf{1.790\text{ câu}}$.
  - `test` ngoài phân phối — Tiêu đề báo chí 2026 do nhóm gán nhãn tay (Tuần 6): $\mathbf{300\text{ câu}}$.
  - `test` ngoài miền — Financial PhraseBank AllAgree (`Sentences_AllAgree.txt`, Tuần 6): $\mathbf{2.264\text{ câu}}$.
- **Hai kịch bản số lượng câu (loại bỏ hoàn toàn việc chạy lặp tập Validation):**
  1. **Kịch bản chuẩn 1 prompt (`6.144 câu`):** Chạy 1 lần trên `validation` ($1.790$ câu) + 1 lần trên 3 tập `test` ở Tuần 6 ($1.790 + 300 + 2.264 = 4.354$ câu) $\Rightarrow$ tổng cộng **$6.144$ câu**.
  2. **Kịch bản tính cả bước chọn 3 biến thể prompt ở Tuần 2 (`9.724 câu`):** Ở Tuần 2 (`PLAN.md:L20`), nếu đánh giá cả 3 biến thể prompt A/B/C trên tập `validation` ($3 \times 1.790 = 5.370$ câu), lưu lại kết quả của prompt tốt nhất làm kết quả validation chính thức của `E11` (không chạy lại ở Tuần 5/6), và chỉ chạy prompt đó 1 lần trên 3 tập `test` ở Tuần 6 ($4.354$ câu) $\Rightarrow$ tổng cộng $5.370 + 4.354 = \mathbf{9.724\text{ câu}}$.

### 3.5 Hai phạm vi tổng hợp: Nhóm 1 (`WEEK1.md`) và Nhóm 2 (`TV3-tuan1.md`)
- **Nhóm 1 — Phạm vi `E5–E8, E11, E12` + Teacher LLM (`docs/WEEK1.md:L52`):** Bao gồm toàn bộ các thí nghiệm thuộc **RQ3** (Ngoài phân phối & Chưng cất tri thức) do TV3 phụ trách chính.
- **Nhóm 2 — Phạm vi `E5–E12` (`docs/guides/TV3-tuan1.md:L40,L227-L236`):** Bao gồm **Nhóm 1** cộng thêm phần thực nghiệm **RQ1 trên kiến trúc DistilBERT (`E9` và `E10`)** được phân công cho TV3 ở Tuần 5 (`PLAN.md:L41`).

---

## 4. Bảng tổng hợp Thời gian Tính toán GPU T4 (GPU Compute Time)

*(Đơn vị được tính chính xác theo **giây**, sau đó quy đổi sang **phút** $[\div 60]$ và **giờ** $[\div 3600]$).*

| Thí nghiệm / Hạng mục | Phân loại | Số lần chạy & Công thức tính (theo giây) | Kịch bản $N = 5$ epochs<br>**Giờ (Phút) [Giây]** | Kịch bản $N = 20$ epochs<br>**Giờ (Phút) [Giây]** |
| :--- | :---: | :--- | :---: | :---: |
| **E5** — DistilBERT fine-tune | **[M]+[E]**<br>*(với $N$ thuộc [A])* | 3 seeds × $N$ ep × 8.350 mẫu<br>$= N \times 82.665\text{ s}$ | **0.115 giờ**<br>(6.9 phút) `[413.3 s]` | **0.459 giờ**<br>(27.6 phút) `[1,653.3 s]` |
| **E6** — DistilBERT + Chưng cất nhãn mềm LLM | **[E]+[A]** | 3 seeds × $N$ ep × $(8.350 + M_{2026})$ mẫu<br>- Hạ cận ($10.550$ mẫu): $N \times 104.445\text{ s}$<br>- Thượng cận ($11.350$ mẫu): $N \times 112.365\text{ s}$ | **0.145 – 0.156 giờ**<br>(8.7 – 9.4 phút)<br>`[522.2 – 561.8 s]` | **0.580 – 0.624 giờ**<br>(34.8 – 37.5 phút)<br>`[2,088.9 – 2,247.3 s]` |
| **E7** — DistilBERT đóng băng encoder | **[A]** | 3 seeds × $N$ ep<br>Ngân sách bảo thủ: $1.000 \times \text{E5}$<br>*(Nhận định lý thuyết chưa benchmark: bỏ backward qua encoder sẽ nhanh hơn)* | **0.115 giờ**<br>(6.9 phút) `[413.3 s]` | **0.459 giờ**<br>(27.6 phút) `[1,653.3 s]` |
| **E8** — FinBERT fine-tune | **[A]** | 3 seeds × $N$ ep (Kịch bản ngoại suy kiến trúc)<br>- Kịch bản $1.0\times \text{E5}$: $N \times 82.665\text{ s}$<br>- Kịch bản $2.0\times \text{E5}$: $N \times 165.330\text{ s}$ | **0.115 – 0.230 giờ**<br>(6.9 – 13.8 phút)<br>`[413.3 – 826.7 s]` | **0.459 – 0.919 giờ**<br>(27.6 – 55.1 phút)<br>`[1,653.3 – 3,306.6 s]` |
| **Teacher LLM** — Sinh nhãn cho kho 2026 không nhãn | **[M]+[E]+[A]** | 1 lần chạy Qwen 4-bit ($0.35\text{ s/câu}$, ngoại suy từ 20 mẫu)<br>- $2.200$ câu: $770.0\text{ s}$<br>- $3.000$ câu: $1,050.0\text{ s}$ | **0.214 – 0.292 giờ**<br>(12.8 – 17.5 phút)<br>`[770.0 – 1,050.0 s]` | **0.214 – 0.292 giờ**<br>(12.8 – 17.5 phút)<br>`[770.0 – 1,050.0 s]` |
| **E11** — Qwen2.5-7B 4-bit Zero-Shot (không tính trùng Val) | **[M]+[E]+[A]** | 1 seed ($0.35\text{ s/câu}$, ngoại suy từ 20 mẫu)<br>- 1 prompt ($6.144$ câu): $2,150.4\text{ s}$<br>- 3 prompts trên Val + 1 prompt trên 3 Test ($9.724$ câu): $3,403.4\text{ s}$ | **0.597 – 0.945 giờ**<br>(35.8 – 56.7 phút)<br>`[2,150.4 – 3,403.4 s]` | **0.597 – 0.945 giờ**<br>(35.8 – 56.7 phút)<br>`[2,150.4 – 3,403.4 s]` |
| **E12** — DistilBERT + Nhãn cứng LLM | **[E]+[A]** | 3 seeds × $N$ ep × $(8.350 + M_{2026})$ mẫu<br>- Hạ cận ($10.550$ mẫu): $N \times 104.445\text{ s}$<br>- Thượng cận ($11.350$ mẫu): $N \times 112.365\text{ s}$ | **0.145 – 0.156 giờ**<br>(8.7 – 9.4 phút)<br>`[522.2 – 561.8 s]` | **0.580 – 0.624 giờ**<br>(34.8 – 37.5 phút)<br>`[2,088.9 – 2,247.3 s]` |
| **TỔNG NHÓM 1 (`E5–E8, E11, E12` + Teacher — Phạm vi `WEEK1.md`)** | — | Tổng chính xác từng cận | **1.446 – 2.008 giờ**<br>**(86.7 – 120.5 phút)**<br>`[5,204.8 – 7,230.4 s]` | **3.349 – 4.323 giờ**<br>**(201.0 – 259.4 phút)**<br>`[12,058.1 – 15,561.2 s]` |
| **E9** — DistilBERT trên split ngẫu nhiên (RQ1) | **[M]+[E]** | 3 seeds × $N$ ep × 8.350 mẫu ($= 1.000 \times \text{E5}$)<br>$= N \times 82.665\text{ s}$ | **0.115 giờ**<br>(6.9 phút) `[413.3 s]` | **0.459 giờ**<br>(27.6 phút) `[1,653.3 s]` |
| **E10** — DistilBERT rò rỉ 10/25/50% (RQ1) | **[E]+[A]** | 3 mức × 3 seeds × $N$ ep (9 lần chạy)<br>- Giữ cỡ train ($25.050$ mẫu $= 3.000\times\text{E5}$): $N \times 247.995\text{ s}$<br>- Cộng gộp ($26.572$ mẫu $= 3.182\times\text{E5}$): $N \times 263.063\text{ s}$ | **0.344 – 0.365 giờ**<br>(20.7 – 21.9 phút)<br>`[1,240.0 – 1,315.3 s]` | **1.378 – 1.461 giờ**<br>(82.7 – 87.7 phút)<br>`[4,959.9 – 5,261.3 s]` |
| **TỔNG NHÓM 2 (`E5–E12` toàn bộ bảng 3.4 trong `TV3-tuan1.md`)** | — | **Tổng Nhóm 1 + `E9` + `E10`** | **1.905 – 2.489 giờ**<br>**(114.3 – 149.3 phút)**<br>`[6,858.1 – 8,959.0 s]` | **5.186 – 6.243 giờ**<br>**(311.2 – 374.6 phút)**<br>`[18,671.3 – 22,475.8 s]` |

---

## 5. Phân biệt Thời gian Tính toán GPU (Estimated GPU Compute Time) và Tổng Thời gian Thực thi Thực tế (Total Wall-Clock Execution Time)

Con số **1.45 – 4.32 giờ** (Nhóm 1) và **1.91 – 6.24 giờ** (Nhóm 2) ở Mục 4 chỉ phản ánh **thời gian tính toán thuần túy trên GPU (Estimated GPU Compute Time)** của các vòng lặp huấn luyện và suy luận. Khi triển khai thực tế trên Google Colab / Kaggle từ Tuần 2 đến Tuần 6, **tổng thời gian thực thi thực tế (wall-clock session time)** sẽ bao gồm thêm các chi phí phụ (overheads) chưa nằm trong công thức trên:

1. **Tải trọng số mô hình & Khởi tạo lượng tử hóa (Model Download & Initialization Overhead):**
   - Mỗi phiên runtime Colab mới cần tải trọng số `Qwen/Qwen2.5-7B-Instruct` (~15 GB, mất khoảng 5–10 phút tùy băng thông mạng) và lượng tử hóa sang 4-bit NF4 trước khi bắt đầu suy luận, cũng như tải `DistilBERT` / `FinBERT` (~260–440 MB).
2. **Đánh giá Validation & Lưu Checkpoint sau mỗi Epoch:**
   - Công thức `epoch_seconds` ở Mục 3.2 chỉ đo vòng lặp `model.train()`. Thực tế mỗi epoch còn kèm bước chạy `model.eval()` trên ~1.790 mẫu `validation`, tính Macro-F1 để kiểm tra điều kiện early stopping và ghi checkpoint tốt nhất ra đĩa.
3. **Chi phí ghép Batch trộn và xử lý dữ liệu (`E6`, `E12`):**
   - Ở `E6` (chưng cất nhãn mềm với hàm mất mát kết hợp $\text{CE} + T^2 \cdot \text{KL}$) và `E12`, DataLoader phải đọc đồng thời hai nguồn dữ liệu (`Twitter FNS` và kho tiêu đề `2026` kèm logits thầy) và chuẩn bị hai nhánh tính loss trong mỗi bước tối ưu.
4. **Cài đặt môi trường & Quản lý phiên (Session Setup & Restart Overhead):**
   - Thời gian `git clone`, `pip install` (~1–3 phút mỗi phiên) và thời gian khởi động lại session khi chuyển đổi giữa mô hình Transformer và LLM.
5. **Tìm kiếm siêu tham số $(\alpha, T)$ cho `E6` trên Validation:**
   - Theo `SPEC.md:L45`, hệ số trộn $\alpha$ và nhiệt độ $T$ của `E6` được chọn trên tập validation trước khi chạy 3 seeds chính thức. Mỗi cấu hình $(\alpha, T)$ thử nghiệm (1 seed) sẽ cộng thêm $\approx 2.9\text{–}3.1\text{ phút}$ ($N=5$) hoặc $\approx 11.6\text{–}12.5\text{ phút}$ ($N=20$) thời gian tính toán.

---

## 6. Kết luận cho Task 3.4 (Biên bản CP2)

> **Kết luận Mục 3.4 — Ước tính ngân sách GPU T4 cho TV3:**
> - **Số đo cơ sở trên Colab Tesla T4 (`15.6 GB` VRAM):**
>   - `DistilBERT` (`batch_size=32, max_len=64`, FP16 AMP) mất **`3.3 giây/epoch`** trên 1.000 mẫu train (VRAM đỉnh **`1.64 GB`**) $\Rightarrow$ ngoại suy tuyến tính sang tập train FNS đầy đủ (~8.350 mẫu) là **`27.6 giây/epoch`** (`0.023 giờ` cho 3 seeds × 1 epoch).
>   - `Qwen2.5-7B-Instruct` (`nf4-4bit`, `float16`) chạy thành công trên Tesla T4 với tốc độ trung bình **`0.35 giây/tiêu đề`** đo trên 20 mẫu train FNS (`min_abc_mass = 1.0`); các ước tính cho tập lớn hơn được ngoại suy tuyến tính từ số đo này.
> - **Tổng thời gian tính toán GPU T4 ước tính (Estimated GPU Compute Time):**
>   1. **Nhóm 1 — Phạm vi RQ3 (`E5–E8, E11, E12` + sinh nhãn LLM cho kho 2026 theo `docs/WEEK1.md`):** **`1.45 – 2.01 giờ`** (`86.7 – 120.5 phút`) với kịch bản $N=5$ epochs, và **`3.35 – 4.32 giờ`** (`201.0 – 259.4 phút`) với kịch bản trần $N=20$ epochs không dừng sớm.
>   2. **Nhóm 2 — Toàn bộ phạm vi `E5–E12` (bao gồm cả `E9, E10` của RQ1 theo `docs/guides/TV3-tuan1.md`):** **`1.91 – 2.49 giờ`** (`114.3 – 149.3 phút`) với $N=5$ epochs, và **`5.19 – 6.24 giờ`** (`311.2 – 374.6 phút`) với $N=20$ epochs.
> - **Đánh giá tính khả thi & Khuyến nghị tại CP2:**
>   - Tài liệu hướng dẫn có nêu mốc tham khảo (~30 giờ GPU/tuần trên Kaggle, vài giờ/ngày trên Colab), tuy nhiên hạn mức GPU miễn phí thực tế trên cả hai nền tảng được phân bổ động và không cố định cứng.
>   - Ngay cả khi cộng thêm các chi phí phụ ngoài thời gian tính toán thuần túy (tải trọng số Qwen ~15 GB mỗi phiên, đánh giá validation mỗi epoch, cài đặt môi trường và chọn siêu tham số $\alpha, T$ cho `E6`), tổng thời gian GPU T4 cần thiết của TV3 trải đều trong 5 tuần (Tuần 2 → Tuần 6) vẫn rất nhỏ (trung bình chỉ ~0.5–1.5 giờ tính toán/tuần). Do đó, **kế hoạch thí nghiệm `E5–E12` hoàn toàn khả thi và không cần cắt giảm bất kỳ thí nghiệm nào (giữ nguyên cả `E12`)**.
>   - **Việc cần chốt ở Tuần 2:** Xác định `max_epochs` ($N$) chính thức trong `configs/E5.yaml` và quy tắc trộn dữ liệu rò rỉ của `E10`.

