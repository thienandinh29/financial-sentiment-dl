# Chương 2 (phần Transformer & Chưng cất) — Mô hình Transformer và chưng cất tri thức từ LLM cho phân loại tiêu đề tài chính

> **Bản nháp (TV3, 27/11/2026).** Ghép nối trực tiếp với phần RNN/CNN và attention của TV2 (`report/ch2_rnn_cnn.md`).
> Mọi số liệu thực nghiệm trong chương này là kết quả trên tập **validation** (`1.791` mẫu của `data/splits/cluster_split.jsonl`), **không phải** kết quả trên tập test; ba tập test chỉ được đánh giá đúng một lần ở tuần 6 (`AGENTS.md` quy tắc 1).
> Đánh dấu `[HÌNH]` là vị trí đặt sơ đồ khi ghép báo cáo tổng hợp.

Nếu các mô hình CNN và BiLSTM ở mục 2.1–2.4 học biểu diễn câu từ vector từ tĩnh (GloVe 6B 100d) trên tập từ vựng cố định của tập train, thì phần này khảo sát bước chuyển sang **mô hình ngôn ngữ tiền huấn luyện dựa trên kiến trúc Transformer** (`DistilBERT`, `FinBERT`) và **chưng cất tri thức (knowledge distillation)** từ một mô hình ngôn ngữ lớn (`Qwen2.5-7B-Instruct` 4-bit) trên kho tiêu đề tin tức tài chính năm 2026 không nhãn để trả lời câu hỏi nghiên cứu **RQ3**: *khi chuyển từ tweet tài chính sang tiêu đề báo chí năm 2026, mô hình suy giảm bao nhiêu và việc chưng cất phân phối xác suất mềm từ LLM (`E6`) có vượt qua việc chỉ gán nhãn giả cứng (`E12`) hay không?*

---

## 2.8 Kiến trúc Transformer, Self-Attention và DistilBERT

**Tách từ subword (WordPiece).** Khác với bộ tách từ theo từ nguyên vẹn của RNN/CNN, `distilbert/distilbert-base-uncased` sử dụng từ điển WordPiece 30.522 token có sẵn từ quá trình tiền huấn luyện: các từ hiếm hoặc tên riêng tài chính chưa từng gặp được tách thành các đơn vị con (subwords) thay vì rơi vào một token `<unk>` duy nhất. Mỗi câu được bao bởi token đặc biệt `[CLS]` ở đầu và `[SEP]` ở cuối, cắt hoặc đệm tới độ dài tối đa `max_len: 64` token (`configs/E5.yaml`).

**Self-attention tích vô hướng có chuẩn hóa (Scaled Dot-Product Self-Attention; Vaswani và cộng sự, 2017).** Ở mỗi tầng Transformer, dãy biểu diễn đầu vào $H \in \mathbb{R}^{n \times d}$ ($d = 768$) được chiếu tuyến tính thành ba ma trận truy vấn $Q = H W_Q$, khóa $K = H W_K$ và giá trị $V = H W_V$ (trong mỗi đầu chú ý có chiều $d_k = d / h = 64$, với $h = 12$ đầu):

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V .$$

Trong đó ma trận mặt nạ $M$ gán $0$ cho các token thật và $-\infty$ cho các vị trí đệm `<pad>`. Khác với BiLSTM phải truyền trạng thái tuần tự qua $n$ bước thời gian, self-attention nối trực tiếp mọi cặp token $(i, j)$ trong câu chỉ bằng một phép nhân ma trận có đường đi độ dài $O(1)$. Nhờ vậy, các cấu trúc đảo chiều hoặc phụ thuộc xa trong tiêu đề tài chính — ví dụ mối liên hệ giữa cụm từ chỉ chủ thể ở đầu câu và mệnh đề điều kiện *"despite beating Q3 EPS estimates"* ở cuối câu — được mô hình hóa đồng thời trên nhiều không gian con qua $h = 12$ đầu chú ý (multi-head attention).

**DistilBERT (Sanh và cộng sự, 2019).** Để phù hợp với ngân sách tính toán GPU T4 và yêu cầu suy luận độ trễ thấp trên CPU (xuất ONNX ở tuần 7), dự án sử dụng `distilbert-base-uncased` làm mô hình học trò (student) trung tâm của RQ3. DistilBERT được chưng cất từ `bert-base-uncased` (12 tầng, ~110 triệu tham số) xuống còn **6 tầng Transformer encoder** ($d = 768$, $h = 12$, tầng FFN trung gian $3.072$ chiều, hàm kích hoạt GELU), giữ lại ~97% năng lực biểu diễn ngôn ngữ của BERT nhưng giảm 40% số tham số và tăng tốc độ suy luận ~60%. Trên đỉnh vector ẩn của token `[CLS]` ở tầng thứ 6 ($h_{\text{[CLS]}} \in \mathbb{R}^{768}$), tầng phân loại của `AutoModelForSequenceClassification` áp dụng:

$$z = W_{\text{cls}}\,\mathrm{Dropout}\!\Big(\mathrm{ReLU}\big(W_{\text{pre}}\, h_{\text{[CLS]}} + b_{\text{pre}}\big)\Big) + b_{\text{cls}}, \qquad W_{\text{pre}} \in \mathbb{R}^{768 \times 768},\ W_{\text{cls}} \in \mathbb{R}^{3 \times 768}.$$

Tổng số tham số của mô hình là **$66.955.779$** tham số (`results/E5/seed1/run.json`). `[HÌNH: sơ đồ DistilBERT 6 tầng encoder → vector [CLS] → pre_classifier (768×768) → dropout 0,3 → classifier (768×3)]`

---

## 2.9 Tinh chỉnh toàn bộ (`E5`) so với Đóng băng bộ mã hóa (`E7`)

Một câu hỏi nền tảng khi áp dụng Transformer tiền huấn luyện cho văn bản tài chính là: *các biểu diễn ngữ cảnh tổng quát học từ Wikipedia và BookCorpus đã đủ để phân tách cảm xúc `bearish / neutral / bullish` bằng một bộ phân loại đặt trên đỉnh, hay bắt buộc phải cập nhật lại các tầng self-attention bên trong bộ mã hóa?*

Để kiểm định câu hỏi này theo đúng quy tắc "một thay đổi mỗi thí nghiệm" (`AGENTS.md` quy tắc 4), cấu hình `configs/E7.yaml` được sao chép nguyên bản từ `configs/E5.yaml` và chỉ thay đổi duy nhất tham số `model.freeze_encoder`:

| Thành phần cấu hình | `E5` — Tinh chỉnh toàn bộ (`configs/E5.yaml`) | `E7` — Đóng băng encoder (`configs/E7.yaml`) |
| --- | --- | --- |
| Bộ mã hóa (6 tầng DistilBERT + embedding) | Huấn luyện (`requires_grad = True`) | **Đóng băng (`requires_grad = False`)** |
| Đầu phân loại (`pre_classifier` + `classifier`) | Huấn luyện (`requires_grad = True`) | Huấn luyện (`requires_grad = True`) |
| Số tham số huấn luyện / Tổng tham số | **$66.955.779 / 66.955.779$ (`100,00%`)** | **$592.899 / 66.955.779$ (`0,89%`)** |
| Số tham số đóng băng | $0$ (`0,00%`) | $66.362.880$ (`99,11%`) |
| Siêu tham số chung | Weighted CE (trọng số lớp tính chỉ trên train FNS), AdamW ($\text{lr} = 2\times 10^{-5}$, $\text{weight\_decay} = 0{,}01$), linear warmup 10% + linear decay, batch 32, tối đa 4 epoch, dropout 0,3, FP16 AMP, early stopping theo `val_macro_f1` | Giữ nguyên 100% như `E5` |

Ở `E7`, chỉ có $592.899$ tham số của hai tầng tuyến tính `pre_classifier` ($768 \times 768 + 768 = 590.592$) và `classifier` ($768 \times 3 + 3 = 2.307$) được cập nhật (`models/transformer.py`, `results/E7/seed1/run.json`).

---

## 2.10 Trích xuất phân phối xác suất từ LLM (`P2`) và Chưng cất tri thức (`E12` vs. `E6`)

### 2.10.1 Trích xuất xác suất từ logits token `A / B / C` của `Qwen2.5-7B-Instruct` 4-bit
Mô hình học trên tweet tài chính (Twitter FNS) thường chịu sự lệch phân phối kết hợp về cả nguồn/văn phong và thời gian khi chuyển sang tiêu đề báo chí năm 2026. Để tận dụng tri thức của mô hình ngôn ngữ lớn mà không vi phạm quy tắc trung thực thực nghiệm (LLM không bao giờ gán nhãn tập test; `AGENTS.md` quy tắc 5), dự án sử dụng `Qwen/Qwen2.5-7B-Instruct` lượng tử hóa 4-bit NormalFloat (`nf4-4bit`) làm **mô hình thầy (teacher)** chỉ trên kho tiêu đề năm 2026 không nhãn (`pool == "train_unlabeled"`, nguồn GDELT, gồm đúng **$1.409$ tiêu đề** sau bộ lọc chất lượng trong `data/raw/headlines_2026/headlines.jsonl`, `SHA-256: 28bd2f3d...`). Mọi tiêu đề thuộc nguồn ứng viên test RSS (`759` câu), bao gồm `30` ID gán thử (`labeling/trial30_ids.csv`) và `300` ID test chính thức (`labeling/assignment.csv`), đều bị loại trừ nghiêm ngặt bởi chốt chặn tự động trong `models/llm_soft_labels.py`.

Thay vì yêu cầu LLM sinh văn bản tự do (vốn không phản ánh phân phối xác suất thực của mô hình), prompt **`P2`** (đã được chọn trên tập validation FNS ở tuần 2 với `val_macro_f1 = 0,7934`, vượt `P1` `0,7311` và `P3` `0,7057`; xem `results/llm_prompts/top10_check.json`) cung cấp định nghĩa ba lớp theo `labeling/GUIDE.md` và buộc mô hình trả lời bằng một ký tự đơn `A` (`bearish`), `B` (`neutral`) hoặc `C` (`bullish`). Tại bước giải mã đầu tiên sau prompt, gọi $\ell \in \mathbb{R}^{|V|}$ là vector log-softmax trên toàn bộ từ vựng $V$ của Qwen. Vì bộ tách từ BPE mã hóa ký tự đứng đầu câu (`"A"`) và ký tự có khoảng trắng đứng trước (`" A"`) thành hai token ID khác nhau ($\mathcal{I}_c$ với $c \in \{A, B, C\}$), log-xác suất gộp của mỗi phương án được tính bằng phép cộng xác suất trong miền log:

$$z_{t,c} = \log \sum_{k \in \mathcal{I}_c} \exp(\ell_k), \qquad c \in \{A, B, C\}.$$

Bộ ba giá trị $\text{letter\_logp} = (z_{t,A}, z_{t,B}, z_{t,C})$ cùng tổng khối lượng xác suất rơi vào ba phương án $\text{abc\_mass} = \sum_{c \in \{A,B,C\}} \exp(z_{t,c})$ được lưu trực tiếp vào `results/llm_soft_labels/p2_soft_labels.jsonl` (tuyệt đối không lưu văn bản tiêu đề hay URL). Trên toàn bộ $1.409$ tiêu đề `train_unlabeled`, Qwen2.5-7B-Instruct đạt `mean_abc_mass = 0,9998`, `min_abc_mass = 0,9890`, và `100%` số mẫu có token đứng đầu toàn từ vựng là `A`, `B` hoặc `C` (`results/llm_soft_labels/run.json`).

### 2.10.2 Nhãn giả cứng (`E12`) so với Chưng cất nhãn mềm có nhiệt độ (`E6`)
Từ cùng bộ dự đoán của thầy trên $1.409$ tiêu đề `train_unlabeled`, hai chiến lược chuyển giao tri thức cho mô hình học trò DistilBERT được đặt đối chứng trực tiếp:

1. **Học trò nhãn giả cứng (`E12` — Hard Pseudo-Labeling; `configs/E12.yaml`):**
   Mỗi tiêu đề không nhãn $j \in \{1, \dots, 1.409\}$ được gán một nhãn rời rạc duy nhất theo quy tắc $\hat{y}_j = \arg\max_{c \in \{0,1,2\}} z_{t,j,c}$ (`bearish`: 81 câu, `neutral`: 1.186 câu, `bullish`: 142 câu) và gộp chung với $8.351$ câu train của Twitter FNS (tổng cộng $9.760$ mẫu train). Để giữ nguyên mốc so sánh công bằng với `E5`, trọng số lớp $w_c$ của hàm mất mát `weighted_ce` **chỉ được tính trên $8.351$ mẫu train gốc của FNS** (`models/transformer.py`). Điểm hạn chế của nhãn cứng là nó xóa bỏ hoàn toàn thông tin về độ không chắc chắn giữa các lớp gần nhau (ví dụ một tiêu đề có $60\%$ `neutral` và $38\%$ `bearish` bị ép thành $100\%$ `neutral`).

2. **Chưng cất tri thức nhãn mềm (`E6` — Soft-Label Knowledge Distillation; Hinton và cộng sự, 2015; `configs/E6.yaml`):**
   Vì phân phối softmax ở nhiệt độ $T = 1$ của Qwen khá sắc (`probs` gần như one-hot ở đa số mẫu), `E6` làm mềm cả phân phối của thầy lẫn học trò ở nhiệt độ $T \in \{1, 2, 4\}$ trực tiếp từ vector `letter_logp` $z_t$ của thầy và vector logit $z_s$ của học trò DistilBERT:

   $$p_{t,c}^{(T)} = \frac{\exp(z_{t,c} / T)}{\sum_{k=0}^{2} \exp(z_{t,k} / T)}, \qquad p_{s,c}^{(T)} = \frac{\exp(z_{s,c} / T)}{\sum_{k=0}^{2} \exp(z_{s,k} / T)}.$$

   Trong mỗi mini-batch trộn gồm $n_{\text{FNS}}$ mẫu Twitter FNS (có nhãn thật $y_i$) và $n_{\text{pseudo}}$ mẫu tiêu đề 2026 (có phân phối mềm $p_{t,j}^{(T)}$), hàm mất mát trên dòng FNS là $\mathcal{L}_{\text{WCE}}$ như ở `E5`, còn hàm mất mát trên dòng nhãn giả là độ phân kỳ Kullback–Leibler có nhân hệ số bù độ lớn gradient $\alpha \cdot T^2$:

   $$\mathcal{L}_{\text{pseudo}}(j) = \alpha \cdot T^2 \cdot \mathrm{KL}\!\left(p_{t,j}^{(T)} \,\middle\|\, p_{s,j}^{(T)}\right) = \alpha \cdot T^2 \sum_{c=0}^{2} p_{t,j,c}^{(T)} \left(\log p_{t,j,c}^{(T)} - \log p_{s,j,c}^{(T)}\right).$$

   Siêu tham số $(T, \alpha)$ được chọn bằng dò lưới 9 tổ hợp $T \in \{1, 2, 4\} \times \alpha \in \{0{,}3;\ 0{,}5;\ 0{,}7\}$ trên tập **validation** ở `seed = 1` (`results/E6/grid.md`), sau đó chạy đủ 3 seeds (`1, 2, 3`) ở tổ hợp tốt nhất. `[HÌNH: sơ đồ luồng chưng cất E6 — nhánh FNS qua Weighted CE và nhánh 2026 qua Qwen P2 letter_logp -> Softmax(T) -> alpha * T^2 * KLDivLoss]`

---

## 2.11 Kết quả tạm trên tập Validation (RQ3)

Bảng dưới đây tổng hợp kết quả trên tập **validation** (`1.791` mẫu của `data/splits/cluster_split.jsonl`) của các mô hình đã hoàn tất tính đến thời điểm hiện tại (mọi con số dẫn nguồn trực tiếp từ `results/`):

| Thí nghiệm | Mô tả & Thay đổi duy nhất so với mốc | Tham số huấn luyện | Macro-F1 validation | Bearish F1 | Neutral F1 | Bullish F1 | Nguồn dữ liệu trong `results/` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`E5`** | DistilBERT fine-tune toàn bộ (mốc RQ3, 3 seeds) | $66.955.779$ (`100%`) | **$0{,}8030 \pm 0{,}0054$** | $0{,}7324 \pm 0{,}0080$ | $0{,}8915 \pm 0{,}0046$ | $0{,}7851 \pm 0{,}0066$ | `results/E5/seed{1,2,3}/run.json` |
| **`E7`** | Đóng băng 6 tầng encoder, chỉ train head (3 seeds) | $592.899$ (`0,89%`) | **$0{,}5913 \pm 0{,}0023$** | $0{,}4402 \pm 0{,}0128$ | $0{,}8022 \pm 0{,}0021$ | $0{,}5315 \pm 0{,}0076$ | `results/E7/seed{1,2,3}/run.json` |
| **`E11` (`P2`)** | `Qwen2.5-7B-Instruct` 4-bit zero-shot (1 seed) | $0$ (zero-shot) | **$0{,}7934$** | $0{,}7535$ | $0{,}8696$ | $0{,}7572$ | `results/llm_prompts/top10_check.json` |
| **`E12`** | `E5` + $1.409$ nhãn cứng `argmax(P2)` (3 seeds) | $66.955.779$ (`100%`) | *(Cập nhật từ `results/E12/`)* | *(Cập nhật)* | *(Cập nhật)* | *(Cập nhật)* | `results/E12/seed{1,2,3}/run.json` |
| **`E6`** | `E5` + $1.409$ nhãn mềm `P2` ở $(T^*, \alpha^*)$ (3 seeds) | $66.955.779$ (`100%`) | *(Cập nhật từ `results/E6/`)* | *(Cập nhật)* | *(Cập nhật)* | *(Cập nhật)* | `results/E6/seed{1,2,3}/run.json`, `results/E6/grid.md` |

**Nhận xét tạm từ bằng chứng trên tập Validation:**
1. **Tinh chỉnh toàn bộ bộ mã hóa Transformer là thiết yếu (`E5` so với `E7`):** Khi đóng băng 6 tầng encoder của DistilBERT (`E7`), Macro-F1 trên validation giảm tới **$21{,}17$ điểm phần trăm** ($0{,}8030 \to 0{,}5913$), thậm chí thấp hơn cả baseline hồi quy logistic `E0` ($0{,}7096$, `results/E0/seed1/run.json`). Sự sụt giảm nghiêm trọng nhất diễn ra ở hai lớp thiểu số `bearish` ($-29{,}22\text{ pp}$) và `bullish` ($-25{,}36\text{ pp}$), chứng tỏ các tầng self-attention phải được cập nhật trọng số để học cách kết hợp từ vựng tài chính và cấu trúc phủ định/so sánh trong tiêu đề tin tức.
2. **Độ lệch lớp tự nhiên của kho tiêu đề không nhãn 2026:** Trong $1.409$ tiêu đề GDELT `train_unlabeled` được chấm bằng prompt `P2`, có tới $1.186/1.409$ ($84{,}17\%$) tiêu đề có nhãn cứng `argmax` là `neutral` (`results/llm_soft_labels/p2_soft_labels.jsonl`). Ở nhiệt độ $T = 1$, trung bình entropy của phân phối mục tiêu chỉ đạt $0{,}0735\text{ nats}$, nhưng khi nâng nhiệt độ lên $T = 2$ và $T = 4$, entropy trung bình tăng lên $0{,}3654\text{ nats}$ và $0{,}7416\text{ nats}$, giúp bộc lộ xác suất thứ cấp ở hai lớp `bearish` và `bullish` cho quá trình chưng cất `E6`.
3. **Giới hạn suy diễn:** Mọi con số ở trên chỉ phản ánh hiệu năng trên tập **validation trong miền (Twitter FNS)**; đánh giá khả năng tổng quát hóa ngoài phân phối trên tập test tiêu đề báo chí 2026 (`300` câu gold do người gán) và Financial PhraseBank sẽ chỉ được thực hiện một lần ở tuần 6.

---

## Tài liệu tham khảo (phần Transformer & Chưng cất)

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*.
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*.
- Hinton, G., Vinyals, O., Dean, J. (2015). Distilling the knowledge in a neural network. *NIPS Deep Learning and Representation Learning Workshop*.
- Qwen Team (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.
- Sanh, V., Debut, L., Chaumond, J., Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS EMC² Workshop (arXiv:1910.01108)*.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. (2017). Attention is all you need. *NeurIPS*.
