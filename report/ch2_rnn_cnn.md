# Chương 2 (phần RNN/CNN) — Mô hình CNN, RNN và attention cho phân loại tiêu đề tài chính

> **Bản nháp (TV2, 23/11/2026).** Phần Transformer và chưng cất do TV3 viết (`report/ch2_transformer.md`).
> Số liệu ở mục 2.7 là **validation**, không phải test; kết quả cuối của RQ2 lấy từ test ở tuần 6.
> Đánh dấu `[HÌNH]` là chỗ cần vẽ hình khi ghép báo cáo.

Bài toán: cho một câu ngắn (tweet hoặc tiêu đề tin tài chính), dự đoán một trong ba nhãn
`bearish` (0), `neutral` (1), `bullish` (2). Mọi mô hình trong phần này có chung ba bước:
(1) biến văn bản thành dãy vector từ, (2) mã hóa dãy thành **một** vector câu, (3) lớp tuyến tính + softmax
cho ba lớp. Các mô hình chỉ khác nhau ở bước (2), nên so sánh giữa chúng trả lời câu hỏi RQ2:
*kiến trúc nào và thành phần nào thật sự giúp?*

## 2.1 Biểu diễn văn bản

**Tách từ** (`models/tokenizer.py`). Văn bản được viết thường (GloVe 6B là bản không phân biệt hoa thường),
rồi thay ba loại token bằng ký hiệu chung: đường link → `<url>`, mã cổ phiếu (`$AAPL`, hoặc mã viết hoa
nằm trong danh sách mã lớn) → `<ticker>`, số → `<num>`. Các từ viết tắt như `AI`, `US`, `CEO`, `GDP`
không bị coi là mã cổ phiếu. Mục đích: mô hình không học thuộc từng mã hay từng con số cụ thể.

**Từ điển** (`models/embeddings.py`) chỉ được dựng trên tập **train** (quy tắc 3 của `AGENTS.md`): 13.289 loại từ,
chỉ số 0 là `<pad>` (vector 0), 1 là `<unk>` (từ chưa gặp). Validation và test chỉ được mã hóa bằng từ điển này.
Câu dài hơn 64 token bị cắt (`max_len: 64`; tweet tài chính hầu hết ngắn hơn nhiều).

**Word embedding.** Mỗi từ ứng với một vector $x_t \in \mathbb{R}^{100}$.

- *GloVe 6B 100d* (Pennington và cộng sự, 2014): vector học trước trên 6 tỷ token Wikipedia + Gigaword,
  từ thống kê đồng xuất hiện. 11.353/13.289 loại từ của train có trong GloVe (thiếu 14,5% theo loại, nhưng chỉ
  2,7% theo token vì từ thiếu thường hiếm, ví dụ `brexit`, `covid`, `wework`). Từ thiếu được khởi tạo ngẫu nhiên
  cùng độ lệch chuẩn với GloVe. Embedding được **tinh chỉnh** cùng mô hình (`embedding_trainable: true`).
- *Ngẫu nhiên* (E4): $x_t \sim \mathcal{N}(0, 1/d)$ với $d = 100$, học từ đầu chỉ trên ~8.000 tweet train.

So E2 với E4 cho biết tri thức ngôn ngữ học trước từ dữ liệu lớn đóng góp bao nhiêu khi dữ liệu có nhãn ít.

## 2.2 CNN cho văn bản — TextCNN (E1)

Theo Kim (2014). Một bộ lọc kích thước $k$ là ma trận $W \in \mathbb{R}^{k \times d}$ trượt trên dãy embedding;
tại vị trí $i$ nó nhìn một "cửa sổ" $k$ từ liên tiếp và cho một số:

$$c_i = \mathrm{ReLU}\big(W \cdot x_{i:i+k-1} + b\big).$$

Mỗi bộ lọc vì vậy là một bộ phát hiện cụm từ $k$-gram (ví dụ tự đặt: "misses revenue forecast", "lifts annual guidance").
**Max-over-time pooling** giữ giá trị lớn nhất theo vị trí: $\hat c = \max_i c_i$ — "cụm từ này có xuất hiện ở đâu đó
trong câu không", bất kể vị trí và độ dài câu.

Cấu hình (`configs/E1.yaml`): kích thước 3, 4, 5, mỗi kích thước 100 bộ lọc → vector câu 300 chiều → dropout 0,3
(Srivastava và cộng sự, 2014) → lớp tuyến tính 3 lớp. Chỉ những cửa sổ bắt đầu trong phần chữ thật mới được
max-pool, nên phần đệm `<pad>` không bao giờ ảnh hưởng kết quả. Số tham số: 1,45 triệu, trong đó 1,33 triệu là
embedding và chỉ ~121 nghìn là phần CNN. `[HÌNH: sơ đồ TextCNN 3 nhánh bộ lọc → max-pool → nối → softmax]`

Điểm mạnh: nhanh (~1 phút/seed trên CPU), song song hoàn toàn. Điểm yếu: chỉ thấy cụm từ tối đa 5 từ và bỏ qua thứ
tự giữa các cụm.

## 2.3 RNN, LSTM và BiLSTM

**RNN** đọc câu từ trái sang phải, cập nhật trạng thái ẩn $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$. Khi lan truyền
ngược qua nhiều bước, gradient bị nhân liên tiếp với $W_h$ nên dễ **biến mất** hoặc **bùng nổ**, khiến RNN thường
khó nhớ thông tin xa.

**LSTM** (Hochreiter và Schmidhuber, 1997) thêm một ô nhớ $c_t$ và ba cổng (giá trị trong $(0,1)$):

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{(cổng quên)}\\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{(cổng vào)}\\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{(cổng ra)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c)\\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

Ô nhớ được cập nhật bằng phép **cộng** có cổng thay vì nhân lặp, nên gradient đi qua nhiều bước ổn định hơn.

**BiLSTM** (Graves và Schmidhuber, 2005) chạy hai LSTM: một chiều xuôi, một chiều ngược, rồi nối
$h_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t]$. Mỗi vị trí vì thế "biết" cả ngữ cảnh trước và sau — quan trọng
với câu (ví dụ tự đặt) như "sales drop, *but* margins hit a record", nơi nghĩa phụ thuộc phần sau câu.

Cấu hình (`configs/E2.yaml`): 2 tầng, 128 chiều mỗi hướng → $h_t \in \mathbb{R}^{256}$, dropout 0,3 giữa các tầng.
Dãy được **đóng gói** (`pack_padded_sequence`) nên phần đệm không đi vào phép đệ quy.

## 2.4 Gộp dãy thành vector câu: attention và max-pooling

BiLSTM cho một vector $h_t$ cho **mỗi** từ; cần gộp thành một vector câu $s$.

**Attention cộng** (E2; theo dạng của Bahdanau và cộng sự, 2015; Yang và cộng sự, 2016). Mỗi vị trí được chấm điểm
bởi một mạng nhỏ, rồi chuẩn hóa bằng softmax **chỉ trên các từ thật** (vị trí đệm bị gán $-\infty$):

$$
e_t = v^\top \tanh(W h_t + b), \qquad
a_t = \frac{\exp(e_t)}{\sum_{j \le n} \exp(e_j)}, \qquad
s = \sum_{t \le n} a_t\, h_t .
$$

$a_t \ge 0$, $\sum_t a_t = 1$: mô hình tự học **từ nào quan trọng** cho nhãn. Trọng số $a_t$ được lưu lại
(`last_attention`) để trực quan hóa ở tuần 7. Lưu ý: trọng số attention cao **không** chứng minh từ đó là nguyên nhân
của dự đoán (Jain và Wallace, 2019); trong báo cáo chỉ dùng hình attention như minh họa định tính.
`[HÌNH: BiLSTM + attention, thanh màu trọng số trên một tiêu đề ví dụ FNS]`

**Max-pooling** (E3): $s_j = \max_{t \le n} h_{t,j}$ cho từng chiều $j$ — không có tham số học thêm.
E3 chỉ khác E2 ở dòng `pooling: max`, nên chênh lệch E2 − E3 đo đúng đóng góp của attention.

Số tham số E2: 2,03 triệu (1,33 triệu embedding; ~632 nghìn BiLSTM + lớp ra; ~66 nghìn cho attention).

## 2.5 Hàm mất mát

Dữ liệu lệch lớp (khoảng 15% bearish, 65% neutral, 20% bullish trong train), nên cross-entropy thường làm mô hình
thiên về `neutral`.

**Weighted cross-entropy** (mốc E2). Trọng số lớp tính **chỉ trên train**: $w_c = \dfrac{N}{K \cdot N_c}$
($N$ mẫu, $K = 3$ lớp, $N_c$ mẫu lớp $c$) — lớp hiếm có trọng số lớn hơn. Với lô $B$:

$$\mathcal{L}_{\text{WCE}} = -\frac{\sum_{i \in B} w_{y_i} \log p_{i,y_i}}{\sum_{i \in B} w_{y_i}} .$$

**Focal loss** (E13; Lin và cộng sự, 2017) nhân thêm hệ số $(1 - p_{i,y_i})^\gamma$:

$$\mathcal{L}_{\text{focal}} = -\frac{\sum_{i} w_{y_i} (1 - p_{i,y_i})^{\gamma} \log p_{i,y_i}}{\sum_i w_{y_i}}, \quad \gamma = 2 .$$

Mẫu dễ ($p_{i,y_i} \to 1$) gần như không đóng góp, mô hình tập trung vào mẫu khó. Với $\gamma = 0$ focal loss trùng
với weighted CE (có test kiểm tra trong `tests/test_losses.py`). E13 giữ nguyên trọng số lớp, chỉ đổi hàm mất mát.

## 2.6 Quy trình huấn luyện chung

Một vòng huấn luyện dùng chung cho mọi mô hình RNN/CNN (`train/loop.py`, `train/run.py`), để khác biệt giữa các thí
nghiệm chỉ đến từ yếu tố được thay đổi:

| Thành phần | Giá trị |
| --- | --- |
| Tối ưu | Adam (Kingma và Ba, 2015), lr = 10⁻³, batch 64 |
| Số epoch | tối đa 20; **early stopping** khi Macro-F1 validation không tăng trong 3 epoch, khôi phục trọng số tốt nhất |
| Điều chuẩn | dropout 0,3 trên embedding, giữa các tầng LSTM và trước lớp ra |
| Độ chính xác số | FP16 tự động (AMP) khi có GPU; trên CPU chạy FP32 |
| Lặp lại | 3 seed (1, 2, 3) cho mỗi thí nghiệm; seed cố định cho Python, NumPy, PyTorch |
| Dữ liệu | split theo cụm đã khóa (`data/splits/cluster_split.jsonl`, kiểm tra SHA-256 trước khi chạy) |
| Lưu vết | mỗi seed ghi `run.json` (commit, cây làm việc sạch hay không, config, phiên bản thư viện, thời gian, đường cong theo epoch) và `preds_validation.jsonl` |

**Thước đo chính: Macro-F1** — trung bình không trọng số của F1 ba lớp:
$\text{F1}_c = \dfrac{2\,TP_c}{2\,TP_c + FP_c + FN_c}$, $\ \text{Macro-F1} = \tfrac13 \sum_c \text{F1}_c$.
Khác với accuracy, Macro-F1 không bị lớp `neutral` chiếm đa số "che" kết quả kém ở bearish/bullish.

**So sánh hai mô hình** trên cùng tập dùng paired bootstrap (lấy mẫu lại 1.000 lần cùng chỉ số cho cả hai mô hình,
lấy khoảng 2,5–97,5% của chênh lệch Macro-F1) và kiểm định McNemar chính xác trên các mẫu hai mô hình dự đoán khác
nhau (`evaluate/stats.py`, TV1). Chênh lệch được coi là có ý nghĩa khi CI 95% không chứa 0.

## 2.7 Thiết kế thí nghiệm RQ2 và kết quả tạm (validation)

Mỗi thí nghiệm copy config E2 và đổi **đúng một** yếu tố (quy tắc 4):

| Thí nghiệm | Thay đổi so với E2 | Macro-F1 validation (3 seed) |
| --- | --- | --- |
| E2 BiLSTM-Attention | — (mốc) | 0,758 ± 0,010 |
| E1 TextCNN | bộ mã hóa CNN thay BiLSTM-Attention | 0,755 ± 0,005 |
| E3 | attention → max-pooling | 0,757 ± 0,007 |
| E4 | GloVe → embedding ngẫu nhiên | 0,706 ± 0,011 |
| E13 | weighted CE → focal loss | 0,754 ± 0,009 |
| E0 (tham chiếu) | TF-IDF + hồi quy logistic | 0,710 (1 seed) |

Nhận xét tạm (chi tiết và CI từng seed: `results/rq2_validation.md`):

- **Embedding học trước là yếu tố quan trọng nhất:** bỏ GloVe giảm ~0,05; CI của cả ba seed nằm dưới 0.
  Không có GloVe, BiLSTM gần như chỉ ngang TF-IDF + LR.
- **Attention, focal loss và lựa chọn CNN/LSTM chưa cho khác biệt rõ:** chênh lệch ≤ 0,005, nhỏ hơn dao động giữa
  các seed, CI chứa 0. Với câu ngắn như tweet, cách gộp vector câu ít quan trọng hơn chất lượng biểu diễn từ.
- Kết luận chính thức chỉ rút ra từ test ở tuần 6.

## Tài liệu tham khảo

- Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR*.
- Graves, A., Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. *Neural Networks*, 18(5–6).
- Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8).
- Jain, S., Wallace, B. C. (2019). Attention is not explanation. *NAACL*.
- Kim, Y. (2014). Convolutional neural networks for sentence classification. *EMNLP*.
- Kingma, D. P., Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). Focal loss for dense object detection. *ICCV*.
- Pennington, J., Socher, R., Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP*.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15.
- Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., Hovy, E. (2016). Hierarchical attention networks for document classification. *NAACL*.
