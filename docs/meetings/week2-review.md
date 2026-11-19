# Họp "hiểu bài" sau tuần 2

Mục đích: **cả 3 người hiểu và tự giải thích được mọi thứ nhóm đã làm ở tuần 1–2**, không chỉ phần của mình.
Giảng viên yêu cầu nhóm hiểu việc đã làm; biên bản cuộc họp này cũng là minh chứng đóng góp.

Thời điểm: ngay sau CP4 (07/11), hoặc gộp vào CP4. Thời lượng: **90 phút**. Cần: máy có repo đã cài, màn hình chung.

## Quy tắc

1. Giải thích **bằng lời của mình**. Không đọc code hay đọc câu trả lời của AI trên màn hình.
2. Dùng AI để viết code thì được (đã khai báo), nhưng **người đứng tên phải giải thích được từng dòng quan trọng**.
3. Không biết thì nói "chưa biết" — ghi vào mục *Câu hỏi còn mở*, giao người tìm hiểu, trả lời ở họp sau.
4. Demo bằng **lệnh chạy thật**, không slide.

## Chương trình

| Phút | Phần | Ai dẫn | Nội dung |
| --- | --- | --- | --- |
| 0–10 | Bức tranh chung | TV2 | Đề tài, 3 RQ, đường đi của một tiêu đề: dữ liệu → split → mô hình → dự đoán → metric (sơ đồ dưới) |
| 10–25 | Demo TV1 | TV1 | Chạy `make_splits.py`, mở `summary.json`; chạy E0; giải thích vì sao split theo cụm |
| 25–40 | Demo TV2 | TV2 | Chạy `python -m unittest discover -s tests`; đi qua `train/data.py` → `losses.py` → `loop.py` → `run.py` |
| 40–55 | Demo TV3 | TV3 | Chạy `evaluate.metrics` trên dự đoán E0; giải thích logits A/B/C và 3 prompt P1–P3 |
| 55–75 | Hỏi chéo | Cả nhóm | Mỗi người hỏi 2 câu về phần **của người khác** (bộ câu hỏi bên dưới); người được hỏi trả lời không xem tài liệu |
| 75–85 | Câu hỏi còn mở | Cả nhóm | Liệt kê câu chưa trả lời được, giao người tìm hiểu |
| 85–90 | Chốt | TV2 | Ghi biên bản, việc tuần 3 |

## Bức tranh chung (TV2 vẽ lên bảng)

```text
Twitter FNS (11.931 tweet) ──dedup.py──► cụm gần trùng ──make_splits.py──► split 70/15/15 (chỉ ID + SHA-256)
                                                                             │
          ┌──────────────────────────────────────────────────────────────────┤
          ▼                              ▼                                   ▼
   E0 TF-IDF + LR (TV1)       E1/E2… RNN/CNN qua train/run.py (TV2)    E5… DistilBERT, E11 Qwen (TV3)
          └──────────────► preds_validation.jsonl {"id", "probs"} ◄──────────┘
                                        │
                              evaluate/metrics.py → Macro-F1 (validation)
                              (test: chỉ một lần, tuần 6)

Tiêu đề 2026 (Paper 1) ──export_headlines.py──► RSS: ứng viên test (người gán nhãn)
                                               GDELT: kho không nhãn (nhãn LLM, chỉ để train)
```

## Bộ câu hỏi (kèm ý chính cần có)

Người hỏi đọc câu hỏi; người trả lời nói bằng lời mình; người hỏi đối chiếu *ý chính*. Đủ ý chính là đạt.

### Thiết kế thực nghiệm (ai cũng phải trả lời được)

1. **Vì sao tập test chỉ chạy một lần, ở tuần 6?**
   Ý chính: nếu nhìn kết quả test để chọn epoch/siêu tham số/prompt thì test không còn "chưa thấy" →
   số báo cáo lạc quan giả. Mọi lựa chọn làm trên validation.
2. **Vì sao không dùng accuracy làm metric chính mà dùng Macro-F1?**
   Ý chính: neutral chiếm ~65%; mô hình đoán toàn neutral có accuracy ~65% mà vô dụng.
   Macro-F1 lấy trung bình F1 của 3 lớp → lớp thiểu số (bearish ~15%) được tính ngang.
3. **"Fit chỉ trên train" nghĩa là gì? Cho 2 ví dụ trong code của nhóm.**
   Ý chính: mọi thứ "học" từ dữ liệu chỉ được học từ train: từ điển (`build_vocab`), TF-IDF (`fit_transform`
   chỉ trên train), trọng số lớp (`class_weights`). Học trên validation/test là rò rỉ thông tin.
4. **Vì sao kết luận bằng CI 95% của chênh lệch, không chỉ so hai con số?**
   Ý chính: tập test nhỏ (nhất là 300 tiêu đề 2026) → chênh 1–2 điểm có thể do may rủi.
   Paired bootstrap cho khoảng tin cậy; CI chứa 0 thì chưa kết luận được mô hình nào hơn.

### Dữ liệu và rò rỉ (TV1 trình bày, người khác trả lời)

5. **Bản "gần trùng" là gì? Vì sao nó gây rò rỉ?**
   Ý chính: cùng một tin viết lại (đổi vài chữ, thêm dấu câu). Nếu bản này ở train và bản kia ở test,
   mô hình "nhớ" thay vì "hiểu" → điểm test cao giả. RQ1 đo đúng hiện tượng này.
6. **Jaccard 0,75 và cosine 0,85 khác nhau thế nào?**
   Ý chính: Jaccard so tập từ/cặp từ (trùng chữ); cosine so vector nghĩa của MiniLM (trùng ý, có thể khác chữ).
   MinHash LSH chỉ để tìm ứng viên nhanh; Jaccard chính xác mới quyết định.
7. **Split theo cụm khác split ngẫu nhiên ở điểm nào? Số liệu nào chứng minh?**
   Ý chính: theo cụm → cả cụm gần trùng nằm trọn một phía, nên 0 cặp vắt qua train–test.
   Ngẫu nhiên → 359 cặp vắt qua train–test. E9 so hai cách này.
8. **Vì sao file split chỉ chứa ID và có SHA-256?**
   Ý chính: không commit văn bản; SHA-256 bảo đảm mọi người, mọi lần chạy dùng **đúng cùng một split**,
   ai sửa một dòng là code từ chối chạy.
9. **Vì sao 30 tiêu đề gán thử bị loại khỏi tập test?**
   Ý chính: nhóm sửa hướng dẫn gán nhãn dựa trên chính các tiêu đề này → giữ lại là chỉnh quy tắc theo test.
10. **Kappa đo gì? Vì sao không chỉ dùng tỷ lệ đồng ý?**
    Ý chính: kappa trừ phần đồng ý do may rủi (ví dụ cả hai cùng hay chọn neutral). Mục tiêu ≥ 0,6.

### Vòng huấn luyện RNN/CNN (TV2 trình bày, người khác trả lời)

11. **Tokenizer biến `$AAPL` thành gì, và vì sao "AI", "US" không thành `<ticker>`?**
    Ý chính: `<ticker>` — mô hình học "có một mã cổ phiếu" thay vì nhớ từng mã. AI, US, CEO… là từ viết tắt
    thông dụng, biến thành ticker sẽ làm sai nghĩa câu.
12. **GloVe là gì? OOV là gì và nhóm xử lý thế nào?**
    Ý chính: vector từ học sẵn trên kho văn bản lớn (100 chiều), giúp mô hình nhỏ biết "rise ≈ gain".
    OOV = từ GloVe không có → khởi tạo ngẫu nhiên có seed. E4 thay GloVe bằng ngẫu nhiên để đo GloVe giúp bao nhiêu.
13. **`<pad>` là gì, vì sao vector của nó bằng 0 và không được cập nhật?**
    Ý chính: câu dài ngắn khác nhau, phải đệm cho bằng nhau trong một batch. Pad không mang nghĩa,
    nên không được ảnh hưởng kết quả hay được học.
14. **Weighted cross-entropy làm gì? Focal loss khác gì?**
    Ý chính: weighted CE tăng trọng số lỗi ở lớp hiếm (bearish). Focal loss nhân thêm (1 − p)^γ:
    giảm trọng số ví dụ dễ (mô hình đã chắc), tập trung vào ví dụ khó. E13 thử focal thay weighted CE.
15. **Early stopping theo dõi gì, và vì sao lại lấy trọng số của epoch tốt nhất?**
    Ý chính: theo dõi Macro-F1 validation mỗi epoch; 3 epoch không cải thiện thì dừng (tránh overfit).
    Epoch cuối chưa chắc tốt nhất nên khôi phục trọng số epoch có Macro-F1 cao nhất.
16. **Vì sao mỗi thí nghiệm chạy 3 seed?**
    Ý chính: kết quả mạng nơ-ron dao động theo khởi tạo ngẫu nhiên. 3 seed cho thấy độ dao động;
    cùng seed thì kết quả giống hệt (test e2e đã kiểm tra điều này).

### Transformer, LLM và đánh giá (TV3 trình bày, người khác trả lời)

17. **Vì sao lấy xác suất từ logits của token A/B/C thay vì đọc chữ Qwen sinh ra?**
    Ý chính: chữ sinh ra chỉ là một nhãn, không có độ tin cậy; logits cho phân phối xác suất thật trên 3 lựa chọn
    (cần cho nhãn mềm/chưng cất E6). Gộp "A" và " A" vì tokenizer coi là hai token khác nhau.
18. **`abc_mass` thấp nghĩa là gì?**
    Ý chính: mô hình muốn trả lời thứ khác ngoài A/B/C (ví dụ "The") → prompt chưa ép được định dạng,
    xác suất sau chuẩn hóa kém tin cậy.
19. **Chọn prompt P1/P2/P3 trên tập nào? Vì sao ví dụ mẫu của P3 lấy từ train?**
    Ý chính: chọn trên validation, không bao giờ trên test. Ví dụ mẫu lấy từ train để validation vẫn là dữ liệu
    "chưa thấy" khi chấm.
20. **Vì sao mọi mô hình phải ghi dự đoán cùng định dạng `{"id", "probs"}` theo thứ tự bearish, neutral, bullish?**
    Ý chính: để một đoạn code duy nhất chấm điểm, bootstrap, McNemar cho mọi mô hình. FNS gốc dùng thứ tự khác
    (1 = bullish, 2 = neutral) → lẫn thứ tự là so sánh sai mà không ai phát hiện.

## Biên bản (điền trong buổi họp)

- Ngày / giờ / có mặt:
- Ai giải thích phần nào (tên + nội dung):

| # | Câu hỏi | Người trả lời | Đạt / chưa |
| --- | --- | --- | --- |
| | | | |

- **Câu hỏi còn mở** (ai tìm hiểu, hạn):
- Quyết định:
- Việc tuần 3 (theo `PLAN.md`):
