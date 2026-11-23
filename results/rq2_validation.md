# RQ2 — kết quả tạm trên validation (tuần 3)

**Chỉ là validation.** Tập test chưa chạy và chỉ chạy một lần ở tuần 6 (AGENTS.md quy tắc 1). Bảng này dùng để
theo dõi, không phải kết quả cuối; kết luận RQ2 dựa trên CI 95% của chênh lệch (paired bootstrap) trên test.

Chạy ngày 17/11/2026 trên CPU (laptop TV2), commit `af1fd64` (cây làm việc sạch), split
`data/splits/cluster_split.jsonl` (SHA-256 `8c2c35ea…`, PR #10), GloVe 6B 100d (SHA-256 khớp báo cáo TV1).
Chi tiết từng lần chạy (config, seed, phiên bản thư viện, đường cong validation theo epoch) trong
`results/<E>/seed<k>/run.json`; dự đoán trong `preds_validation.jsonl`.

## Macro-F1 trên validation (1.791 tweet)

| Thí nghiệm | Seed 1 | Seed 2 | Seed 3 | Trung bình ± độ lệch chuẩn | Epoch tốt nhất | Thời gian / seed |
| --- | --- | --- | --- | --- | --- | --- |
| E0 TF-IDF + LR (TV1) | 0,7096 | — | — | 0,7096 (một seed) | — | ~1 giây |
| E1 TextCNN | 0,7597 | 0,7535 | 0,7505 | **0,7546 ± 0,0047** | 11 / 9 / 8 | ~1 phút |
| E2 BiLSTM-Attention | 0,7629 | 0,7658 | 0,7466 | **0,7584 ± 0,0103** | 18 / 9 / 8 | 5–10 phút |
| E3 BiLSTM + max-pooling (bỏ attention) | 0,7512 | 0,7651 | 0,7547 | **0,7570 ± 0,0072** | 7 / 9 / 10 | ~6 phút |
| E4 BiLSTM-Attention + embedding ngẫu nhiên (bỏ GloVe) | 0,7161 | 0,6948 | 0,7075 | **0,7061 ± 0,0107** | 7 / 5 / 6 | ~4–5 phút |

## Ghi chú

- E2 cao hơn E1 khoảng 0,004 điểm trung bình, nhỏ hơn độ dao động giữa các seed (độ lệch chuẩn E2 là 0,010).
  Chưa thể nói kiến trúc nào tốt hơn; cần bootstrap trên test ở tuần 6.
- E2 seed 1 đạt tốt nhất ở epoch 18 và chạy đủ 20 epoch mà early stopping chưa kích hoạt, nên giới hạn
  `max_epochs: 20` có thể đã cắt sớm lần chạy này. Nếu muốn tăng, phải đổi cho **mọi** thí nghiệm RQ2 cùng lúc và
  ghi vào `context/decisions.md` (không đổi riêng một thí nghiệm).
- Độ phủ GloVe: 11.353 / 13.289 loại từ của train có vector (OOV theo loại 14,5%, theo token chỉ 2,7%);
  5,8% token validation là `<unk>`.
- Cả 6 lần chạy dùng cùng từ điển và cùng split; khác nhau chỉ ở seed và kiến trúc.

E0 chạy ngày 22/11 trên CPU, commit `16e44ba`, cây làm việc có thay đổi; chi tiết cấu hình và phiên bản trong `results/E0/seed1/run.json`. Kết quả dùng validation, không phải test.

## Ablation E3, E4 so với E2 (validation, 23/11)

E3, E4 chạy ngày 23/11 trên CPU, commit `277fc66` (cây làm việc sạch), cùng split, cùng từ điển; mỗi config copy E2 và
đổi đúng một dòng (`pooling: max`; `embedding: random`). So ghép cặp theo seed (E2 seed k với E3/E4 seed k) bằng
`evaluate/stats.py`: bootstrap 1.000 lần (seed 1), McNemar chính xác.

| So sánh | Seed | Chênh Macro-F1 (B − E2) | CI 95% bootstrap | McNemar p |
| --- | --- | --- | --- | --- |
| E3 − E2 | 1 | −0,0117 | [−0,0268; 0,0044] | 0,036 |
| E3 − E2 | 2 | −0,0006 | [−0,0119; 0,0121] | 1,000 |
| E3 − E2 | 3 | +0,0080 | [−0,0047; 0,0204] | 0,033 |
| E4 − E2 | 1 | −0,0468 | [−0,0670; −0,0255] | < 0,001 |
| E4 − E2 | 2 | −0,0710 | [−0,0947; −0,0490] | < 0,001 |
| E4 − E2 | 3 | −0,0392 | [−0,0607; −0,0182] | 0,033 |

- **GloVe giúp rõ rệt:** bỏ GloVe (E4) giảm khoảng 0,05 Macro-F1; CI 95% của cả ba seed đều nằm dưới 0.
- **Attention chưa cho thấy lợi ích:** E3 (max-pooling) gần bằng E2 (chênh trung bình −0,001); CI của cả ba seed chứa 0,
  dấu chênh đổi theo seed. McNemar p < 0,05 ở seed 1 và 3 nhưng ngược chiều nhau, nghĩa là hai mô hình sai ở các tweet
  khác nhau chứ không phải một mô hình tốt hơn.
- Đây là validation, chỉ để theo dõi; kết luận RQ2 dựa trên test ở tuần 6.
