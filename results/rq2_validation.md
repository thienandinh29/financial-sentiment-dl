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
| E0 TF-IDF + LR (TV1) | — | | | *chờ TV1 (việc 1.12)* | | |
| E1 TextCNN | 0,7597 | 0,7535 | 0,7505 | **0,7546 ± 0,0047** | 11 / 9 / 8 | ~1 phút |
| E2 BiLSTM-Attention | 0,7629 | 0,7658 | 0,7466 | **0,7584 ± 0,0103** | 18 / 9 / 8 | 5–10 phút |

## Ghi chú

- E2 cao hơn E1 khoảng 0,004 điểm trung bình, nhỏ hơn độ dao động giữa các seed (độ lệch chuẩn E2 là 0,010).
  Chưa thể nói kiến trúc nào tốt hơn; cần bootstrap trên test ở tuần 6.
- E2 seed 1 đạt tốt nhất ở epoch 18 và chạy đủ 20 epoch mà early stopping chưa kích hoạt, nên giới hạn
  `max_epochs: 20` có thể đã cắt sớm lần chạy này. Nếu muốn tăng, phải đổi cho **mọi** thí nghiệm RQ2 cùng lúc và
  ghi vào `context/decisions.md` (không đổi riêng một thí nghiệm).
- Độ phủ GloVe: 11.353 / 13.289 loại từ của train có vector (OOV theo loại 14,5%, theo token chỉ 2,7%);
  5,8% token validation là `<unk>`.
- Cả 6 lần chạy dùng cùng từ điển và cùng split; khác nhau chỉ ở seed và kiến trúc.
