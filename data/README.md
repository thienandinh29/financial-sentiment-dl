# data/

| Đường dẫn | Nội dung | Commit? |
| --- | --- | --- |
| `data/raw/` | Bản tải về của Twitter FNS, PhraseBank, GloVe | Không (script tải lại được) |
| `data/raw/headlines_2026/` | Tiêu đề 2026 xuất từ pipeline thu thập (thành viên 2) | **Không bao giờ**: văn bản thuộc nhà xuất bản |
| `data/splits/` | File split (danh sách ID) và SHA-256 của chúng | Có |
| `labeling/` | Nhãn test 2026 (ID, URL, nhãn), không kèm văn bản | Có |

Quy tắc:

- Split được chốt ở tuần 2. Mọi thí nghiệm đọc cùng file split và kiểm tra SHA-256 trước khi chạy.
- Ba tập test chỉ được đánh giá một lần ở tuần 6.
- Mọi bước fit (từ điển, TF-IDF, ngưỡng) chỉ dùng train/validation.
