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

## Nguồn, giấy phép và trích dẫn (xác minh 01/11/2026)

| Nguồn | Giấy phép và nguồn xác minh | Cách ghi nguồn |
| --- | --- | --- |
| Twitter FNS | [Thẻ dữ liệu chính thức: MIT](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | Zeroshot, *Twitter Financial News Sentiment*, kèm URL và revision trong `sources.json`. Thẻ HF khai báo MIT; không suy ra quyền sở hữu của các nhà xuất bản gốc. |
| Financial PhraseBank AllAgree | [Thẻ dữ liệu](https://huggingface.co/datasets/takala/financial_phrasebank), [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | Malo, Sinha, Korhonen, Wallenius & Takala (2014), *Good debt or bad debt: Detecting semantic orientations in economic texts*, JASIST; [bản tác giả](https://arxiv.org/abs/1307.5336). Ghi công, phi thương mại, chia sẻ bản phái sinh theo cùng giấy phép. |
| GloVe 6B 100d | [README Stanford: pretrained vectors theo PDDL](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors-new-2024-vectors), [PDDL 1.0](https://opendatacommons.org/licenses/pddl/1-0/) | Pennington, Socher & Manning (2014), [*GloVe: Global Vectors for Word Representation*, EMNLP](https://aclanthology.org/D14-1162/). Giấy phép vector khác giấy phép Apache-2.0 của mã nguồn GloVe. |
| Tiêu đề 2026 | Quyền văn bản thuộc từng nhà xuất bản; không có giấy phép chung được xác nhận | Không phân phối lại văn bản; chỉ commit ID, URL và nhãn do người gán. |

## Tải lại dữ liệu

Chạy từ gốc repo, Python 3.11; `huggingface_hub` là dependency trực tiếp trong
`requirements.txt`. Nguồn công khai tải được không cần token. Script không đọc hay in `.env`.

```powershell
python data/download.py
# Tùy chọn tải riêng một nguồn:
python data/download.py --only fns phrasebank
```

`data/sources.json` ghim **commit nguồn bất biến**. Script đọc CSV FNS theo đúng thứ tự
gốc, ánh xạ `0=bearish, 1=bullish, 2=neutral`; ID `fns-train-00000` và
`fns-validation-00000` là ID nguồn, **không phải split thí nghiệm**.
PhraseBank đọc trực tiếp zip latin-1, chỉ lấy AllAgree (2.264 câu), tách nhãn ở dấu `@`
cuối. GloVe chỉ giải nén `glove.6B.100d.txt`, không giải nén đường dẫn tùy ý.

Đầu ra: `data/raw/fns/{train,validation}.jsonl`, `data/raw/phrasebank/allagree.jsonl`,
`data/raw/glove/glove.6B.100d.txt`. Cache ở `data/cache/huggingface/`.
Mỗi lần chạy lưu revision, SHA-256 của nguồn/đầu ra và môi trường trong
`results/downloads/{fns,phrasebank,glove}.json`; lịch sử chạy được giữ lại.
Không commit dữ liệu thô/cache. Hai lần chạy ngày 01/11 cho SHA-256 giống nhau ở cả 4 đầu ra.

FNS gần như xếp theo nhãn; lấy mẫu smoke phải xáo trộn với seed, không lấy N dòng đầu.
Tuần 1 không tạo split; không dùng số liệu PhraseBank để chọn cấu hình.

## EDA và kiểm tra khả thi tuần 1

Xem [báo cáo TV1](../results/TV1_WEEK1.md) để biết lệnh dùng các dependency TV2
đã ghim commit, kết quả, giới hạn và các mục cần nhóm xác nhận.
