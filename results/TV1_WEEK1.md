# TV1 — kết quả tuần 1

Thực hiện ngày **01/11/2026**, sau hạn CP2 dự kiến 28/10. Đây là bản bàn giao kỹ thuật,
không phải biên bản họp hay xác nhận nộp bài. Mã nền `947b65105164e5455924bca02efe2e7591657af2`,
nhánh làm việc `tv1/download-data`; các lần chạy có thay đổi chưa commit và đã ghi
`working_tree_dirty: true` trong provenance. Không gán lại ngày hoàn thành về tháng 10.

## Trạng thái từng việc

| Việc | Trạng thái | Bằng chứng |
| --- | --- | --- |
| 1.1 Tải FNS, PhraseBank, GloVe | Hoàn thành | `data/download.py`, revision nguồn cố định; 2 lần chuẩn bị cho SHA-256 giống nhau ở cả 4 đầu ra |
| 1.2 Giấy phép và trích dẫn | Hoàn thành | Bảng có link nguồn gốc và trích dẫn trong `data/README.md` |
| 1.3 EDA | Hoàn thành | `results/eda/team/summary.json` và 2 hình PNG đã kiểm tra trực quan |
| 1.4 GUIDE | Đã bổ sung; chờ duyệt nhóm | 9 ví dụ FNS theo ID, ca khó, quy trình gán độc lập/phân xử; chưa có xác nhận của 3 người nên còn BẢN NHÁP |
| 1.5 Khả thi khử trùng | Hoàn thành phần đo | `results/feasibility/{fns,fns-lexical}/`; quyết định E10 vẫn thuộc cả nhóm |
| 1.6 Proposal | Đã rà nội dung; chưa nộp | PDF có RQ3 ngoài phân phối, chưng cất E6, focal loss E13; còn tên nhóm, họ tên/MSSV, URL repo dạng placeholder |

Không xác nhận CP0/CP1/CP2, nộp Google Classroom, gửi ảnh nhóm, duyệt/gộp PR hay đồng thuận
quy trình nhánh khi chưa có bằng chứng. Đã hỏi người dùng thông tin nhóm/lớp cần nộp và
trạng thái duyệt GUIDE. Không có connector Google Classroom trong phiên làm việc hiện tại.

## Dữ liệu và tính tái lập

| Đầu ra (không commit) | Quy mô | SHA-256 |
| --- | --- | --- |
| `data/raw/fns/train.jsonl` | 9.543 mẫu | `52158609a11e71afd3dab4cd292773bd3397a9e7890c9354fb5b16cebf7acde0` |
| `data/raw/fns/validation.jsonl` | 2.388 mẫu | `a8cb54b82b1a01c767cdc9b63c697483360fe43ab37cc90c68cb90169af4484a` |
| `data/raw/phrasebank/allagree.jsonl` | 2.264 mẫu | `6eaede7aa7f0c6b42b5fbbca64a6fc7e73a450261a487f31143643eaa0def7d8` |
| `data/raw/glove/glove.6B.100d.txt` | 347.117.594 byte | `be4367dd257eb945217234f16307c5c74236b648a222cc0b4ffd0dda6a3350b6` |

Hai lần chạy dùng cùng nguồn ghim revision, lần sau tái sử dụng cache và tạo lại đầu ra.
Chưa có phép chạy trên một máy sạch độc lập. Lịch sử tải, checksum file gốc, phiên bản
Python/thư viện và thời gian nằm trong `results/downloads/`. Chỉ cài các dependency cần
cho phần TV1; không thay thế `requirements.lock` do TV2 phụ trách.

## EDA trên FNS

| Nhãn | Số mẫu | Tỷ lệ |
| --- | --- | --- |
| bearish | 1.789 | 14,99% |
| bullish | 2.398 | 20,10% |
| neutral | 7.744 | 64,91% |

Tổng 11.931 mẫu, gộp **các phần nguồn** train/validation; không tạo split thí nghiệm.
Bearish là lớp thiểu số, phù hợp lý do khảo sát weighted CE/focal loss đã có trong SPEC.
Không fit trọng số lớp ở bước này; trọng số huấn luyện phải tính lại trên train đã chốt.

Tokenizer TV2 ở commit `b40cc8b22509985361d68e9d9fa4b8a1b05ade49` cho độ dài:
median **14**, p95 **28**, p99 **31**, max **49**, 100% ≤ 64. Hướng dẫn cũ nêu p95 = 27;
bản đo này dùng đúng code/hash đã ghi trong summary nên giữ **28**, không sửa kết quả để khớp.
Đây là độ dài tokenizer RNN/CNN trước padding/truncation, không phải WordPiece của DistilBERT.
Không đổi `configs/E2.yaml` và không suy ra độ phủ của tokenizer Transformer.

FNS có 1.804 mẫu chứa `$cashtag` (15,12%), 1.868 mẫu có `<ticker>` sau chuẩn hóa (15,66%;
bao gồm whitelist ticker trần), 5.612 mẫu có URL/`<url>` (47,04%). Hai cách đếm ticker có
định nghĩa khác nhau và được lưu riêng trong JSON.

PhraseBank chỉ đếm: bearish 303, bullish 570, neutral 1.391; **không** phân tích độ dài,
chọn tham số, chạy mô hình hay sửa nhãn trên PhraseBank.

## Khả thi khử trùng và E10

TV2 script tại commit `c66a0dc1eb3ad0fbc184baa488c4b074a484cb6f`, MiniLM tại revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Chạy CPU, seed 1; chi tiết môi trường,
hash mã nguồn và thời gian trong `results/feasibility/run.json`.

| Chỉ số | Jaccard ≥ 0,75 | Jaccard ≥ 0,75 hoặc cosine ≥ 0,85 |
| --- | --- | --- |
| Cặp từ vựng | 391 | 391 |
| Cặp ngữ nghĩa | Không chạy | 1.448 |
| Cặp hợp nhất (không đếm đôi) | 391 | 1.492 |
| Cặp hợp nhất / mọi cặp có thể | 0,0005494% | 0,0020964% |
| Cụm có ≥ 2 mẫu | 192 | 618 |
| Mẫu trong cụm trùng | 430 (3,60%) | 1.518 (12,72%) |
| Cụm lớn nhất | 12 | 28 |
| Σ(k − 1) | 238 | 900 |

MinHash LSH chỉ sinh ứng viên, mỗi ứng viên được xác minh Jaccard chính xác; không bảo đảm
recall 100%. Tìm cặp ngữ nghĩa bằng cosine theo block; cụm là thành phần liên thông, không
đòi mọi cặp trong cùng cụm đều vượt ngưỡng. Kiểm tra tay 100 cặp là việc tuần 2, chưa làm.

Với cỡ test dự kiến ceil(15% × 11.931) = 1.790, trần thô của cấu hình hợp nhất:
900 / 1.790 = **50,28%**; lexical-only là **13,30%**. Đây chỉ là trần đếm theo cụm,
còn phải xét cặp nối trực tiếp, phân bổ cụm, phân tầng nhãn, kích thước train và tập test
chung ở tuần 2. Không đủ căn cứ bảo đảm E10 50% khả thi. Đề nghị nhóm xem lại mức cao nhất
tại CP2; **SPEC và quyết định 10/25/50% vẫn giữ nguyên**, chưa có quyết định thay thế.

## Lệnh tái lập (từ thư mục gốc)

```powershell
# Python 3.11, dependency TV1; requirements.txt chứa danh sách đầy đủ của dự án.
python -m pip install huggingface_hub numpy matplotlib datasketch sentence-transformers
python data/download.py

# Dùng lại phần bàn giao của TV2, không chép/sửa code TV2 trên nhánh TV1.
git worktree add --detach data/cache/tv2-glove b40cc8b22509985361d68e9d9fa4b8a1b05ade49
git worktree add --detach data/cache/tv2-dedup c66a0dc1eb3ad0fbc184baa488c4b074a484cb6f
python scripts/eda.py --tokenizer-path data/cache/tv2-glove/models/tokenizer.py
python scripts/feasibility.py --dedup-script data/cache/tv2-dedup/scripts/dedup.py --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41

# Chỉ các fixture tổng hợp, không đánh giá test dữ liệu.
python -m unittest discover -s tests -v
python -m unittest discover -s data/cache/tv2-dedup/tests -p test_dedup.py -v
```

Không tạo lại worktree nếu đường dẫn đó đã tồn tại. EDA và feasibility từ chối ghi đè báo
cáo cũ; khi tái lập, giữ lại kết quả cũ ở thư mục lưu trữ trước. Phiên bản thư viện thực
tế nằm trong JSON provenance; đây không phải lock môi trường chuẩn của cả nhóm.

## Kiểm chứng và giới hạn bàn giao

- 5/5 kiểm thử TV1 đạt: ánh xạ nhãn, ID/thứ tự/UTF-8, hai bản ghi giống hash, AllAgree latin-1
  và dấu `@` trong câu, từ chối nhãn lạ/ID lặp, EDA và công thức trần rò rỉ.
- 8/8 kiểm thử dedup TV2 đạt với fixture tổng hợp, gồm cặp trùng/không trùng và cosine biên.
- Hai biểu đồ PNG đã xem trực quan. `git diff --check` sạch.
- Dữ liệu/cache, GloVe và trọng số MiniLM nằm dưới đường dẫn bị gitignore; bảng cặp/cụm chỉ
  có ID/điểm. Không tạo/sửa split; không train hay đánh giá ba tập test; không đọc tiêu đề 2026.
- GUIDE cần đồng thuận 3 người; E10 cần nhóm chốt; proposal cần thông tin nhóm và nơi nộp.
  Những mục này chưa được đánh dấu hoàn thành trong PLAN.
