# Tuần 1 (Thứ Năm 22/10 – Thứ Tư 28/10/2026) — việc chi tiết và checkpoint

Mốc ◆ của tuần: **RQ và ma trận thí nghiệm chốt trước mọi lần huấn luyện.**
Tuần này không huấn luyện mô hình thật, không tạo split (split là việc tuần 2).
Việc gốc nằm ở [PLAN.md](../PLAN.md); file này chia nhỏ thành đầu ra và tiêu chí "xong".
Hướng dẫn từng bước kèm code: [TV1](guides/TV1-tuan1.md), [TV3](guides/TV3-tuan1.md).

TV1 = dữ liệu & thống kê · TV2 = RNN/CNN · TV3 = Transformer & demo.

## Lịch checkpoint

| CP | Thời điểm | Hình thức | Phải có |
| --- | --- | --- | --- |
| CP0 | Thứ Năm 22/10, tối | Họp 20 phút | Mọi người clone repo, tạo `.env`, cài `requirements.txt`, tạo nhánh việc đầu tiên |
| CP1 | Chủ Nhật 25/10, tối | Họp 30 phút | Dữ liệu tải được; tiêu đề 2026 đã xuất; Colab chạy được; bản nháp GUIDE đã rà |
| CP2 ◆ | Thứ Tư 28/10, tối | Họp 45 phút | Tỷ lệ trùng thực tế → chốt mức rò rỉ E10; GUIDE chốt; proposal đã nộp; PR tuần 1 đã gộp |

Không đạt CP2 thì không sang tuần 2. Việc trễ ghi lý do vào biên bản CP và dời vào tuần 2
trước khi nhận việc mới.

## TV1 — dữ liệu & thống kê

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 1.1 | Script tải Twitter FNS, PhraseBank AllAgree, GloVe 6B 100d | `data/download.py` | Chạy lại trên máy sạch ra `data/raw/` giống nhau; không commit dữ liệu | CP1 |
| 1.2 | Kiểm tra giấy phép từng nguồn | Bảng giấy phép trong `data/README.md` | Mỗi nguồn có link giấy phép và cách trích dẫn | CP1 |
| 1.3 | EDA: số mẫu, phân bố lớp, độ dài, tỷ lệ `$ticker`, URL | `scripts/eda.py`, hình trong `results/eda/` | Biết lớp thiểu số và độ dài phủ 95% (kiểm tra `max_len: 64`) | CP1 |
| 1.4 | Rà và chốt hướng dẫn gán nhãn | `labeling/GUIDE.md` bỏ chữ "BẢN NHÁP", có ví dụ khó | Cả 3 người đồng ý tại CP2; ghi dòng vào "Lịch sử sửa đổi" | CP2 |
| 1.5 | Chạy script khử trùng lặp (từ 2.2) trên toàn bộ Twitter FNS | Bảng tỷ lệ trùng trong `results/feasibility/` | Có tỷ lệ cặp/cụm gần trùng theo Jaccard 0.75 và cosine 0.85 | CP2 |
| 1.6 | Nộp proposal | `docs/Proposal.pdf` bản cuối | Đã nộp, ảnh chụp xác nhận gửi nhóm | CP2 |

## TV2 — RNN/CNN

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 2.1 | Xuất tiêu đề 2026 từ pipeline Paper 1 | `data/raw/headlines_2026/` (không commit) + `scripts/export_headlines.py` chỉ đọc | ~3.300 tiêu đề có ID, URL, ngày; không ghi gì ngược vào dự án Paper 1 | CP1 |
| 2.2 | Đóng gói script khử trùng lặp (MinHash LSH + MiniLM) | `scripts/dedup.py` + test nhỏ | Nhận file văn bản, trả cặp/cụm theo ngưỡng trong SPEC; test có 1 cặp trùng, 1 cặp không | CP1 |
| 2.3 | Tokenizer: chuẩn hóa, `<ticker>`, `<url>`, `<num>` | `models/tokenizer.py` + `tests/test_tokenizer.py` | `AI`, `IT`, `US`, `CEO`, `GDP`, `ON`, `ALL` không bị thành `<ticker>`; `$AAPL` thì có | CP2 |
| 2.4 | Nạp GloVe vào ma trận embedding | `models/embeddings.py` + test | Từ điển chỉ fit trên train; báo tỷ lệ OOV; `<pad>` = 0, `<unk>` ngẫu nhiên cố định seed | CP2 |
| 2.5 | Khóa phiên bản thư viện | `requirements.lock` | `pip install -r requirements.lock` chạy được trên Python 3.11 | CP2 |

Lưu ý 2.4: tuần này chưa có split, nên viết hàm nhận danh sách văn bản train làm tham số,
không tự đọc toàn bộ FNS.

## TV3 — Transformer & demo

| # | Việc | Đầu ra | Xong khi | CP |
| --- | --- | --- | --- | --- |
| 3.1 | Notebook Colab: cài môi trường, nạp `.env` qua Colab Secrets | `notebooks/colab_setup.ipynb` | Không in `HF_TOKEN`; ghi phiên bản GPU, CUDA, torch | CP1 |
| 3.2 | Chạy thử DistilBERT trên 1.000 mẫu FNS, 1 epoch | Log trong `results/smoke/` | Chạy hết không lỗi; ghi thời gian/epoch và bộ nhớ GPU T4 (để ước tính E5–E8) | CP1 |
| 3.3 | Chạy thử Qwen2.5-7B 4-bit, đọc logits token A/B/C | Log trong `results/smoke/` | Gộp biến thể có/không khoảng trắng; 20 tiêu đề FNS ra xác suất tổng = 1 | CP2 |
| 3.4 | Ước tính chi phí GPU cho E5–E8, E11, E12 | Bảng trong biên bản CP2 | Tổng giờ T4 nằm trong hạn mức Colab/Kaggle của nhóm | CP2 |

Lưu ý 3.3: chỉ chạy trên tiêu đề FNS hoặc tiêu đề 2026 không nhãn — không bao giờ trên
tập test 2026.

## Việc cả nhóm

| Việc | Người chính | Xong khi | CP |
| --- | --- | --- | --- |
| Kiểm tra khả thi: tỷ lệ trùng thực tế trong Twitter FNS | TV1 chạy, TV2 cấp script | Chọn mức rò rỉ E10 (mặc định 10/25/50%); nếu đổi thì ghi `context/decisions.md` và SPEC | CP2 |
| Chốt RQ và ma trận E0–E13 | Cả nhóm | Không ai còn đề xuất đổi; mọi thay đổi đã vào `decisions.md` | CP2 |
| Nộp proposal | TV1 | Đã nộp | CP2 |
| Gộp PR tuần 1 | Mỗi người | Mỗi PR có 1 người khác duyệt; MAP.md cập nhật cho file mới | CP2 |

## Phụ thuộc giữa các việc

- 2.2 (script khử trùng) → 1.5 (tỷ lệ trùng) → quyết định mức rò rỉ E10. Hạn 2.2 là CP1.
- 1.1 (tải dữ liệu) → 1.3, 1.5, 2.3, 3.2. TV2, TV3 có thể dùng `datasets` tải tạm FNS trước.
- 1.3 (EDA) → kiểm tra `max_len` và trọng số lớp trong `configs/E2.yaml`.

## Biên bản checkpoint

Mỗi CP ghi một mục: ngày, ai có mặt, việc xong/trễ, quyết định. Quyết định lâu dài chép thêm
vào `context/decisions.md`.

### CP0 — 22/10

- [ ] TV1 đã cài môi trường, có nhánh `<ten>/download-data`
- [ ] TV2 đã cài môi trường, có nhánh `<ten>/export-headlines`
- [ ] TV3 đã cài môi trường, có nhánh `<ten>/colab-smoke`
- [ ] Mỗi người đã đọc AGENTS.md (quy tắc test một lần, không commit dữ liệu)

### CP1 — 25/10

- [ ] 1.1, 1.2, 1.3
- [ ] 2.1, 2.2
- [ ] 3.1, 3.2
- [ ] Mỗi người đã đọc GUIDE.md và gửi góp ý

### CP2 ◆ — 28/10

- [ ] 1.4, 1.5, 1.6
- [ ] 2.3, 2.4, 2.5
- [ ] 3.3, 3.4
- [ ] Mức rò rỉ E10 đã chốt và ghi vào `decisions.md`
- [ ] PLAN.md tuần 1 đã đánh dấu `[x]`
