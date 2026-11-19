# Checklist tuần 2 — Thành viên 1

Nguồn phân công và tiêu chí hoàn thành: [docs/WEEK2.md](../docs/WEEK2.md). Các mốc 13/11 và 15/11 là lịch bù đề xuất, chờ cả nhóm duyệt.

## Trạng thái phần việc tuần 2

**Đã chuẩn bị:**

- Chọn 100 cặp với seed 1 (50 từ vựng, 50 chỉ ngữ nghĩa) từ kết quả tuần 1; tạo CSV chỉ chứa ID, điểm và ô chờ người kiểm tra. Văn bản để đọc nằm trong `data/raw/fns/pairs_to_read.txt` trên máy và không commit.
- Viết công cụ tổng hợp kết quả kiểm tay, tạo split, chọn 30 ID, tính đồng thuận/kappa; có kiểm thử trên dữ liệu giả. Công cụ tạo split yêu cầu hoàn tất bảng kiểm tay và từ chối ghi đè split có sẵn.
- Viết cấu hình và mã baseline E0, kiểm tra hash trước khi nạp split, fit TF-IDF và trọng số lớp trên train, ghi dự đoán validation và metadata; đã kiểm thử trên dữ liệu giả.
- Chạy `python -m unittest discover -s tests`: 38 kiểm thử đạt. Đã cập nhật `MAP.md`; chưa chạy đánh giá tập test.

**Đang chờ:**

- Đã kiểm tra 100 cặp: từ vựng 41/50 (82%), chỉ ngữ nghĩa 30/50 (60%); nhóm đã duyệt cosine ≥ 0,92.
- Đã tạo và kiểm chứng hai split theo ngưỡng 0,92; còn chờ nhóm duyệt và đưa lên `main`, sau đó mới chạy E0 chính thức trên validation.
- Nhận kho RSS 2026 từ TV2, chọn và chia sẻ 30 ID, bảo đảm loại khỏi 300 tiêu đề test; ba thành viên gán độc lập, tính kappa và cùng sửa GUIDE.

Đối chiếu GitHub: các PR tuần 1 #7, #6, #2, #5 và #8 vẫn mở; PR tuần 2 #9 cũng
chưa gộp. PR [#10](https://github.com/thienandinh29/financial-sentiment-dl/pull/10)
chứa công việc trên nhánh này và chờ nhóm duyệt sau khi xử lý các phụ thuộc.

## Việc mang sang từ tuần 1

- [x] Nộp proposal sau khi điền tên nhóm, MSSV và URL repo; gửi ảnh xác nhận cho nhóm.
- [ ] Mở PR `tv1/download-data` để người khác duyệt; công việc trên nhánh tuần 2 hiện được đưa vào PR #10.
- [ ] Cùng nhóm duyệt và gộp các PR tuần 1 theo thứ tự trong `docs/WEEK2.md`.

## Trước khi chốt split — CP3

Mẫu kiểm tra đã được chọn với seed 1 từ kết quả trên nhánh `origin/tv1/download-data`.
Người kiểm tra mở `data/raw/fns/pairs_to_read.txt` (file cục bộ bị gitignore), điền
`yes` hoặc `no` vào `results/feasibility/manual_check_100.csv`, sau đó chạy
`python scripts/summarize_manual_check.py` để báo tỷ lệ từng loại. Kết quả là 41/50
cặp từ vựng và 30/50 cặp chỉ ngữ nghĩa đúng; nhóm đã duyệt cosine ≥ 0,92.

Để nhóm cân nhắc đề xuất cosine ≥ 0,92: dựng lại cụm từ `pairs.csv` và toàn bộ ID
trong `clusters.csv` tái hiện đúng cụm hiện tại ở ngưỡng 0,85. Khi giữ các cạnh từ vựng
và lọc cạnh chỉ ngữ nghĩa ở 0,92, số cụm nhiều hơn một ID giảm từ 618 xuống 379,
số ID trong các cụm đó giảm từ 1.518 xuống 867, và cụm lớn nhất giảm từ 28 xuống 15 ID.
Có 314 cụm cũ (860 ID) bị tách thành nhiều cụm. Một cặp có điểm lưu làm tròn 0,92;
điểm tính lại bằng mô hình cố định là 0,9200063 và cặp được giữ. Đây là phân tích
trên các cặp đã lưu, không kiểm chứng mọi cặp có thể có.

- [x] Kiểm tra tay 100 cặp gần trùng: chọn ngẫu nhiên có seed 50 cặp từ vựng và 50 cặp ngữ nghĩa.
- [x] Ghi kết quả vào `results/feasibility/manual_check_100.csv` gồm ID hai tiêu đề, loại cặp, đúng/sai và ghi chú; báo tỷ lệ đúng từng loại.
- [x] Nếu các cặp ngữ nghĩa sai nhiều, đề xuất đổi ngưỡng và ghi quyết định vào `context/decisions.md`.
- [x] Cùng nhóm thống nhất hợp đồng về thứ tự nhãn, định dạng split và file dự đoán.

## Chốt split — CP3

- [x] Tạo split theo cụm 70/15/15 bằng `scripts/make_splits.py` và bổ sung test: mỗi cụm nằm trọn trong một tập, phân bố nhãn mỗi tập lệch tối đa 2 điểm phần trăm so với toàn bộ.
- [x] Tạo `data/splits/cluster_split.jsonl` và `data/splits/SHA256SUMS`; dùng seed cố định, chạy lại để kiểm tra hash giống nhau.
- [x] Tạo `data/splits/random_split.jsonl` cho E9 với cùng tỷ lệ, phân tầng nhãn; có 178 cặp gần trùng bị tách giữa train và test (theo ngưỡng 0,92).
- [ ] Điền `split_sha256` vào `configs/E2.yaml` và cùng nhóm đưa split đã chốt lên `main` qua PR được duyệt.
- [ ] Chọn ngẫu nhiên có seed 30 tiêu đề từ kho `test_candidate` của TV2; lưu ID và URL vào `labeling/trial30_ids.csv`, gửi danh sách cho nhóm và loại các tiêu đề này khỏi 300 tiêu đề test.

## Baseline và gán thử — CP4

Khi đủ ba file nhãn độc lập, chạy `python scripts/trial_agreement.py labeling/trial/<nguoi1>.csv labeling/trial/<nguoi2>.csv labeling/trial/<nguoi3>.csv` để tính đồng ý, kappa và liệt kê ID bất đồng. Công cụ yêu cầu đủ đúng 30 ID và lý do; kappa không xác định được ghi là `null`.

- [x] Tạo mã baseline E0 TF-IDF + Logistic Regression: `configs/E0.yaml` và `models/tfidf_lr.py` (hash còn trống đến khi split được chốt).
- [x] Bổ sung kiểm tra SHA-256 trước khi chạy; mã chỉ fit TF-IDF và tính trọng số lớp trên train (đã kiểm thử bằng dữ liệu giả).
- [ ] Đánh giá E0 trên validation, ghi Macro-F1 và `results/E0/seed1/preds_validation.jsonl`; ghi thông tin lần chạy theo quy tắc dự án.
- [ ] Tự gán nhãn độc lập 30 tiêu đề, không dùng AI; lưu `labeling/trial/<ten>.csv` với các cột `id, label, reason`.
- [ ] Sau khi cả ba người nộp nhãn, tính tỷ lệ đồng ý và kappa từng cặp; liệt kê các trường hợp bất đồng.
- [ ] Cùng nhóm thảo luận bất đồng, sửa `labeling/GUIDE.md` và cập nhật lịch sử sửa đổi.

## Kiểm tra hoàn thành

- [ ] Không commit dữ liệu thô hoặc văn bản tiêu đề 2026; chỉ lưu thông tin được phép.
- [ ] Không đánh giá các tập test trong tuần 2; test chỉ chạy một lần ở tuần 6 sau khi mọi lựa chọn đã chốt.
- [ ] Cập nhật `MAP.md` khi tạo hoặc di chuyển file, bổ sung test cho thay đổi hành vi và đánh dấu công việc hoàn thành trong `PLAN.md`.
