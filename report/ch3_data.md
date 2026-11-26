# Chương 3. Dữ liệu và phương pháp xử lý (bản nháp)

## Nguồn và phân chia

Dữ liệu trong miền là Financial News Sentiment (FNS), gồm ba lớp bearish, neutral và bullish. Split theo cụm đã khóa chứa 8.349 train, 1.791 validation và 1.791 test. PhraseBank là tập ngoài nguồn, không dùng để huấn luyện. Tập 2026 lấy ứng viên từ RSS; 300 ID được chọn riêng cho test. GDELT đã lọc được dùng làm kho không nhãn cho huấn luyện. Chi tiết nguồn và revision xem `data/sources.json` và `context/decisions.md`.

## Gần trùng và split cố định

Ứng viên gần trùng được lấy bằng MinHash/LSH và embedding MiniLM; cạnh từ vựng dùng Jaccard ≥ 0,75 và cạnh ngữ nghĩa dùng cosine ≥ 0,92. Thành phần liên thông được đặt trọn trong một tập để giảm rò rỉ. Việc chọn ngưỡng có kiểm tra thủ công và vẫn có giới hạn về khả năng tìm đủ cặp. Split chỉ lưu ID; trước xử lý phải xác minh SHA-256 theo `data/splits/SHA256SUMS`. Hash đảm bảo đúng file, không tự chứng minh dữ liệu không rò rỉ.

## Thí nghiệm E10

Phương án A chốt ngày 23/11 giữ nguyên train gốc và toàn bộ 1.791 ID validation, thêm donor từ validation vào train. Một ID bị lộ khi chính nó hoặc một ID có cạnh cosine trực tiếp ≥ 0,92 nằm trong train. Chọn donor theo seed, ưu tiên cụm có nhiều ID validation, rồi dùng bản sao chính xác; các mức 5%, 10%, 20% lồng nhau, lần lượt 90, 180, 359 ID. Manifest và cách sử dụng xem `results/E10/README.md`. Báo cáo Macro-F1 cho toàn validation, nhóm lộ và nhóm sạch, so ghép cặp với E2/E5 sạch trên cùng ID và seed tương ứng. Bản sao chính xác tạo rò rỉ mạnh hơn gần trùng, do đó cần báo cáo riêng hai nhóm; kết quả không đại diện đơn thuần cho rò rỉ gần trùng tự nhiên. Cạnh dựa trên các cặp ứng viên đã lưu nên có thể bỏ sót cặp.

## Nhãn 2026 và giới hạn

Theo quyết định 23/11, TV2 tự gán tay gold mà không dùng AI và không xem nhãn LLM. Một người thứ hai cần gán tay độc lập 100 ID để tính kappa và phân xử ở tuần 5; nếu không có thì ghi «một người gán, không có kappa». Hiện có 68 nhãn gold TV2, chưa đạt mục tiêu 180. Ba bộ nhãn thử 30 tiêu đề và nhãn đợt 1 TV1/TV3 được phát hiện do LLM tạo, không phải gold; kappa và phân xử từ chúng không phản ánh sự đồng ý của người. Chúng được lưu riêng trong `labeling/llm_annotations/` và chỉ được so với gold đã khóa một lần vào tuần 6. Ba tập test chỉ đánh giá ở tuần 6 sau khi khóa lựa chọn và nhãn.
