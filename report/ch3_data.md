# Chương 3. Dữ liệu và phương pháp xử lý (bản nháp)

## Nguồn và phân chia

Dữ liệu trong miền là Financial News Sentiment (FNS), gồm ba lớp bearish, neutral và bullish. Split theo cụm đã khóa chứa 8.349 train, 1.791 validation và 1.791 test. PhraseBank là tập ngoài nguồn, không dùng để huấn luyện. Tập 2026 lấy ứng viên từ RSS; 300 ID được chọn riêng cho test. GDELT đã lọc được dùng làm kho không nhãn cho huấn luyện. Chi tiết nguồn và revision xem `data/sources.json` và `context/decisions.md`.

Phân bố của split validation là khoảng 15,02% bearish, 64,88% neutral và 20,10% bullish; chênh lệch tỷ lệ nhãn lớn nhất giữa một tập và toàn bộ FNS là 0,027 điểm phần trăm (`data/splits/summary.json`). Mất cân bằng này là lý do dùng Macro-F1 bên cạnh các chỉ số theo lớp. FNS được tải theo revision cố định từ `zeroshot/twitter-financial-news-sentiment`; thông tin giấy phép cần được nhóm kiểm tra theo nguồn trước khi nộp, không suy đoán từ tên bộ dữ liệu.

Theo quyết định ngày 26/10, nguồn RSS cung cấp tiêu đề báo chí cho tập test 2026; tiêu đề GDELT được tạo từ URL slug chỉ dùng trong kho không nhãn sau lọc. Ba mươi ID dùng thử bị loại trước khi lấy mẫu 300 ID với seed 2026 (`labeling/test300_selection.json`). Cần loại các bản gần trùng với tập test khỏi kho huấn luyện không nhãn trước khi chưng cất; giữ các bước kiểm tra này tách khỏi việc chấm test.

## Gần trùng và split cố định

Ứng viên gần trùng được lấy bằng MinHash/LSH và embedding MiniLM; cạnh từ vựng dùng Jaccard ≥ 0,75 và cạnh ngữ nghĩa dùng cosine ≥ 0,92. Thành phần liên thông được đặt trọn trong một tập để giảm rò rỉ. Việc chọn ngưỡng có kiểm tra thủ công và vẫn có giới hạn về khả năng tìm đủ cặp. Split chỉ lưu ID; trước xử lý phải xác minh SHA-256 theo `data/splits/SHA256SUMS`. Hash đảm bảo đúng file, không tự chứng minh dữ liệu không rò rỉ.

Mẫu kiểm tay gồm 50 cặp từ vựng và 50 cặp chỉ ngữ nghĩa: lần lượt 41/50 và 30/50 được đánh dấu đúng ở ngưỡng ban đầu. Với ngưỡng cosine 0,92, 10 cặp được đánh dấu trùng và 2 cặp không trùng được giữ, nhưng 20 cặp được đánh dấu trùng trong mẫu ngữ nghĩa bị bỏ. Nhóm chốt ngưỡng này ngày 14/11; mẫu không đo được chất lượng các cặp dưới 0,85 (`plan/plan_tv1_week2.md`, `context/decisions.md`). Split ngẫu nhiên dùng cùng tỷ lệ để nghiên cứu E9, không thay thế split theo cụm; số cạnh đã lưu qua train–test là 178 (`data/splits/summary.json`), không tự chứng minh mức tăng điểm dự đoán.

## Thí nghiệm E10

Phương án A chốt ngày 23/11 giữ nguyên train gốc và toàn bộ 1.791 ID validation, thêm donor từ validation vào train. Một ID bị lộ khi chính nó hoặc một ID có cạnh cosine trực tiếp ≥ 0,92 nằm trong train. Chọn donor theo seed, ưu tiên cụm có nhiều ID validation, rồi dùng bản sao chính xác; các mức 5%, 10%, 20% lồng nhau, lần lượt 90, 180, 359 ID. Manifest và cách sử dụng xem `results/E10/README.md`. Báo cáo Macro-F1 cho toàn validation, nhóm lộ và nhóm sạch, so ghép cặp với E2/E5 sạch trên cùng ID và seed tương ứng. Bản sao chính xác tạo rò rỉ mạnh hơn gần trùng, do đó cần báo cáo riêng hai nhóm; kết quả không đại diện đơn thuần cho rò rỉ gần trùng tự nhiên. Cạnh dựa trên các cặp ứng viên đã lưu nên có thể bỏ sót cặp.

## Nhãn 2026 và giới hạn

Theo quyết định 23/11, TV2 tự gán tay gold mà không dùng AI và không xem nhãn LLM. Một người thứ hai cần gán tay độc lập 100 ID để tính kappa và phân xử ở tuần 5; nếu không có thì ghi «một người gán, không có kappa». Hiện có 68 nhãn gold TV2, chưa đạt mục tiêu 180. Ba bộ nhãn thử 30 tiêu đề và nhãn đợt 1 TV1/TV3 được phát hiện do LLM tạo, không phải gold; kappa và phân xử từ chúng không phản ánh sự đồng ý của người. Chúng được lưu riêng trong `labeling/llm_annotations/` và chỉ được so với gold đã khóa một lần vào tuần 6. Ba tập test chỉ đánh giá ở tuần 6 sau khi khóa lựa chọn và nhãn.
