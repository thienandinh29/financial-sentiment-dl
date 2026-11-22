# Mô phỏng phát hiện chênh lệch Macro-F1

Chạy `python -m scripts.simulate_power`; dữ liệu hoàn toàn tổng hợp, không đọc nhãn hay dự đoán test. Bảng `simulation.csv` và cấu hình, seed, hash script, môi trường trong `run.json`.

Giả định: ba lớp có tỷ lệ 25%/50%/25%; mô hình A đúng 70% ở mỗi lớp, khi sai chọn ngẫu nhiên một trong hai lớp khác. Mô hình B giữ nguyên mọi dự đoán đúng của A và sửa một tỷ lệ các lỗi của A; hai mô hình vì vậy tương quan rất cao và B không tạo thêm lỗi. Đây là kịch bản lạc quan, không phải ước lượng sức mạnh thống kê chung. Mỗi điều kiện sinh 100 bộ mẫu độc lập; mỗi bộ dùng 1.000 mẫu bootstrap ghép cặp và CI percentile 95%. Tỷ lệ CI không chứa 0 là ước lượng xác suất phát hiện; 100 lần lặp đem lại sai số Monte Carlo đáng kể.

Nếu chọn quy ước sức mạnh ít nhất 80%, mức nhỏ nhất trong lưới đã thử có chênh Macro-F1 trung bình khoảng **0,0060** ở n=1.791 (98/100 bộ), và **0,0240** ở n=300 (89/100 bộ). Đây chỉ là điểm nhỏ nhất trong lưới và theo mô hình giả định, không phải ngưỡng liên tục hay bảo đảm cho test thực. Với n=300, mức 0,0191 đạt 77/100 nên gần ranh giới; cần thêm lần lặp và điều kiện thực tế hơn trước khi dùng cho thiết kế. Với n=1.791, mức 0,0029 đạt 73/100. Không dùng kết quả này để điều chỉnh test hay chọn mô hình.
