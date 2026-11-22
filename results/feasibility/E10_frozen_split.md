# Khả thi E10 trên split đã khóa

Kiểm tra ngày 22/11 bằng `python -m scripts.audit_leakage_capacity`, chỉ đọc ID của split và cụm, không đọc văn bản, nhãn hay dự đoán test. Split SHA-256 `8c2c35ea789acf5b2bd9c74c5e59fa0df325034422be42ae7b47f2e8c9c46aaa`; cụm tại ngưỡng 0,92 SHA-256 `2e2fe84a52262582bae0cc85fb12a2e9929d01e398280ce759817db8f2f7f84d` (kiểm tra với log tạo cụm). Không có cụm nào đi qua các tập.

| Tập | Số ID | ID trong cụm có từ 2 ID | Trần k−1 trong các cụm | Tỷ lệ trần |
| --- | ---: | ---: | ---: | ---: |
| Validation | 1.791 | 132 | 76 | 4,24% |
| Test | 1.791 | 130 | 74 | 4,13% |

Trần k−1 áp dụng cho cách giả định: lấy ít nhất một ID của mỗi cụm làm nguồn đưa vào train và chỉ đếm những ID còn lại là đối tượng đánh giá. Nó không chứng minh có thể thực hiện E10: một cụm liên thông có thể không nối trực tiếp mọi cặp, và lấy mẫu test vào train dùng nhãn test, xung đột với quy tắc thực nghiệm. Nếu vẫn đánh giá cả ID đã đưa vào train thì đó là một thiết kế khác, có trùng chính xác và dùng nhãn test, không được mặc định cho phép.

Theo cách đếm k−1, mục tiêu 5%, 10%, 20% trên 1.791 cần tối thiểu 90, 180, 359 ID (làm tròn lên), vượt trần 74. Con số 488/1.791 trong quyết định ngày 20/11 là trần thô toàn bộ dữ liệu trước phân bổ cụm, không phải trần của split test đã khóa. Ngay cả khi xét 130 ID test có bất kỳ hàng xóm trong cùng cụm, hai mức 10% và 20% vẫn vượt số này. Giới hạn phụ thuộc đồ thị cặp đã lưu và giả định cụm hiện tại; không được suy rộng thành trần cho dữ liệu ngoài hoặc thiết kế khác.

**Cần nhóm quyết định lại thiết kế trước khi tạo dữ liệu hoặc thay mức E10.** Không sửa split, không tạo dữ liệu rò rỉ, không chạy hay gán nhãn test trong kiểm tra này.
