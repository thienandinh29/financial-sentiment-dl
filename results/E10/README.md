# Bàn giao E10: overlay validation

Manifest `manifests/leakage_{5,10,20}.json` và SHA-256 trong `manifests/SHA256SUMS` dùng seed 1, split và cặp/cụm có hash ghi trong từng manifest. Không thay đổi file split. Với mỗi mức: giữ train gốc, lấy thêm mẫu FNS theo `train_add_validation_ids` cùng nhãn công khai; dự đoán toàn bộ `validation_evaluation_ids`. Train và validation có chủ ý trùng ID; loader hiện tại chưa tích hợp overlay, TV2/TV3 cần bổ sung trước khi chạy. Không đọc hoặc đánh giá test.

`validation_exact_ids` là donor; `validation_near_only_ids` là các ID khác có cạnh cosine trực tiếp với donor; hợp của hai nhóm là `validation_exposed_ids`, phần bù là `validation_clean_ids`. Các tập phơi nhiễm và donor lồng nhau. Số lượng (exact / near-only / sạch): 5% = 39 / 51 / 1701; 10% = 113 / 67 / 1611; 20% = 292 / 67 / 1432. Đây là phơi nhiễm theo tập cặp ứng viên đã lưu, không bảo đảm tìm đủ mọi cặp.

Tính Macro-F1 trên toàn validation và trên từng tập ID lộ/sạch; căn chỉnh dự đoán E10 và E2/E5 sạch theo ID và seed trước khi so ghép cặp. Ghi cả số mẫu và phân bố nhãn của từng nhóm; không chọn mô hình bằng test. Kiểm tra SHA-256 manifest trước khi đọc. Tái tạo vào thư mục mới:

```powershell
python -m scripts.make_leakage --split data/splits/cluster_split.jsonl --split-sha256 8c2c35ea789acf5b2bd9c74c5e59fa0df325034422be42ae7b47f2e8c9c46aaa --pairs results/feasibility/fns-cosine092/pairs.csv --pairs-sha256 ef5f5f42e0a188cc257d9f0d91518874285eaafb852f287832a14dee0366d96a --clusters results/feasibility/fns-cosine092/clusters.csv --clusters-sha256 2e2fe84a52262582bae0cc85fb12a2e9929d01e398280ce759817db8f2f7f84d --output-dir <thu-muc-moi> --seed 1
```
