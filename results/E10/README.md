# Bàn giao E10: overlay validation

Manifest `manifests/leakage_{5,10,20}.json` và SHA-256 trong `manifests/SHA256SUMS` dùng seed 1, split và cặp/cụm có hash ghi trong từng manifest. Không thay đổi file split. Với mỗi mức: giữ train gốc, lấy thêm mẫu FNS theo `train_add_validation_ids` cùng nhãn công khai; dự đoán toàn bộ `validation_evaluation_ids`. Train và validation có chủ ý trùng ID; loader hiện tại chưa tích hợp overlay, TV2/TV3 cần bổ sung trước khi chạy. Không đọc hoặc đánh giá test.

`validation_exact_ids` là donor; `validation_near_only_ids` là các ID khác có cạnh cosine trực tiếp với donor; hợp của hai nhóm là `validation_exposed_ids`, phần bù là `validation_clean_ids`. Các tập phơi nhiễm và donor lồng nhau. Số lượng (exact / near-only / sạch): 5% = 39 / 51 / 1701; 10% = 113 / 67 / 1611; 20% = 292 / 67 / 1432. Đây là phơi nhiễm theo tập cặp ứng viên đã lưu, không bảo đảm tìm đủ mọi cặp.

Tính Macro-F1 trên toàn validation và trên từng tập ID lộ/sạch; căn chỉnh dự đoán E10 và E2/E5 sạch theo ID và seed trước khi so ghép cặp. Ghi cả số mẫu và phân bố nhãn của từng nhóm; không chọn mô hình bằng test. Kiểm tra SHA-256 manifest trước khi đọc.

Ví dụ sau dùng các hàm đánh giá có sẵn, sau khi đã có dự đoán E10 trên **toàn validation**. Thay hai đường dẫn dự đoán cho cùng seed; hàm `compare` trả chênh lệch E10 trừ baseline và khoảng tin cậy bootstrap. Đoạn này chỉ đọc nhãn validation qua config và từ chối dự đoán thiếu hoặc thừa ID:

```python
import hashlib
import json
from pathlib import Path

import yaml
from evaluate.metrics import read_preds, score, validation_gold
from evaluate.stats import compare
from train.data import LABELS

manifest_path = Path("results/E10/manifests/leakage_5.json")
checksums = dict(line.split()[::-1] for line in
                 (manifest_path.parent / "SHA256SUMS").read_text().splitlines())
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == checksums[manifest_path.name]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
cfg = yaml.safe_load(Path("configs/E2.yaml").read_text(encoding="utf-8"))
gold = validation_gold(cfg)  # kiểm tra split hash; chỉ đọc validation
baseline = read_preds("results/E2/seed1/preds_validation.jsonl")
leaky = read_preds("<predictions-e10-seed1-validation.jsonl>")
assert set(gold) == set(manifest["validation_evaluation_ids"]) == set(baseline) == set(leaky)
exposed, clean = set(manifest["validation_exposed_ids"]), set(manifest["validation_clean_ids"])
assert not exposed & clean and exposed | clean == set(gold)
for name, ids in (("all", set(gold)), ("exposed", exposed), ("clean", clean)):
    y = {i: gold[i] for i in ids}
    a, b = ({i: preds[i] for i in ids} for preds in (baseline, leaky))
    print(name, "class_counts", {label: sum(y[i] == k for i in y)
                                  for k, label in enumerate(LABELS)})
    print("baseline", score(a, y), "E10", score(b, y))
    print("paired", compare(a, b, y, repetitions=1000, seed=1))
```

Với E5 sạch, thay config và file dự đoán bằng `configs/E5.yaml` và `results/E5/seed1/preds_validation.jsonl`. Nếu seed hoặc mức rò rỉ khác, thay các đường dẫn và seed tương ứng; không ghép nhầm seed. Báo cáo các nhóm nhỏ kèm số lượng và khoảng tin cậy, không coi mức chênh quan sát được là kết luận trên test.

Tái tạo manifest vào thư mục mới:

```powershell
python -m scripts.make_leakage --split data/splits/cluster_split.jsonl --split-sha256 8c2c35ea789acf5b2bd9c74c5e59fa0df325034422be42ae7b47f2e8c9c46aaa --pairs results/feasibility/fns-cosine092/pairs.csv --pairs-sha256 ef5f5f42e0a188cc257d9f0d91518874285eaafb852f287832a14dee0366d96a --clusters results/feasibility/fns-cosine092/clusters.csv --clusters-sha256 2e2fe84a52262582bae0cc85fb12a2e9929d01e398280ce759817db8f2f7f84d --output-dir <thu-muc-moi> --seed 1
```
