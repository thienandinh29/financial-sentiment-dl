# Hướng dẫn tuần 1 cho TV1 (dữ liệu & thống kê)

Hạn: **CP2 tối Thứ Tư 28/10**. Việc gốc và tiêu chí "xong": [docs/WEEK1.md](../WEEK1.md).
Mỗi việc làm trên một nhánh riêng `<ten>/<viec>`, mở PR, nhờ một người khác duyệt.

Trước khi bắt đầu (CP0): clone repo, tạo venv Python 3.11, chạy
`pip install -r requirements.lock` (sau khi PR #6 được gộp; trước đó dùng `requirements.txt`),
chép `.env.example` thành `.env` và điền `HF_TOKEN`. Đọc `AGENTS.md`.

**Ba quy tắc quan trọng nhất cho TV1:**
1. Không commit dữ liệu: mọi thứ tải về nằm trong `data/raw/` (đã gitignore).
2. PhraseBank và tiêu đề 2026 là tập **test**: chỉ đếm số lượng, không dùng để chọn bất cứ thứ gì.
3. Tuần này **chưa tạo split** (split là việc tuần 2).

---

## 1.1 Script tải dữ liệu — `data/download.py`

Nhánh: `<ten>/download-data`. Đã kiểm tra trước (27/10):

| Nguồn | Lấy ở đâu | Ghi chú |
| --- | --- | --- |
| Twitter FNS | `load_dataset("zeroshot/twitter-financial-news-sentiment")` | 2 split: train 9.543, validation 2.388; nhãn **0 = bearish, 1 = bullish, 2 = neutral** |
| PhraseBank AllAgree | file `data/FinancialPhraseBank-v1.0.zip` trong repo HF `takala/financial_phrasebank` | `load_dataset` **không** chạy được (datasets 5.x bỏ loading script) → đọc zip trực tiếp; file `Sentences_AllAgree.txt`, mã hóa latin-1, mỗi dòng `câu@nhãn`; 2.264 câu |
| GloVe 6B | file `glove.6B.zip` (862 MB) trong repo HF `stanfordnlp/glove` | chỉ giải nén `glove.6B.100d.txt` |

Khung script (đủ để chạy; bổ sung docstring và xử lý lỗi theo ý bạn):

```python
"""Download Twitter FNS, PhraseBank AllAgree and GloVe 6B 100d into data/raw/ (never committed)."""
import json
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

RAW = Path("data/raw")
FNS_LABELS = {0: "bearish", 1: "bullish", 2: "neutral"}
PB_LABELS = {"negative": "bearish", "positive": "bullish", "neutral": "neutral"}


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_fns():
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    for split in ds:                       # stable ids: fns-train-00000, fns-validation-00000
        rows = [{"id": f"fns-{split}-{k:05d}", "text": r["text"], "label": FNS_LABELS[r["label"]]}
                for k, r in enumerate(ds[split])]
        write_jsonl(rows, RAW / "fns" / f"{split}.jsonl")


def download_phrasebank():
    zpath = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip",
                            repo_type="dataset")
    with zipfile.ZipFile(zpath) as z:
        lines = z.read("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt").decode("latin-1").splitlines()
    rows = []
    for k, line in enumerate(l for l in lines if l.strip()):
        text, label = line.rsplit("@", 1)
        rows.append({"id": f"pb-{k:05d}", "text": text.strip(), "label": PB_LABELS[label.strip()]})
    write_jsonl(rows, RAW / "phrasebank" / "allagree.jsonl")


def download_glove():
    zpath = hf_hub_download("stanfordnlp/glove", "glove.6B.zip")
    with zipfile.ZipFile(zpath) as z:
        z.extract("glove.6B.100d.txt", RAW / "glove")


if __name__ == "__main__":
    download_fns()
    download_phrasebank()
    download_glove()
```

Xong khi: chạy `python data/download.py` trên máy sạch ra đúng 4 file
(`fns/train.jsonl`, `fns/validation.jsonl`, `phrasebank/allagree.jsonl`, `glove/glove.6B.100d.txt`),
`git status` không hiện file dữ liệu nào. Ghi SHA-256 của 4 file vào mô tả PR để người khác so.

> ID cố định (`fns-train-00000`…) rất quan trọng: tuần 2 split sẽ lưu danh sách ID.
>
> **Lưu ý:** FNS được lưu gần như **xếp theo nhãn** (20 dòng train đầu đều bearish; 64 dòng
> validation đầu không có neutral). Lấy mẫu con luôn phải xáo trộn (có seed) trước, và báo cho cả
> nhóm biết. Split tuần 2 theo cụm nên không bị ảnh hưởng, nhưng mọi "lấy N dòng đầu" đều lệch.

## 1.2 Giấy phép — bảng trong `data/README.md`

Thêm bảng này (đã tra trên thẻ dữ liệu HF ngày 27/10; bạn kiểm tra lại link gốc):

| Nguồn | Giấy phép | Trích dẫn |
| --- | --- | --- |
| Twitter FNS | MIT | thẻ HF `zeroshot/twitter-financial-news-sentiment` |
| Financial PhraseBank | CC BY-NC-SA 3.0 (chỉ phi thương mại — đồ án môn học được) | Malo et al., 2014, *Good debt or bad debt*, JASIST |
| GloVe 6B | PDDL (Public Domain) | Pennington, Socher, Manning, 2014, *GloVe*, EMNLP |
| Tiêu đề 2026 | Văn bản thuộc nhà xuất bản | Không phân phối lại; chỉ commit ID + URL + nhãn |

## 1.3 EDA — `scripts/eda.py`, hình trong `results/eda/`

Chỉ làm EDA đầy đủ trên **FNS** (train + validation). PhraseBank: chỉ đếm số câu và phân bố nhãn.

Cần ra được:
1. Phân bố lớp. Số đã đếm sẵn để đối chiếu: bearish 1.789 (15,0%), bullish 2.398 (20,1%),
   neutral 7.744 (64,9%) → **bearish là lớp thiểu số**; ghi vào báo cáo vì nó lý giải weighted CE và focal loss (E13).
2. Độ dài theo token: dùng tokenizer của nhóm (`from models.tokenizer import tokenize`, PR #4).
   Số đã đo: trung vị 14, p95 27, p99 31, max 49 → `max_len: 64` đủ.
3. Tỷ lệ tweet có `$ticker`, có URL (gợi ý: đếm `"<ticker>" in tokenize(text)`, `"<url>" in ...`).
4. Một biểu đồ cột phân bố lớp, một histogram độ dài → lưu PNG vào `results/eda/`.

## 1.4 Chốt hướng dẫn gán nhãn — `labeling/GUIDE.md`

1. Đọc bản nháp cùng cả nhóm (đã gửi góp ý ở CP1).
2. Điền mục "Ví dụ mẫu": 2–3 ví dụ mỗi nhãn và các ca khó (giá dầu tăng: tốt cho hãng dầu, xấu
   cho hãng hàng không; "beats estimates but cuts guidance"; tin sáp nhập…).
   **Lấy ví dụ từ FNS, không lấy từ tiêu đề 2026** (tiêu đề 2026 có thể vào tập test).
3. Xóa dòng "BẢN NHÁP"; thêm một dòng vào "Lịch sử sửa đổi" (ngày, thay đổi, lý do).
4. Xong khi cả 3 người đồng ý tại CP2.

## 1.5 Tỷ lệ trùng thực tế — quyết định mức rò rỉ E10

Dùng `scripts/dedup.py` (PR #2). Cần file jsonl có `id`, `text` — gộp hai file FNS ở bước 1.1.
Chạy trong **Git Bash** (trong PowerShell 5.1, `>` ghi UTF-16 và làm hỏng file jsonl):

```bash
cat data/raw/fns/train.jsonl data/raw/fns/validation.jsonl > data/raw/fns/all.jsonl
python scripts/dedup.py --input data/raw/fns/all.jsonl --out results/feasibility/fns
python scripts/dedup.py --input data/raw/fns/all.jsonl --out results/feasibility/fns-lexical --no-semantic
```

Chạy ~3 phút trên CPU. Kết quả chạy thử của TV2 (26/10) để đối chiếu: 391 cặp từ vựng,
1.448 cặp ngữ nghĩa, 618 cụm có > 1 phần tử, **12,7% tweet nằm trong cụm trùng**.
`pairs.csv`, `clusters.csv` chỉ chứa ID và điểm → được commit vào `results/feasibility/`.

**Cách suy ra mức rò rỉ khả thi** (đem số này đến CP2):
- Test trong miền ≈ 15% × 11.931 ≈ 1.790 tweet.
- Một tweet test chỉ "rò rỉ" được nếu nó có bản gần trùng nằm trong train, tức là nó thuộc một
  cụm có ≥ 2 phần tử và ít nhất một phần tử khác ở lại train.
- Trần thô: mỗi cụm cỡ k đóng góp tối đa k − 1 tweet test rò rỉ →
  `trần ≈ Σ(k − 1)` trên các cụm có k ≥ 2, chia cho 1.790.
  Với số chạy thử: Σ(k − 1) = 1.518 − 618 = 900 → **trần ≈ 50%**, sát mức cao nhất của E10.
- Vì vậy đề xuất tại CP2: giữ 10% và 25%; mức 50% chỉ giữ nếu số chính thức của bạn cho trần
  rõ ràng trên 50%, nếu không thì hạ (ví dụ 40%) và ghi vào `context/decisions.md` + SPEC.

Tuần 2 bạn còn phải kiểm tra tay 100 cặp — tuần này chỉ cần con số.

## 1.6 Nộp proposal

1. Mở `docs/Proposal.pdf`, đối chiếu với `context/decisions.md`: RQ3 đã đổi thành "ngoài phân phối",
   DAPT đã thay bằng chưng cất (E6), thêm focal loss (E13). Nếu proposal còn ghi cũ, sửa trước khi nộp.
2. Nộp trên Google Classroom, chụp màn hình xác nhận gửi vào nhóm.

---

## Mang đến CP2 (28/10)

- [ ] PR `download-data` (1.1 + 1.2) đã mở, có SHA-256 của 4 file
- [ ] PR `eda` với 2 hình và bảng số (1.3)
- [ ] `labeling/GUIDE.md` bản chốt (1.4)
- [ ] `results/feasibility/fns/summary.json` + trần rò rỉ đã tính (1.5)
- [ ] Ảnh xác nhận nộp proposal (1.6)
