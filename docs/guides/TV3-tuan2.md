# Hướng dẫn tuần 2 cho TV3 (Transformer & đánh giá)

Việc và tiêu chí "xong": [docs/WEEK2.md](../WEEK2.md). Hạn: **CP3 Thứ Tư 04/11**, **CP4 ◆ Thứ Bảy 07/11**.
Phần không cần GPU TV2 đã **chạy thử** ngày 02/11; phần Qwen chỉ thay mô hình bằng hàm giả để kiểm tra
luồng dữ liệu — **chưa chạy Qwen thật**, bạn là người chạy đầu tiên.

Thứ tự hợp lý: **3.7 làm ngay** (không cần GPU, không cần split) → 3.8 → 3.5/3.6 khi TV1 đã chốt split.

Quy tắc nhắc lại: chỉ dùng **validation** để chọn prompt; không chạy gì trên `test`; không lưu văn bản
tiêu đề vào file commit; token chỉ trong Colab Secrets.

**Tên prompt:** gọi ba biến thể là **P1, P2, P3** để không lẫn với ba chữ cái trả lời A/B/C.
A/B/C trùng đúng thứ tự nhãn chung `bearish, neutral, bullish` (hợp đồng trong WEEK2.md).

---

## 3.7 Khung đánh giá chung — `evaluate/metrics.py` (làm trên máy hoặc Colab, hạn CP4)

`evaluate/` là vùng của TV1 (MAP.md): báo TV1 trước và nhờ TV1 duyệt PR.

```python
"""Shared evaluation for every model: preds_<split>.jsonl + gold labels -> Macro-F1, per-class F1, confusion.

Usage:
    python -m evaluate.metrics --preds results/E0/seed1/preds_validation.jsonl --config configs/E0.yaml
Only validation is scored before week 6 (AGENTS.md rule 1).
"""
import argparse
import json
from pathlib import Path

import yaml
from sklearn.metrics import confusion_matrix, f1_score

from train.data import DEFAULT_RAW, LABELS, load_split_data


def read_preds(path):
    """Read {"id", "probs"} lines into dict id -> probs; each probs has 3 values summing to ~1."""
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if len(row["probs"]) != len(LABELS) or abs(sum(row["probs"]) - 1) > 1e-3:
                raise ValueError(f"Bad probs for id {row['id']}: {row['probs']}")
            if row["id"] in preds:
                raise ValueError(f"Duplicate id in predictions: {row['id']}")
            preds[row["id"]] = row["probs"]
    return preds


def score(preds, gold):
    """Score predictions against gold labels (dict id -> label index); ids must match exactly.

    Output: {"n", "macro_f1", "f1": {label: value}, "confusion": rows = gold, cols = predicted}.
    """
    if set(preds) != set(gold):
        raise ValueError(f"Ids differ: {len(set(gold) - set(preds))} missing, {len(set(preds) - set(gold))} extra")
    ids = sorted(gold)
    y_true = [gold[i] for i in ids]
    y_pred = [max(range(len(LABELS)), key=lambda k: preds[i][k]) for i in ids]
    labels = list(range(len(LABELS)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {"n": len(ids),
            "macro_f1": round(float(per_class.mean()), 4),
            "f1": {LABELS[k]: round(float(v), 4) for k, v in enumerate(per_class)},
            "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def validation_gold(cfg, root="."):
    """Return dict id -> label index for the validation split named in a config."""
    root, dcfg = Path(root), cfg["data"]
    data = load_split_data(root / dcfg["split_file"], dcfg["split_sha256"],
                           [root / p for p in dcfg.get("raw_files", DEFAULT_RAW)], splits=("validation",))
    return {i: y for i, _, y in data["validation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preds", required=True)
    parser.add_argument("--config", required=True, help="config whose split gives the gold labels")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(score(read_preds(args.preds), validation_gold(cfg)), indent=2))


if __name__ == "__main__":
    main()
```

Test — `tests/test_metrics.py` (ví dụ tính tay: bearish F1 = 2/3, neutral F1 = 2/3, bullish F1 = 1):

```python
"""Tests for evaluate/metrics.py with a hand-computed example."""
import json
import tempfile
import unittest
from pathlib import Path

from evaluate.metrics import read_preds, score

B, N, U = [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]   # predicts bearish / neutral / bullish


class MetricsTest(unittest.TestCase):
    def test_hand_computed_example(self):
        # gold:  a=bearish b=bearish c=neutral d=bullish
        # pred:  a=bearish b=neutral c=neutral d=bullish
        # bearish F1 = 2*1*0.5/1.5 = 2/3; neutral P=0.5 R=1 -> 2/3; bullish = 1
        result = score({"a": B, "b": N, "c": N, "d": U}, {"a": 0, "b": 0, "c": 1, "d": 2})
        self.assertEqual(result["f1"], {"bearish": 0.6667, "neutral": 0.6667, "bullish": 1.0})
        self.assertEqual(result["macro_f1"], round((2 / 3 + 2 / 3 + 1) / 3, 4))
        self.assertEqual(result["confusion"], [[1, 1, 0], [0, 1, 0], [0, 0, 1]])

    def test_ids_must_match(self):
        with self.assertRaises(ValueError):
            score({"a": B}, {"a": 0, "b": 1})

    def test_read_preds_rejects_bad_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            for rows in ([{"id": "a", "probs": [0.5, 0.5]}], [{"id": "a", "probs": [0.9, 0.9, 0.9]}],
                         [{"id": "a", "probs": B}, {"id": "a", "probs": N}]):
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_preds(path)


if __name__ == "__main__":
    unittest.main()
```

```bash
python -m unittest tests.test_metrics
```

TV2 chạy thử: 3 test đạt; chấm file dự đoán E0 ra Macro-F1 = **0,7371**, khớp số E0 tự tính — nghĩa là
mọi mô hình (E0, E2, E5, E11…) giờ được chấm bằng **một** đoạn code.

## 3.8 Sửa bộ đọc split trong `notebooks/colab_setup.ipynb` (hạn CP4)

File split chỉ có `id` và `split`, không có `text`. Trong ô DistilBERT, thay toàn bộ hàm
`load_train_val_splits` bằng bộ nạp chung của TV2:

```python
from datasets import Dataset
from train.data import load_split_data

def load_train_val_splits(seed, n_train, n_val, split_file="data/splits/cluster_split.jsonl",
                          split_sha256="<SHA của cluster_split.jsonl>"):
    """Smoke-test subsets from the frozen split (SHA-256 checked inside load_split_data)."""
    data = load_split_data(split_file, split_sha256)            # train + validation only
    def to_ds(rows, n):
        ds = Dataset.from_list([{"text": t, "label": y} for _, t, y in rows])
        return ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return to_ds(data["train"], n_train), to_ds(data["validation"], n_val), {
        "source": split_file, "sha256": split_sha256, "verified": True}
```

Lưu ý: nhãn bây giờ theo thứ tự chung (0 = bearish, **1 = neutral, 2 = bullish**), khác nhãn gốc của
FNS trên Hugging Face (1 = bullish, 2 = neutral). Code trên đã đúng thứ tự vì đọc qua `load_split_data`.

Trên Colab cần dữ liệu thô trước: ô cài đặt thêm `!python data/download.py --only fns`.

## 3.5 + 3.6 Ba prompt P1/P2/P3 trên validation (hạn CP4, sau khi split chốt)

Giữ nguyên phần nạp Qwen 4-bit của tuần 1. Sửa `letter_probs` để nhận **cả prompt** thay vì chỉ tiêu đề:

```python
def letter_probs(prompt):
    """Return (probs over bearish/neutral/bullish, mass on A/B/C tokens, top-10 next tokens)."""
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    per_letter = torch.stack([torch.logsumexp(logp[LETTER_IDS[L]], dim=0) for L in LETTERS])
    probs = torch.softmax(per_letter, dim=0)
    top10 = [tok.decode([i]) for i in logits.topk(10).indices.tolist()]
    return dict(zip(LETTERS.values(), probs.tolist())), per_letter.exp().sum().item(), top10
```

Ô prompt (P3 lấy 3 ví dụ mẫu từ **train**, không bao giờ từ validation):

```python
import json
import os
from pathlib import Path

PROMPTS = {
    "P1": ("Classify the sentiment of this financial news headline for the main company or market it is about.\n"
           "Headline: {text}\n"
           "A) bearish\nB) neutral\nC) bullish\n"
           "Answer with one letter only."),
    "P2": ("You label financial news headlines.\n"
           "bearish = a clearly unfavourable development or outlook for the main subject "
           "(price falls, losses, downgrades, fines).\n"
           "bullish = a clearly favourable development or outlook for the main subject "
           "(price rises, beats estimates, upgrades).\n"
           "neutral = financial information with no clear direction.\n"
           "Judge only what the headline says.\n"
           "Headline: {text}\n"
           "A) bearish\nB) neutral\nC) bullish\n"
           "Answer with one letter only."),
    "P3": ("Classify the sentiment of a financial news headline for its main subject.\n"
           "{examples}"
           "Headline: {text}\n"
           "A) bearish\nB) neutral\nC) bullish\n"
           "Answer with one letter only."),
}
LETTER_OF = {0: "A", 1: "B", 2: "C"}          # A/B/C = LABELS order: bearish, neutral, bullish


def few_shot_block(train_examples, per_label=1, seed=1):
    """Pick one train example per label (seeded) and format them for P3; train only, never validation."""
    import random
    rng = random.Random(seed)
    lines = []
    for label in (0, 1, 2):
        pool = sorted(i for i, _, y in train_examples if y == label)
        chosen = rng.sample(pool, per_label)
        text = {i: t for i, t, _ in train_examples}
        lines += [f"Headline: {text[i]}\nAnswer: {LETTER_OF[label]}\n" for i in chosen]
    return "Examples:\n" + "".join(lines) + "\n"


def run_prompt(name, val_examples, letter_probs, examples=""):
    """Score every validation headline with one prompt; return (preds dict, check dict without text)."""
    preds, masses, top1_is_letter = {}, [], 0
    for i, text, _ in val_examples:
        probs, mass, top10 = letter_probs(PROMPTS[name].format(text=text, examples=examples))
        preds[i] = [probs["bearish"], probs["neutral"], probs["bullish"]]
        masses.append(mass)
        top1_is_letter += top10[0].strip() in ("A", "B", "C")
    check = {"prompt": name, "n": len(val_examples), "min_abc_mass": round(min(masses), 4),
             "mean_abc_mass": round(sum(masses) / len(masses), 4),
             "share_top1_is_letter": round(top1_is_letter / len(val_examples), 4)}
    return preds, check


def write_preds(path, preds):
    """Write {"id", "probs"} lines (shared contract)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, p in preds.items():
            f.write(json.dumps({"id": i, "probs": [round(float(x), 6) for x in p]}) + "\n")
```

Ô chạy:

```python
import json
from train.data import load_split_data
from evaluate.metrics import read_preds, score

SPLIT, SHA = "data/splits/cluster_split.jsonl", "<SHA của cluster_split.jsonl>"
data = load_split_data(SPLIT, SHA)
val = data["validation"]
gold = {i: y for i, _, y in val}
examples = few_shot_block(data["train"])
checks = []
for name in PROMPTS:                                            # ~10 phút mỗi prompt trên T4
    preds, check = run_prompt(name, val, letter_probs, examples=examples)
    path = f"results/E11/{name}/preds_validation.jsonl"
    write_preds(path, preds)
    check.update(score(read_preds(path), gold))                  # Macro-F1 trên validation
    checks.append(check)
    print(name, check["macro_f1"], check["min_abc_mass"], check["share_top1_is_letter"])
os.makedirs("results/llm_prompts", exist_ok=True)
json.dump(checks, open("results/llm_prompts/top10_check.json", "w"), indent=2)
```

Đọc kết quả (3.6):
- `share_top1_is_letter`: tỷ lệ mẫu mà token xác suất cao nhất là A/B/C. Gần 1 là tốt.
- `min_abc_mass` / `mean_abc_mass`: phần xác suất đặt vào A/B/C. Thấp (< 0,5) → prompt đó hay "lạc đề".
- Chọn prompt có **Macro-F1 validation** cao nhất (nếu sát nhau, chọn cái có `abc_mass` cao hơn). Ghi lựa
  chọn và lý do vào `context/decisions.md`. Không thử thêm prompt trên test.

TV2 chạy thử phần này với mô hình giả trên 50 dòng validation thật: ghi đúng định dạng dự đoán, P3 có
đúng 3 ví dụ mẫu + tiêu đề, file kiểm tra không chứa văn bản tiêu đề.

**Python 3.13 trên Colab:** sau khi PR #6 gộp, chạy lại ô cài đặt và báo `requirements.lock` cài được không.

---

## Mang đến CP3 (04/11)
- [ ] `evaluate/metrics.py` + test (3.7) — PR mở, nhờ TV1 duyệt
- [ ] Kết quả cài `requirements.lock` trên Colab (nếu #6 đã gộp)

## Mang đến CP4 (07/11)
- [ ] Notebook đọc split dạng ID (3.8)
- [ ] `results/E11/P1..P3/preds_validation.jsonl`, `results/llm_prompts/top10_check.json`, prompt đã chọn (3.5, 3.6)
- [ ] File gán thử 30 tiêu đề của bạn (`labeling/trial/<ten>.csv`)
