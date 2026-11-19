# Hướng dẫn tuần 2 cho TV1 (dữ liệu & thống kê)

Việc và tiêu chí "xong": [docs/WEEK2.md](../WEEK2.md). Hạn: **CP3 Thứ Tư 04/11**, **CP4 ◆ Thứ Bảy 07/11**.
Mọi đoạn code dưới đây TV2 đã **chạy thử trên dữ liệu thật** ngày 02/11 (số kết quả ghi kèm để bạn đối chiếu).

**Việc của bạn chặn cả nhóm:** split (1.8) là đầu vào của vòng huấn luyện (TV2), prompt (TV3) và E0.
Code split đã sẵn sàng — cố gắng **chốt split ngay ở CP3 (04/11)**, sớm hơn kế hoạch.

Quy tắc nhắc lại: không commit văn bản (FNS thô, tiêu đề 2026); split chỉ chứa ID; mọi thứ ngẫu nhiên đều có seed;
**không đánh giá gì trên phần `test`**.

---

## Bước 0 — chuẩn bị nhánh

Bạn cần `train/data.py` của TV2 (đọc split, kiểm SHA-256). Khi các PR tuần 1 và #9 đã gộp vào `main`:

```bash
git switch main
git pull
git switch -c <ten>/week2
```

Nếu chưa gộp, tạm làm từ nhánh của bạn rồi lấy phần của TV2 (gặp xung đột thì nhắn TV2):

```bash
git fetch origin
git switch -c <ten>/week2 origin/tv1/download-data
git merge origin/Kaorichan/week2
```

Dữ liệu FNS đã có ở `data/raw/fns/` từ tuần 1 (`python data/download.py --only fns` nếu máy mới).

## 1.7 Kiểm tra tay 100 cặp gần trùng (hạn CP3)

Lưu thành `scripts/sample_pairs.py`:

```python
"""Pick 50 lexical + 50 semantic-only near-duplicate pairs (seeded) for a manual check.

Writes results/feasibility/manual_check_100.csv with empty 'is_duplicate' and 'note' columns to
fill by hand, and prints both texts of each pair on screen only (texts are never saved).
"""
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.data import DEFAULT_RAW, load_raw  # noqa: E402

PAIRS = ROOT / "results/feasibility/fns/pairs.csv"
OUT = ROOT / "results/feasibility/manual_check_100.csv"


def main(seed=1, per_type=50):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    with open(PAIRS, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lexical = [r for r in rows if r["jaccard"]]
    semantic_only = [r for r in rows if not r["jaccard"]]
    rng = random.Random(seed)
    picked = ([dict(r, type="lexical") for r in rng.sample(lexical, per_type)]
              + [dict(r, type="semantic_only") for r in rng.sample(semantic_only, per_type)])
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: keep your filled-in answers, do not overwrite")
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id_a", "id_b", "type", "jaccard", "cosine", "is_duplicate", "note"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows({**{k: r[k] for k in ("id_a", "id_b", "type", "jaccard", "cosine")},
                          "is_duplicate": "", "note": ""} for r in picked)
    text = {i: t for i, (t, _) in load_raw([ROOT / p for p in DEFAULT_RAW]).items()}
    for k, r in enumerate(picked, 1):
        print(f"\n[{k}] {r['type']}  jaccard={r['jaccard'] or '-'}  cosine={r['cosine'] or '-'}")
        print("  A:", text[r["id_a"]])
        print("  B:", text[r["id_b"]])


if __name__ == "__main__":
    main()
```

```bash
python scripts/sample_pairs.py > pairs_to_read.txt
```

1. Mở `pairs_to_read.txt` (chỉ trên máy bạn, **không commit** — có văn bản tweet). Mỗi cặp in 2 câu.
2. Mở `results/feasibility/manual_check_100.csv`, điền cột `is_duplicate` = `yes` / `no` cho từng dòng
   (cùng thứ tự với file txt), `note` khi cần (ví dụ "cùng tin, khác công ty").
   *Gần trùng* = cùng một tin/sự kiện viết lại, không phải chỉ cùng chủ đề.
3. Báo ở CP3: tỷ lệ `yes` của 50 cặp `lexical` và của 50 cặp `semantic_only`. Nếu `semantic_only` sai nhiều
   (ví dụ < 70% đúng), đề xuất nâng ngưỡng cosine và ghi `context/decisions.md`.

Commit `scripts/sample_pairs.py` và file CSV đã điền (chỉ ID + điểm + đánh giá). Xóa `pairs_to_read.txt`.

## 1.8 + 1.9 Split theo cụm và split ngẫu nhiên (hạn CP4, cố gắng CP3)

Lưu thành `scripts/make_splits.py`:

```python
"""Make the frozen FNS splits: cluster split (main) and random split (E9), 70/15/15, label-balanced.

Reads data/raw/fns/*.jsonl and the dedup clusters, writes ID-only split files, SHA256SUMS and a
text-free summary. Every near-duplicate cluster lands whole in one split (cluster split).
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.data import DEFAULT_RAW, LABELS, load_raw, write_split  # noqa: E402

RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def assign_groups(groups, labels_of, seed):
    """Put each group (list of ids) whole into one split, keeping each label near 70/15/15.

    Biggest groups first (ties in seeded random order); each goes to the split whose counts
    for the group's labels are furthest below target. Output: dict id -> split name.
    """
    rng = random.Random(seed)
    groups = [sorted(g) for g in groups]
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)                     # stable: equal sizes keep the shuffle
    total = Counter(labels_of[i] for g in groups for i in g)
    target = {s: {lab: r * total[lab] for lab in total} for s, r in RATIOS.items()}
    have = {s: Counter() for s in RATIOS}
    out = {}
    for group in groups:
        need = Counter(labels_of[i] for i in group)
        best = max(RATIOS, key=lambda s: sum((target[s][lab] - have[s][lab]) / target[s][lab] * n
                                             for lab, n in need.items()))
        have[best].update(need)
        out.update((i, best) for i in group)
    return out


def describe(assignment, labels_of):
    """Return per-split size, share and label shares (percent), plus the worst label drift."""
    total = Counter(labels_of.values())
    report, worst = {}, 0.0
    for split in RATIOS:
        ids = [i for i, s in assignment.items() if s == split]
        counts = Counter(labels_of[i] for i in ids)
        shares = {LABELS[lab]: round(100 * counts[lab] / len(ids), 2) for lab in sorted(total)}
        for lab in total:
            worst = max(worst, abs(100 * counts[lab] / len(ids) - 100 * total[lab] / len(labels_of)))
        report[split] = {"size": len(ids), "share": round(100 * len(ids) / len(labels_of), 2), "labels": shares}
    return report, round(worst, 2)


def crossing_pairs(pairs_csv, assignment):
    """Count near-duplicate pairs whose two ids sit in different splits, by split pair."""
    crossing = Counter()
    with open(pairs_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            a, b = assignment[row["id_a"]], assignment[row["id_b"]]
            if a != b:
                crossing["-".join(sorted((a, b)))] += 1
    return dict(sorted(crossing.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clusters", default="results/feasibility/fns/clusters.csv")
    parser.add_argument("--pairs", default="results/feasibility/fns/pairs.csv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="data/splits")
    args = parser.parse_args()
    raw = load_raw([ROOT / p for p in DEFAULT_RAW])
    labels_of = {i: label for i, (_, label) in raw.items()}
    clusters = defaultdict(list)
    with open(ROOT / args.clusters, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            clusters[row["cluster_id"]].append(row["id"])
    if sorted(i for g in clusters.values() for i in g) != sorted(labels_of):
        raise ValueError("Cluster file and raw files do not hold the same ids")
    out, summary, sums = ROOT / args.out, {"seed": args.seed}, []
    for name, groups in (("cluster_split", list(clusters.values())), ("random_split", [[i] for i in labels_of])):
        assignment = assign_groups(groups, labels_of, args.seed)
        sha = write_split(out / f"{name}.jsonl", sorted(assignment.items()))
        report, worst = describe(assignment, labels_of)
        summary[name] = {"sha256": sha, "splits": report, "max_label_drift_points": worst,
                         "near_duplicate_pairs_across_splits": crossing_pairs(ROOT / args.pairs, assignment)}
        sums.append(f"{sha}  {name}.jsonl")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

```bash
python scripts/make_splits.py
```

Ý tưởng: cụm lớn xếp trước; mỗi cụm vào phần nào đang **thiếu nhất** so với mục tiêu 70/15/15 của các nhãn
trong cụm → cả cụm nằm trọn một phía và tỷ lệ nhãn ba phía gần như bằng nhau.

Kết quả TV2 chạy thử (seed 1) — bạn chạy phải ra **đúng các số này**:

| | Cluster split | Random split |
| --- | --- | --- |
| SHA-256 | `ce76573d9171b2a2e8c9e10039fe23c08b09a9d2b1aa37ec14c7e48b1eb2bbfe` | `1d598f73d6ff6e8cd5244f8b785a6cd9a354f50ea72f907d63e2227afee5a07b` |
| train / validation / test | 8.349 / 1.791 / 1.791 | 8.349 / 1.791 / 1.791 |
| Lệch tỷ lệ nhãn lớn nhất | 0,03 điểm % | 0,03 điểm % |
| Cặp gần trùng nằm ở 2 phần khác nhau | **0** | train–test 359, train–validation 283, test–validation 60 |

Cột phải chính là điều RQ1 đo: split ngẫu nhiên để 359 cặp gần trùng vắt qua train–test (rò rỉ), split theo cụm thì không.

Sau khi chạy:
1. Chạy lại lần hai, kiểm tra `data/splits/SHA256SUMS` không đổi.
2. Điền SHA-256 của `cluster_split.jsonl` vào `split_sha256` trong `configs/E2.yaml` (dòng `split_sha256: null`).
3. Commit `scripts/make_splits.py`, `data/splits/cluster_split.jsonl`, `random_split.jsonl`, `SHA256SUMS`,
   `summary.json`. **Từ lúc này split là cố định** — không ai sửa (AGENTS.md quy tắc 2).
4. Viết một test nhỏ `tests/test_make_splits.py`: 3 cụm giả, kiểm tra mỗi cụm nằm trọn một phần và
   `assign_groups` cùng seed cho cùng kết quả.

## 1.10 Baseline E0 — TF-IDF + Logistic Regression (hạn CP4)

`configs/E0.yaml` (thay `<SHA>` bằng SHA-256 của `cluster_split.jsonl`):

```yaml
# E0 — TF-IDF + logistic regression (reference baseline, 1 seed).
experiment: E0
rq: reference

data:
  split_file: data/splits/cluster_split.jsonl
  split_sha256: <SHA>

model:
  name: tfidf_lr
  ngram_range: [1, 2]
  min_df: 2
  C: 1.0

seeds: [1]
```

Lưu thành `models/tfidf_lr.py`:

```python
"""E0 baseline: TF-IDF + logistic regression, fit on train only, scored on validation only.

Usage:
    python -m models.tfidf_lr --config configs/E0.yaml
Writes results/E0/seed1/preds_validation.jsonl ({"id", "probs"} in LABELS order) and run.json.
"""
import argparse
import json
import platform
from pathlib import Path

import sklearn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from train.data import DEFAULT_RAW, LABELS, load_split_data


def run(cfg, root="."):
    """Fit on train, predict validation, write the contract files; return validation Macro-F1."""
    root, dcfg, mcfg = Path(root), cfg["data"], cfg["model"]
    data = load_split_data(root / dcfg["split_file"], dcfg["split_sha256"],
                           [root / p for p in dcfg.get("raw_files", DEFAULT_RAW)])
    train, val = data["train"], data["validation"]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=tuple(mcfg["ngram_range"]),
                                 min_df=mcfg["min_df"], sublinear_tf=True)
    x_train = vectorizer.fit_transform([t for _, t, _ in train])           # fit on train only
    clf = LogisticRegression(C=mcfg["C"], class_weight="balanced", max_iter=2000, random_state=cfg["seeds"][0])
    clf.fit(x_train, [y for _, _, y in train])
    assert list(clf.classes_) == [0, 1, 2], "classes must follow LABELS order"
    probs = clf.predict_proba(vectorizer.transform([t for _, t, _ in val]))
    f1 = f1_score([y for _, _, y in val], probs.argmax(1), average="macro")
    out = root / "results" / cfg["experiment"] / f"seed{cfg['seeds'][0]}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preds_validation.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for (i, _, _), p in zip(val, probs):
            f.write(json.dumps({"id": i, "probs": [round(float(x), 6) for x in p]}) + "\n")
    run_info = {"config": cfg, "label_order": list(LABELS), "val_macro_f1": round(float(f1), 4),
                "vocab_size": len(vectorizer.vocabulary_), "python": platform.python_version(),
                "scikit_learn": sklearn.__version__}
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    return f1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(f"E0 validation Macro-F1: {run(cfg):.4f}")


if __name__ == "__main__":
    main()
```

```bash
python -m models.tfidf_lr --config configs/E0.yaml
```

TV2 chạy thử: **Macro-F1 validation = 0,7371** (1.791 dòng dự đoán, không có dòng test nào).
TF-IDF và trọng số lớp chỉ học từ train. Commit `configs/E0.yaml`, `models/tfidf_lr.py` và `results/E0/`.

## 1.11 Chọn 30 tiêu đề gán thử (hạn CP3)

Cần file `data/raw/headlines_2026/headlines.jsonl` — **TV2 gửi riêng cho bạn** (không commit, có văn bản báo).

```python
import csv
import json
import random
from pathlib import Path

rows = [json.loads(line) for line in open("data/raw/headlines_2026/headlines.jsonl", encoding="utf-8")]
pool = sorted(r["id"] for r in rows if r["pool"] == "test_candidate")       # RSS only
url = {r["id"]: r["url"] for r in rows}
picked = random.Random(2026).sample(pool, 30)                                # fixed seed
Path("labeling").mkdir(exist_ok=True)
with open("labeling/trial30_ids.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["id", "url"])
    writer.writerows((i, url[i]) for i in picked)
print(f"pool {len(pool)}, picked {len(set(picked))}")
```

Ra `labeling/trial30_ids.csv` (chỉ `id, url`). TV2 chạy thử: kho có 759 tiêu đề RSS, chọn 30 không trùng.
**30 tiêu đề này sẽ bị loại khỏi 300 tiêu đề test** (đề xuất trong `decisions.md`).

## Gán thử 30 tiêu đề + kappa (cả nhóm, hạn CP4)

1. Gửi `trial30_ids.csv` cho cả nhóm. Mỗi người tự gán **độc lập, không AI**, lưu
   `labeling/trial/<ten>.csv` với cột `id,label,reason`.
2. Khi đủ 3 file, tính đồng ý từng cặp người gán:

```python
import csv
from itertools import combinations
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

files = sorted(Path("labeling/trial").glob("*.csv"))
labels = {f.stem: {r["id"]: r["label"].strip().lower() for r in csv.DictReader(open(f, encoding="utf-8"))}
          for f in files}
ids = sorted(set.intersection(*(set(v) for v in labels.values())))
print(f"{len(files)} annotators, {len(ids)} common ids")
for a, b in combinations(labels, 2):
    x, y = [labels[a][i] for i in ids], [labels[b][i] for i in ids]
    agree = sum(p == q for p, q in zip(x, y)) / len(ids)
    print(f"{a} vs {b}: agreement {agree:.2f}, kappa {cohen_kappa_score(x, y):.3f}")
disagree = [i for i in ids if len({labels[n][i] for n in labels}) > 1]
print("ids to discuss:", disagree)
```

3. Họp 30 phút về các tiêu đề bất đồng → sửa `labeling/GUIDE.md` + thêm dòng "Lịch sử sửa đổi".

---

## Mang đến CP3 (04/11)
- [ ] `manual_check_100.csv` đã điền + tỷ lệ đúng hai loại (1.7)
- [ ] `trial30_ids.csv` đã gửi cả nhóm (1.11)
- [ ] (nếu kịp) split đã chốt + SHA-256 trong `configs/E2.yaml` (1.8, 1.9)

## Mang đến CP4 (07/11)
- [ ] Split đã chốt, có test (1.8, 1.9)
- [ ] E0 + Macro-F1 validation (1.10)
- [ ] Kappa gán thử + GUIDE đã sửa
