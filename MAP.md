# MAP — tìm gì ở đâu

Cập nhật file này trong cùng commit mỗi khi tạo hoặc di chuyển file/thư mục.

## File gốc

| File | Nội dung |
| --- | --- |
| `README.md` | Giới thiệu dự án, cài đặt, tái lập (cho người đọc) |
| `AGENTS.md` | Quy tắc chung cho mọi trợ lý AI |
| `CLAUDE.md` | Quy tắc riêng cho Claude Code (nhập AGENTS.md) |
| `CLAUDE.local.md` | Sở thích cá nhân, không commit |
| `SPEC.md` | RQ, dữ liệu, ma trận thí nghiệm E0–E13, tiêu chí hoàn thành |
| `PLAN.md` | Việc từng tuần theo thành viên, có checkbox |
| `MAP.md` | File này |
| `.env.example` | Mẫu biến môi trường; secrets thật nằm trong `.env` (không commit) |
| `.gitattributes` | Giữ xuống dòng cố định cho các file split và dữ liệu nguồn có SHA-256 |
| `.worktreeinclude` | File bị gitignore được sao chép vào mỗi worktree song song (`.env`) |
| `requirements.txt` | Thư viện trực tiếp; `requirements.lock` sau tuần 1 |

## Thư mục

| Thư mục | Nội dung | Phụ trách |
| --- | --- | --- |
| `plan/` | Checklist theo thành viên: `plan_tv1_week2.md` cho tuần 2 của TV1 | TV1 |
| `context/` | Quyết định đã chốt và lý do (`decisions.md`) | Cả nhóm |
| `configs/` | Mỗi thí nghiệm một YAML; `E2.yaml` là mẫu | Theo RQ |
| `data/` | Script tải dữ liệu, `splits/` (split + SHA-256); quy tắc trong `data/README.md` | TV1 |
| `labeling/` | Hướng dẫn gán nhãn, nhãn test 2026 (không kèm văn bản), kappa | TV1 |
| `models/` | TextCNN, BiLSTM-Attention, Transformer, chưng cất; `tokenizer.py` tách từ, `embeddings.py` từ điển + GloVe cho RNN/CNN | TV2, TV3 |
| `train/` | Vòng huấn luyện chung: `data.py` đọc split (kiểm SHA-256), `losses.py`, `loop.py`, `runlog.py`, `run.py` (`python -m train.run --config ...`) | TV2 |
| `evaluate/` | Metric, bootstrap, McNemar, robustness | TV1 |
| `inference/` | ONNX, latency, FastAPI, Dockerfile | TV3, TV2 |
| `app/` | Demo Gradio | TV3 |
| `scripts/` | EDA, tiện ích dữ liệu; `review_pairs.py` chia kiểm tay 100 cặp gần trùng cho nhiều người rồi gộp kết quả (TV2 thêm, TV1 duyệt) | TV1 |
| `tests/` | Test nhỏ cho từng thay đổi hành vi (`python -m unittest`) | Cả nhóm |
| `results/` | Bảng kết quả, log, hình (không checkpoint) | Cả nhóm |
| `docs/` | Proposal, kế hoạch chi tiết (PDF); `WEEK1.md`, `WEEK2.md` việc chi tiết + checkpoint từng tuần; `guides/` hướng dẫn từng bước cho từng thành viên; `meetings/` chương trình + bộ câu hỏi họp "hiểu bài" | Cả nhóm |
| `.claude/` | Cấu hình Claude Code dùng chung (`settings.json`) | Cả nhóm |

## Week 2 TV1 files

- `scripts/sample_pairs.py`: seeded manual-review sample.
- `scripts/pick30.py` and `tests/test_pick30.py`: seeded trial-ID selection and checks.
- `scripts/summarize_manual_check.py` and `tests/test_manual_check.py`: validate and summarize completed human reviews.
- `scripts/trial_agreement.py` and `tests/test_trial_agreement.py`: pairwise agreement and kappa for three complete trial annotation files.
- `results/feasibility/fns/` and `results/feasibility/run.json`: TV1 week 1 cluster and pair artifacts with provenance; `manual_check_100.csv` holds the seeded review sample.
- `scripts/make_splits.py` and `tests/test_make_splits.py`: split generation and checks.
- `scripts/recluster_092.py` and `tests/test_recluster_092.py`: derive clusters at the approved cosine threshold from the preserved pair artifact; outputs are in `results/feasibility/fns-cosine092/`.
- `data/splits/cluster_split.jsonl`, `random_split.jsonl`, `SHA256SUMS`, `summary.json`: proposed splits, checksums and generation metadata, pending group review.
- `tests/test_review_gate.py`: checks that split generation requires the completed sampled review.
- `configs/E0.yaml`, `models/tfidf_lr.py`, `tests/test_tfidf_lr.py`: E0 baseline and checks.

- `labeling/trial30_ids.csv`: 30 trial IDs and URLs sampled from the 2026 candidate pool; exclude these IDs when selecting the final test set.
- `data/raw/headlines_2026/trial30_to_read.txt`, `trial30_blank.csv`: local-only reading copy and blank annotation template for the 30 trial IDs.
- `labeling/trial/tv1.csv`, `tv2.csv`, `tv3.csv`: independent human trial annotations (30 IDs, labels and reasons per file).
- `labeling/trial_agreement.json`: pairwise human trial agreement, Cohen kappa and disagreement IDs.
- `labeling/trial_adjudication.csv`: group conclusions for the five disputed trial IDs; original independent labels remain unchanged.
