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
| `requirements.txt` | Thư viện trực tiếp (sửa file này khi thêm thư viện) |
| `requirements.lock` | Phiên bản ghim chính xác cho mọi lần chạy (`pip install -r requirements.lock`) |

## Đầu ra TV1 tuần 1

| File/đường dẫn | Nội dung |
| --- | --- |
| `data/download.py` | Tải nguồn ghim revision, chuyển JSONL có ID ổn định; chỉ giải nén GloVe 100d |
| `data/sources.json` | Repo, revision bất biến và tên file của ba nguồn dữ liệu |
| `scripts/run_metadata.py` | SHA-256 và provenance Git/config/seed/thư viện/CPU/thời gian |
| `scripts/eda.py` | EDA FNS và đếm PhraseBank; dùng tokenizer TV2, có chế độ whitespace tạm |
| `scripts/feasibility.py` | Chạy script dedup TV2 từ checkout riêng; ghim MiniLM, tính trần rò rỉ thô |
| `tests/test_week1_data.py` | Fixture tổng hợp kiểm tra tải/chuyển đổi, EDA, trần rò rỉ; không đánh giá tập test |
| `results/downloads/{fns,phrasebank,glove}.json` | Lịch sử tải, SHA-256 nguồn/đầu ra và môi trường |
| `results/eda/team/` | `summary.json`, `class_distribution.png`, `length_distribution.png` |
| `results/feasibility/` | `run.json`; `fns/` và `fns-lexical/` chứa `pairs.csv`, `clusters.csv`, `summary.json` (ID/điểm, không văn bản) |
| `results/TV1_WEEK1.md` | Báo cáo kết quả, lệnh tái lập và bàn giao các việc cần nhóm xác nhận |
| `data/cache/tv2-{glove,dedup}/` | Worktree dependency TV2 ở commit cố định, không commit nội dung vào nhánh TV1 |

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
| `evaluate/` | Metric (`metrics.py`), bootstrap, McNemar, robustness | TV1, TV3 |
| `inference/` | ONNX, latency, FastAPI, Dockerfile | TV3, TV2 |
| `app/` | Demo Gradio | TV3 |
| `notebooks/` | Notebook thiết lập Colab, chạy smoke test (`notebooks/colab_setup.ipynb`) và đánh giá prompt P1/P2/P3 (`notebooks/llm_prompts.ipynb`) | TV3 |
| `scripts/` | EDA, tiện ích dữ liệu; `review_pairs.py` chia kiểm tay 100 cặp gần trùng cho nhiều người rồi gộp kết quả (TV2 thêm, TV1 duyệt) | TV1 |
| `tests/` | Test nhỏ cho từng thay đổi hành vi (`python -m unittest`) | Cả nhóm |
| `results/` | Bảng kết quả, log, hình (không checkpoint); `results/smoke/` (`env.json`, `distilbert.json`, `qwen.json`, `gpu_budget.md`, `contribution_report.md`) | Cả nhóm |
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

## Week 2 TV3 files

- `evaluate/metrics.py` and `tests/test_metrics.py`: shared evaluation module (`read_preds`, `score`, `validation_gold`) and hand-computed unit tests.
- `notebooks/colab_setup.ipynb`: updated to read ID-only splits via `train.data.load_split_data` and prepare FNS raw data via `data/download.py --only fns`.
- `configs/llm_prompts.yaml` and `notebooks/llm_prompts.ipynb`: P1/P2/P3 prompt templates and Colab T4 validation evaluation + top-10 logits diagnostic notebook.
- `results/E11/{P1,P2,P3}/preds_validation.jsonl`: validation predictions (1,791 rows each) for `Qwen/Qwen2.5-7B-Instruct` (`nf4-4bit`).
- `results/llm_prompts/top10_check.json` and `results/llm_prompts/contribution_report_week2.md`: validation metrics, ABC token probability mass diagnostics (selecting `P2` with `macro_f1 = 0.7934`), and TV3 Week 2 contribution report.
