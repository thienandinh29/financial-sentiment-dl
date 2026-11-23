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
| `plan/` | Checklist TV1: `plan_tv1_week2.md` và `plan_tv1_week3.md` | TV1 |
| `context/` | Quyết định đã chốt và lý do (`decisions.md`) | Cả nhóm |
| `configs/` | Mỗi thí nghiệm một YAML; `E2.yaml` là mẫu; `E1.yaml` (TextCNN); `E3.yaml` (BiLSTM max-pooling), `E4.yaml` (embedding ngẫu nhiên), `E13.yaml` (focal loss); `E5.yaml` (DistilBERT fine-tune); `E7.yaml` (DistilBERT đóng băng encoder) | Theo RQ |
| `data/` | Script tải dữ liệu, `splits/` (split + SHA-256); quy tắc trong `data/README.md` | TV1 |
| `labeling/` | Hướng dẫn gán nhãn, nhãn test 2026 (không kèm văn bản), kappa | TV1 |
| `models/` | TextCNN, BiLSTM-Attention, Transformer (`transformer.py` cho E5, E7, E8), nhãn mềm LLM (`llm_soft_labels.py`), chưng cất; `tokenizer.py` tách từ, `embeddings.py` từ điển + GloVe cho RNN/CNN; `registry.py` tên mô hình → hàm dựng; `textcnn.py` (E1); `bilstm_attention.py` (E2, E3) | TV2, TV3 |
| `train/` | Vòng huấn luyện chung: `data.py` đọc split (kiểm SHA-256), `losses.py`, `loop.py`, `runlog.py`, `run.py` (`python -m train.run --config ...`) | TV2 |
| `evaluate/` | Metric (`metrics.py`), bootstrap, McNemar, robustness | TV1, TV3 |
| `inference/` | ONNX, latency, FastAPI, Dockerfile | TV3, TV2 |
| `app/` | Demo Gradio | TV3 |
| `notebooks/` | Notebook thiết lập Colab (`notebooks/colab_setup.ipynb`), đánh giá prompt P1/P2/P3 (`notebooks/llm_prompts.ipynb`), và chạy E5/E7 + nhãn mềm P2 (`notebooks/week3_transformer_and_soft_labels.ipynb`) | TV3 |
| `scripts/` | EDA, tiện ích dữ liệu; `review_pairs.py` chia kiểm tay 100 cặp gần trùng cho nhiều người rồi gộp kết quả (TV2 thêm, TV1 duyệt); `dedup.py` cặp/cụm gần trùng, `export_headlines.py` xuất tiêu đề 2026 từ pipeline (TV2) | TV1 |
| `tests/` | Test nhỏ cho từng thay đổi hành vi (`python -m unittest`) | Cả nhóm |
| `results/` | Bảng kết quả, log, hình (không checkpoint); `results/smoke/` (`env.json`, `distilbert.json`, `qwen.json`, `gpu_budget.md`, `contribution_report.md`); `results/E1`–`E4/seed{1,2,3}/` (`run.json`, `preds_validation.jsonl`, `vocab.json`), `results/rq2_validation.md` (bảng RQ2 tạm + so ghép cặp E3/E4 với E2); `results/E5/seed{1,2,3}/` và `results/E7/seed{1,2,3}/` (`run.json`, `preds_validation.jsonl`); `results/llm_soft_labels/` (`p2_soft_labels.jsonl`, `run.json`) | Cả nhóm |
| `docs/` | Proposal, kế hoạch chi tiết (PDF); `WEEK1.md`–`WEEK4.md` việc chi tiết + checkpoint từng tuần; `guides/` hướng dẫn từng bước cho từng thành viên; `meetings/` chương trình + bộ câu hỏi họp "hiểu bài"; `WEEK1.md`, `WEEK2.md` việc chi tiết + checkpoint từng tuần | Cả nhóm |
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
- `labeling/llm_annotations/`: LLM-made labels kept for the record, **not gold** (see its `README.md`): `trial/tv1.csv`, `tv2.csv`, `tv3.csv` (trial 30), `trial_agreement.json` and `trial_adjudication.csv` (computed from those files), `round1/tv3.csv` (68 test headlines). Not to be opened while hand-labeling or compared with gold before week 6.
- `labeling/gold/`: human gold labels (`id,label,reason`), hand-labeled without AI.
- `tests/test_gold_labels.py`: checks every `labeling/gold/<annotator>_round<k>.csv` against `assignment.csv`.

## Week 2 TV3 files

- `evaluate/metrics.py` and `tests/test_metrics.py`: shared evaluation module (`read_preds`, `score`, `validation_gold`) and hand-computed unit tests.
- `notebooks/colab_setup.ipynb`: updated to read ID-only splits via `train.data.load_split_data` and prepare FNS raw data via `data/download.py --only fns`.
- `configs/llm_prompts.yaml` and `notebooks/llm_prompts.ipynb`: P1/P2/P3 prompt templates and Colab T4 validation evaluation + top-10 logits diagnostic notebook.
- `results/E11/{P1,P2,P3}/preds_validation.jsonl`: validation predictions (1,791 rows each) for `Qwen/Qwen2.5-7B-Instruct` (`nf4-4bit`).
- `results/llm_prompts/top10_check.json` and `results/llm_prompts/contribution_report_week2.md`: validation metrics, ABC token probability mass diagnostics (selecting `P2` with `macro_f1 = 0.7934`), and TV3 Week 2 contribution report.

## Week 3 TV3 files

- `configs/E5.yaml` and `configs/E7.yaml`: DistilBERT fine-tune (`freeze_encoder: false`) and frozen-encoder (`freeze_encoder: true`) experiment configs on `cluster_split.jsonl` (`8c2c35ea...`).
- `models/transformer.py` and `tests/test_transformer.py`: Transformer training/evaluation runner (weighted CE, AdamW `lr=2e-5`, linear warmup 10%, weight decay 0.01, FP16 AMP, early stopping on `val_macro_f1`) and unit tests.
- `results/E5/seed{1,2,3}/{run.json,preds_validation.jsonl}` and `results/E7/seed{1,2,3}/{run.json,preds_validation.jsonl}`: Three-seed Tesla T4 validation outputs (1,791 records per file) for full fine-tuning (`E5`: `0.8030 ± 0.0054` Macro-F1) and frozen-encoder training (`E7`: `0.5913 ± 0.0023` Macro-F1).
- `models/llm_soft_labels.py` and `tests/test_llm_soft_labels.py`: resumable P2 soft-label generator for the 2026 `train_unlabeled` pool (`1,409` eligible GDELT headlines from `data/raw/headlines_2026/headlines.jsonl`, `SHA-256: 28bd2f3d...`) with strict SHA-256 verification, `trial30_ids.csv` (`30` IDs), `test300_ids.csv`, and `assignment.csv` (`300` IDs) exclusion guards (`330` excluded IDs total), and text-free output validation.
- `results/llm_soft_labels/{p2_soft_labels.jsonl,run.json}`: Completed Colab T4 `Qwen/Qwen2.5-7B-Instruct` (`nf4-4bit`, prompt `P2`) soft-label predictions (`1,409` text-free records with `probs`, `letter_logp`, `abc_mass`, `top1_is_letter`; `min_abc_mass = 0.989`, `mean_abc_mass = 0.9998`, `share_top1_is_letter = 1.0`).
- `notebooks/week3_transformer_and_soft_labels.ipynb`: Colab T4 execution notebook for Tasks 3.9 (`E5`), 3.10 (`E7`), and 3.11 (`P2` soft labels with SHA-256 check, resumable execution, integrity verification, and zip export).
- `results/rq3_week3_contribution_report.md`: TV3 Week 3 implementation, E5/E7 3-seed validation results, completed Task 3.11 P2 soft-label validation, completed Task 3.12 Round 1 human labeling audit, and AI transparency report.

## Week 3 TV1 files

- `evaluate/stats.py`, `tests/test_stats.py`: paired bootstrap and exact McNemar on aligned validation predictions.
- `scripts/simulate_power.py`, `tests/test_simulate_power.py`, `results/power/`: synthetic detection simulation, assumptions, table and provenance.
- `results/E0/seed1/`: E0 validation predictions and run provenance.
- `scripts/pick300.py`, `tests/test_pick300.py`: seeded selection and balanced two-round human assignment of 300 RSS IDs (input remains local).
- `scripts/audit_leakage_capacity.py`, `tests/test_audit_leakage_capacity.py`, `results/feasibility/E10_frozen_split.md`: ID-only feasibility audit of frozen-split clusters.
- `labeling/test300_ids.csv`, `labeling/assignment.csv`, `labeling/test300_selection.json`: selected RSS IDs and URLs, two-round assignments, and selection provenance (no headline text).
- `data/raw/headlines_2026/labeling_packets/`: local-only reading copies and blank label templates for each annotator and round; headline text remains local and must not be staged.
- `scripts/make_leakage.py`, `tests/test_make_leakage.py`, `tests/test_make_leakage_cli.py`: validation-only ID overlay manifest preparation with explicit donors, hash checks, and synthetic tests; not yet integrated with training.
