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
| `.worktreeinclude` | File bị gitignore được sao chép vào mỗi worktree song song (`.env`) |
| `requirements.txt` | Thư viện trực tiếp; `requirements.lock` sau tuần 1 |

## Thư mục

| Thư mục | Nội dung | Phụ trách |
| --- | --- | --- |
| `context/` | Quyết định đã chốt và lý do (`decisions.md`) | Cả nhóm |
| `configs/` | Mỗi thí nghiệm một YAML; `E2.yaml` là mẫu | Theo RQ |
| `data/` | Script tải dữ liệu, `splits/` (split + SHA-256); quy tắc trong `data/README.md` | TV1 |
| `labeling/` | Hướng dẫn gán nhãn, nhãn test 2026 (không kèm văn bản), kappa | TV1 |
| `models/` | TextCNN, BiLSTM-Attention, Transformer, chưng cất | TV2, TV3 |
| `train/` | Vòng huấn luyện chung | TV2 |
| `evaluate/` | Metric, bootstrap, McNemar, robustness | TV1 |
| `inference/` | ONNX, latency, FastAPI, Dockerfile | TV3, TV2 |
| `app/` | Demo Gradio | TV3 |
| `scripts/` | EDA, tiện ích dữ liệu | TV1 |
| `results/` | Bảng kết quả, log, hình (không checkpoint) | Cả nhóm |
| `docs/` | Proposal, kế hoạch chi tiết (PDF) | Cả nhóm |
| `.claude/` | Cấu hình Claude Code dùng chung (`settings.json`) | Cả nhóm |
