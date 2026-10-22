# Phân tích cảm xúc tiêu đề tin tài chính có kiểm soát rò rỉ dữ liệu

Đồ án cuối kỳ môn Học sâu. Phân loại tiêu đề tin tài chính tiếng Anh thành
bearish / neutral / bullish bằng PyTorch, tập trung vào **đo đúng**: chia tập theo
cụm tin gần trùng, kiểm định ngoài phân phối trên tin 2026 do nhóm gán nhãn, và kết
luận bằng khoảng tin cậy của chênh lệch.

- Proposal (5 trang): [docs/Proposal.pdf](docs/Proposal.pdf)
- Kế hoạch chi tiết: [docs/KeHoachChiTiet.pdf](docs/KeHoachChiTiet.pdf)
- Hạn nộp: 20/12/2026
- Làm việc hằng ngày: [SPEC.md](SPEC.md) (thiết kế), [PLAN.md](PLAN.md) (việc từng tuần),
  [MAP.md](MAP.md) (tìm gì ở đâu), [context/decisions.md](context/decisions.md) (quyết định đã chốt),
  [AGENTS.md](AGENTS.md) (quy tắc cho trợ lý AI)

## Câu hỏi nghiên cứu

| RQ | Câu hỏi | Thí nghiệm |
| --- | --- | --- |
| RQ1 | Bản tin gần trùng giữa train và test làm Macro-F1 cao giả tạo bao nhiêu; BiLSTM hay DistilBERT nhạy hơn? | E9, E10 |
| RQ2 | Attention, GloVe và focal loss đóng góp bao nhiêu cho BiLSTM? | E1–E4, E13 |
| RQ3 | Mô hình học trên tweet giảm bao nhiêu trên tiêu đề báo 2026; chưng cất từ LLM có giúp không? | E5–E8, E11, E12 |

## Cấu trúc

```
configs/     # mỗi thí nghiệm E0–E13 một file YAML
data/        # script tải dữ liệu; data/splits/ chứa file split + SHA-256
labeling/    # hướng dẫn gán nhãn, tính kappa
models/      # TextCNN, BiLSTM-Attention, Transformer, chưng cất
train/       # vòng huấn luyện chung (seed, early stopping, logging)
evaluate/    # metric, paired bootstrap, McNemar, robustness
inference/   # xuất ONNX, đo latency, FastAPI, Dockerfile
app/         # demo Gradio
scripts/     # EDA, công cụ dữ liệu
results/     # bảng kết quả, log, hình (không chứa checkpoint)
docs/        # proposal, kế hoạch
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Phiên bản thư viện sẽ được ghim vào `requirements.lock` sau tuần 1. Huấn luyện chạy
trên GPU T4 của Colab/Kaggle; xem phần tái lập bên dưới khi đã có lệnh.

## Tái lập

Mỗi thí nghiệm chạy bằng một lệnh, ví dụ (sẽ có từ tuần 2–3):

```bash
python -m train.run --config configs/E2.yaml --seed 1
```

Mỗi lần chạy ghi lại commit Git, file cấu hình, seed, phiên bản thư viện, GPU và
thời gian. Ba tập test (trong miền, 2026, PhraseBank) chỉ chạy **một lần** ở tuần 6.

## Dữ liệu

| Dữ liệu | Vai trò | Giấy phép |
| --- | --- | --- |
| [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | Train / validation / test trong miền | MIT |
| Tiêu đề 2026 do nhóm gán nhãn | Test ngoài phân phối (RQ3) | Chỉ nhãn và URL, không kèm văn bản |
| [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (AllAgree) | Test ngoài miền | CC BY-NC-SA 3.0 |
| [GloVe 6B 100d](https://nlp.stanford.edu/projects/glove/) | Embedding pretrained | PDDL |

Văn bản tiêu đề 2026 thuộc các nhà xuất bản và **không được commit** vào repo.

## Nhóm

| Thành viên | Vai trò |
| --- | --- |
| [Họ tên 1] (nhóm trưởng) | Dữ liệu, chống rò rỉ, thống kê; RQ1 |
| [Họ tên 2] | RNN/CNN, vòng huấn luyện chung; RQ2 |
| [Họ tên 3] | Transformer, LLM, demo; RQ3 |

## Quy tắc làm việc

- Mỗi thay đổi qua một nhánh và pull request, có ít nhất một người khác xem.
- Không dùng tập test để chọn epoch, siêu tham số hay biến thể.
- Không commit dữ liệu thô, checkpoint hay file `.env`.

## Ghi nhận hỗ trợ AI

Nhóm sử dụng trợ lý AI (Claude) để xây dựng pipeline thu thập dữ liệu, soạn proposal,
kế hoạch và khung dự án. Nhãn do LLM tạo chỉ dùng để huấn luyện, không dùng để đánh
giá. Chi tiết sẽ ghi trong báo cáo cuối kỳ.
