@AGENTS.md

## Dành riêng cho Claude Code

- Đầu mỗi phiên: đọc MAP.md; với việc lớn đọc thêm SPEC.md và PLAN.md.
- Gặp quyết định lâu dài (thay đổi thiết kế, ngưỡng, quy tắc): ghi vào
  `context/decisions.md` (ngày · quyết định · lý do) và cập nhật SPEC/PLAN nếu liên quan.
- Xong một việc trong PLAN.md: đánh dấu `[x]` kèm commit hoặc kết quả.
- Môi trường: Windows 11, Git Bash và PowerShell, Python 3.11. Huấn luyện GPU chạy trên
  Colab/Kaggle (T4), không chạy trên máy này; máy này dùng cho code, test nhỏ, CPU benchmark.
- `gh` đã đăng nhập (`C:\Program Files\GitHub CLI\gh.exe`); repo private
  `thienandinh29/financial-sentiment-dl`.
- Tài liệu nộp (báo cáo, slide) viết tiếng Việt; tên biến, code, commit viết tiếng Anh.
