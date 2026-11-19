# AGENTS.md — quy tắc chung cho mọi trợ lý AI trong dự án

Đồ án cuối kỳ môn Học sâu (nhóm 3 người, hạn 20/12/2026): phân loại cảm xúc tiêu đề
tin tài chính (bearish / neutral / bullish) bằng PyTorch. Mục tiêu và thí nghiệm:
[SPEC.md](SPEC.md). Việc từng tuần: [PLAN.md](PLAN.md). Bản đồ thư mục: [MAP.md](MAP.md).
Quyết định đã chốt: [context/decisions.md](context/decisions.md) — không tự ý đổi lại.

## Quy tắc trung thực thực nghiệm (bắt buộc)

1. **Test chỉ chạy một lần.** Ba tập test (trong miền, 2026, PhraseBank) chỉ được đánh
   giá ở tuần 6, sau khi mọi lựa chọn đã chốt. Không dùng test để chọn epoch, siêu tham số,
   ngưỡng hay biến thể. Không viết code tự động chạy test.
2. **Split cố định.** Đọc split từ `data/splits/` và kiểm tra SHA-256 trước khi chạy.
   Không tạo split mới hay sửa split đã chốt.
3. **Fit chỉ trên train/validation:** từ điển, TF-IDF, chuẩn hóa, ngưỡng.
4. **Một thay đổi mỗi thí nghiệm.** Config mới copy từ config mốc và chỉ đổi một yếu tố;
   ghi vào ma trận trong SPEC.md.
5. **Không tạo nhãn thay người.** Nhãn tập test 2026 do thành viên gán tay. AI và LLM
   không bao giờ gán nhãn cho tập test; nhãn LLM chỉ dùng để huấn luyện.
6. **Báo cáo trung thực:** giữ mọi kết quả, kể cả thất bại hoặc bác bỏ giả thuyết.

## Dữ liệu và bí mật

- Không commit dữ liệu thô, văn bản tiêu đề 2026, checkpoint, file `.onnx`.
- Secrets (HF_TOKEN…) chỉ nằm trong `.env`; không ghi vào file khác, không in ra log.

## Cách làm việc

- **[ĐỀ XUẤT — chờ cả nhóm duyệt tại CP2 28/10]** Mỗi người một nhánh mỗi tuần
  (`<ten>/week<N>`, ví dụ `an/week2`), tạo từ `main` đầu tuần, gộp qua **một** pull request có
  ít nhất một người khác xem, trước checkpoint ◆ của tuần. Tuần sau tạo nhánh mới từ `main`.
  Không push thẳng lên `main`.
- Chỉ tách nhánh riêng (`<ten>/<viec>`) khi người khác cần một phần trước checkpoint; ghi lý do
  trong PR. Mỗi commit một việc nhỏ, để người duyệt đọc PR theo từng commit.
- (Quy tắc cũ, bỏ nếu đề xuất được duyệt: mỗi việc một nhánh `<ten>/<viec>`.)
- Mỗi người sở hữu một vùng thư mục (xem MAP.md); sửa vùng của người khác thì báo trước.
- Code: Python 3.11, PyTorch; hàm ngắn, có docstring nêu đầu vào/đầu ra; seed cố định.
- Mỗi lần chạy ghi commit, config, seed, phiên bản thư viện, GPU, thời gian vào `results/`.
- Tạo hoặc di chuyển file → cập nhật MAP.md trong cùng commit.
- Thay đổi hành vi → kèm test nhỏ chứng minh nó đúng.
