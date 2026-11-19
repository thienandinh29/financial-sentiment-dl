# Hướng dẫn gán nhãn tập test 2026 — BẢN NHÁP

> Bản nháp lấy từ kế hoạch. Nhóm rà soát và **chốt trong tuần 1**, trước vòng gán thử 30 tiêu đề.
> Sau khi chốt, mọi sửa đổi phải ghi ngày và lý do ở cuối file.

## Nhãn

| Nhãn | Khi nào |
| --- | --- |
| **Bullish** | Tiêu đề nói rõ một diễn biến hoặc triển vọng tài chính thuận lợi cho chủ thể chính (giá tăng, lợi nhuận vượt dự báo, được nâng hạng). |
| **Bearish** | Diễn biến hoặc triển vọng bất lợi rõ ràng cho chủ thể chính (giá giảm, thua lỗ, cảnh báo rủi ro, bị phạt). |
| **Neutral** | Thông tin tài chính không có hướng rõ, hoặc hai hướng cân bằng mà không có thông điệp chủ đạo. |

Định nghĩa khớp với nhãn của Twitter Financial News Sentiment để so sánh được.

## Quy tắc

1. Chỉ đọc tiêu đề. Không mở bài báo, không dùng diễn biến sau ngày đăng.
2. Xác định chủ thể chính của tiêu đề trước, rồi mới quyết định hướng.
3. Mỗi tiêu đề ghi một lý do ngắn.
4. Không xem dự đoán của mô hình, nhãn LLM hay nhãn của người khác trước khi gán.

### Làm rõ sau vòng gán thử 30 tiêu đề

Nhóm thống nhất chỉ xét sentiment tài chính có thể suy ra từ tiêu đề, không dùng nội dung bài báo. Khi tiêu đề nhắc nhiều chủ thể hoặc nhiều hướng, cần xác định thông điệp chính trước khi gán. Không tự suy thêm kết quả tài chính từ một sản phẩm hoặc ý định mở rộng khi tiêu đề chưa thể hiện rõ.

Trong năm trường hợp bất đồng, nhóm chốt: cơ hội bán hàng của Nvidia là `bullish`; tin về sản phẩm VW Bus là `neutral`; nhận định so sánh cuộc đua AI Trung Quốc–Mỹ là `neutral`; khuyến nghị mua cổ phiếu của Goldman Sachs là `bullish`; ý định mở rộng của ngành cá cược thể thao vào quỹ là `neutral`. Các kết luận này và ID tương ứng được lưu trong `labeling/trial_adjudication.csv`; không chép văn bản tiêu đề vào repo.

## Ví dụ mẫu

*Nhóm điền 2–3 ví dụ thật cho mỗi nhãn và cho các trường hợp khó (ví dụ: giá dầu tăng là tốt hay xấu, tùy chủ thể) sau vòng gán thử.*

## Lịch sử sửa đổi

| Ngày | Thay đổi | Lý do |
| --- | --- | --- |
| 2026-11-16 | Bổ sung cách xét chủ thể, thông điệp chính và năm kết luận sau gán thử. | Nhóm thống nhất sau khi thảo luận các mẫu bất đồng; chỉ dựa trên tiêu đề. |
