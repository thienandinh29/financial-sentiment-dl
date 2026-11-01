# Hướng dẫn gán nhãn tập test 2026 — BẢN NHÁP

> Đã bổ sung quy tắc và ví dụ ngày 01/11/2026; **chờ cả 3 thành viên duyệt tại CP2**.
> Chưa có bằng chứng đồng thuận nên vẫn giữ trạng thái BẢN NHÁP.
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
5. Nhãn mô tả nội dung được nêu, không phải khuyến nghị mua/bán hay dự đoán của người gán.
   Không suy diễn ảnh hưởng chuỗi cung ứng hoặc diễn biến giá mà tiêu đề không nói tới.
6. Với nhiều chủ thể, ưu tiên chủ thể được tiêu đề tập trung mô tả. Nếu không xác định được
   một chủ thể chính và hai hướng ngang nhau, chọn neutral, ghi rõ lý do.
7. Kết quả so với **kỳ vọng** quan trọng hơn dấu tăng/giảm tuyệt đối khi tiêu đề nêu rõ
   vượt/hụt kỳ vọng. Với kết quả tốt nhưng hạ triển vọng, theo thông điệp chủ đạo được nêu;
   nếu không có hướng trội, neutral và đánh dấu cần thảo luận.
8. Tin sáp nhập, bổ nhiệm, lịch công bố hay một câu hỏi không tự động bullish/bearish.
   Chỉ gán hướng khi tiêu đề nêu tác động rõ; không mở bài để tìm thêm bằng chứng.
9. Không dùng neutral thay cho "không hiểu". Lỗi nguồn, tiêu đề thiếu hoặc không rõ chủ thể
   cần ghi vấn đề riêng để nhóm phân xử; không tự ý loại mẫu hay tạo lớp thứ tư.

## Quy trình gán độc lập và phân xử

- Mỗi lượt lưu ID, nhãn chữ thường (`bearish`, `neutral`, `bullish`), lý do ngắn,
  mã người gán, ngày và phiên bản GUIDE; không chép văn bản tiêu đề vào file nhãn được commit.
- Gán thử 30 tiêu đề ở tuần 2; từng người khóa lượt riêng trước khi so sánh.
- Trong 300 tiêu đề chính thức, 100 tiêu đề được gán đôi. Giữ nguyên cả hai nhãn ban đầu
  để tính kappa **trước phân xử**; người thứ ba phân xử bất đồng và lưu lý do riêng.
- Mục tiêu kappa ≥ 0,6; không sửa nhãn ban đầu để nâng kappa. Nếu thấp, nhóm rà nguyên nhân
  và ghi quyết định, không xin nhãn AI/LLM.
- Mọi ví dụ dưới đây chỉ lấy từ FNS hoặc là tình huống diễn giải. Không dùng tiêu đề 2026
  làm ví dụ do AI gán nhãn, kể cả trong vòng gán thử.

## Ví dụ mẫu

Các ví dụ FNS được **diễn giải**, không chép dữ liệu thô. ID trỏ tới bản FNS cố định trong
`data/sources.json`; nhãn là nhãn có sẵn của nguồn, không phải nhãn AI tạo cho test 2026.
Người gán có thể tra đúng dòng trong `data/raw/fns/train.jsonl` ở máy mình.

| Nhãn nguồn | ID FNS | Nội dung diễn giải và điểm cần đọc |
| --- | --- | --- |
| bearish | `fns-train-00000` | JPMorgan giảm kỳ vọng đối với Beyond Meat: triển vọng kém đi. |
| bearish | `fns-train-00002` | Cemex bị hạ đánh giá vì triển vọng xây dựng yếu. |
| bearish | `fns-train-00894` | VF đóng phần lớn cửa hàng Trung Quốc do dịch: gián đoạn hoạt động được nêu rõ. |
| bullish | `fns-train-00126` | ALTG được khởi đầu đánh giá ở mức mua. |
| bullish | `fns-train-00128` | ANCUF được nâng lên mức outperform. |
| bullish | `fns-train-01064` | Trafigura công bố năm kỷ lục trong dầu và kim loại. |
| neutral | `fns-train-00527` | Thống đốc ngân hàng trung ương Ai Cập được bổ nhiệm nhiệm kỳ hai, không nêu tác động tài chính. |
| neutral | `fns-train-00621` | Thông báo phát biểu trực tiếp của Powell tại cuộc họp FOMC, chưa nêu thông điệp định hướng. |
| neutral | `fns-train-00652` | Thông cáo về cuộc họp hội đồng tiền tệ, không nêu kết quả thuận/bất lợi. |

## Trường hợp khó cần thống nhất khi duyệt

| Trường hợp | Cách xử lý đề nghị |
| --- | --- |
| Dầu tăng, không nêu doanh nghiệp hưởng lợi/chịu thiệt | Nếu chủ thể là giá dầu thì hướng tăng là bullish cho tài sản đó; không suy ra hãng dầu hay hàng không. |
| Tiêu đề nêu rõ chi phí nhiên liệu làm giảm lợi nhuận hãng bay | Bearish cho hãng bay, dù giá dầu tăng. |
| Vượt dự báo quý này nhưng hạ triển vọng năm | Xem thông điệp nào chi phối; nếu cân bằng và không có hướng trội thì neutral; ghi lý do để phân xử. |
| Lỗ nhưng ít hơn dự báo | Có thể bullish khi tiêu đề nhấn mạnh vượt kỳ vọng; không chỉ thấy chữ "lỗ" rồi gán bearish. |
| Công bố mua lại doanh nghiệp | Không tự suy ra bên mua/bên bán đều bullish; cần tác động hoặc phản ứng giá được nêu. |
| Tăng từ mức đáy nhưng vẫn giảm theo năm | Ưu tiên diễn biến được tiêu đề nhấn mạnh; ghi mốc so sánh, tránh tự chọn mốc có lợi. |
| Câu hỏi, tin đồn, mỉa mai hoặc điều kiện | Không biến giả định thành sự thật; thiếu hướng rõ thì neutral, nêu điểm mơ hồ. |

Ví dụ khó từ nguồn: `fns-train-00906` vừa nêu giảm sản lượng Petrobras vừa có phát hiện dầu,
nguồn gán bearish. Đây là ví dụ cần đọc chủ thể và thông điệp, không phải quy tắc rằng mọi
tiêu đề có hai vế đều neutral. Nhãn nguồn cũng có thể nhiễu, nhóm không sửa dữ liệu nguồn.

## Lịch sử sửa đổi

| Ngày | Thay đổi | Lý do |
| --- | --- | --- |
| 2026-11-01 | Bổ sung 9 ví dụ FNS, ca khó, quy trình gán độc lập và phân xử; vẫn chờ 3 thành viên duyệt | Hoàn thiện phần chuẩn bị của TV1 mà không giả lập đồng thuận CP2 hoặc gán nhãn test |
