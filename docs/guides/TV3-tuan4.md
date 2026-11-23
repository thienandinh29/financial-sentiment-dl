# Hướng dẫn tuần 4 cho TV3 (Transformer & LLM)

Giai đoạn S2: **Thứ Bảy 21/11 – Thứ Sáu 27/11**. Kế hoạch chung: [docs/WEEK4.md](../WEEK4.md).
Nhánh của bạn: `tv3/week4`, tạo từ `main` mới nhất, **một PR** trước tối Thứ Sáu 27/11.

```powershell
git switch main
git pull
git switch -c tv3/week4
```

Thứ tự làm: **3.15 → 3.13 (E12) → 3.14 (E6) → 3.16 → 3.17.** E12 phải xong trước E6 vì E6 được so với cả E5 và E12.

---

## 3.15 Sửa ghi chép gán nhãn (hạn CP6a, Thứ Tư 25/11) — 15 phút

Nhãn gán thử 30 tiêu đề và nhãn đợt 1 của bạn được tạo bằng LLM. Nhóm đã chuyển file sang
`labeling/llm_annotations/` (PR #19) và ghi vào `context/decisions.md`. Hai báo cáo của bạn vẫn ghi
"tự gán tay 100%, không AI".

1. `results/rq3_week3_contribution_report.md` (dòng Task 3.12 và mục 4.3) và
   `results/llm_prompts/contribution_report_week2.md` (dòng gán thử 30): **không xóa**, thêm ngay dưới mỗi chỗ:
   `> Sửa 23/11: nhãn này do LLM tạo, không phải gán tay; file đã chuyển sang labeling/llm_annotations/ (xem context/decisions.md).`
2. Nếu bạn đã làm gì với tiêu đề **đợt 2** của mình (dù chỉ thử với LLM), báo TV2 ngay để TV2 không giao chúng cho bạn ở 3.16.
3. Commit: `docs(tv3): correct labeling records (LLM-made)`.

Báo cáo trung thực không làm mất điểm; che giấu mới làm mất.

---

## Trước mọi lần chạy trên Colab: tránh "dirty"

Toàn bộ E5, E7 ghi `working_tree_dirty: true`. Tuần này phải sạch:

1. **Commit và push config trước**, rồi mới chạy.
2. Trên Colab: `git clone` (hoặc `git pull`) đúng nhánh `tv3/week4`; **không sửa file nào trong repo trên Colab**.
   Dữ liệu (`data/raw/...`) để ở đường dẫn bị git bỏ qua.
3. Ngay trước khi chạy, ô lệnh này phải **không in gì**:
   ```
   !git status --porcelain --untracked-files=no
   ```
4. Sau khi chạy, kiểm tra `run.json` có `working_tree_dirty: false` và đúng commit đã push.

---

## 3.13 E12 — E5 + nhãn cứng LLM, 3 seed (hạn CP6)

**Thay đổi duy nhất so với E5:** thêm các tiêu đề `train_unlabeled` 2026 vào train, nhãn = **argmax** của `probs`
trong `results/llm_soft_labels/p2_soft_labels.jsonl` (1.409 dòng; 0 = bearish, 1 = neutral, 2 = bullish).

1. Code: thêm vào `models/transformer.py` một tùy chọn config, ví dụ
   ```yaml
   data:
     pseudo_labels: results/llm_soft_labels/p2_soft_labels.jsonl
     pseudo_mode: hard        # E12
   ```
   Văn bản lấy từ `data/raw/headlines_2026/headlines.jsonl` (kiểm tra SHA-256 như bạn đã làm ở 3.11), ghép theo `id`.
   Khi không có `pseudo_labels`, mọi thứ chạy y như E5 (E5 không đổi kết quả).
2. **Giữ nguyên trọng số lớp của E5** (tính chỉ trên train FNS) — ghi rõ trong comment config. Validation không đổi.
3. Test nhỏ: hàm đọc nhãn giả trả đúng argmax; từ chối ID trùng; từ chối ID nằm trong 330 ID loại trừ
   (trial30 + 300 test); không ID nào thuộc validation/test FNS.
4. `configs/E12.yaml`: copy `E5.yaml`, chỉ thêm hai dòng trên. Commit + push, rồi chạy 3 seed trên T4.
5. Ghi vào `context/decisions.md`: Macro-F1 validation E12 (trung bình ± độ lệch chuẩn) so với E5 0,8030 ± 0,0054.

Lưu ý: nhãn LLM rất lệch về `neutral` (1.186/1.409). Nếu E12 thấp hơn E5, đó vẫn là kết quả thật — giữ lại (quy tắc 6).

---

## 3.14 E6 — chưng cất nhãn mềm (hạn CP6)

**Thay đổi so với E12:** dùng phân phối mềm thay cho nhãn cứng trên cùng 1.409 tiêu đề.

- Phân phối mục tiêu với nhiệt độ T: `softmax(letter_logp / T)` (cột `letter_logp` có sẵn). `probs` gốc gần như
  one-hot, nên T = 2, 4 mới thật sự "mềm".
- Loss trên dòng nhãn giả: `α · T² · KL(target_T ‖ softmax(logits / T))`; dòng FNS: weighted CE như E5.
  `α` là trọng số phần chưng cất.
- Config: `pseudo_mode: soft`, `distill_alpha`, `distill_temperature`.

**Chọn α, T trên validation (không bao giờ trên test):**

1. Lưới: T ∈ {1, 2, 4} × α ∈ {0,3; 0,5; 0,7} = 9 tổ hợp, **seed 1**.
2. Ghi cả 9 kết quả vào `results/E6/grid.md` (kể cả tổ hợp tệ).
3. Chọn tổ hợp có Macro-F1 validation cao nhất, ghi vào `context/decisions.md`, rồi chạy **3 seed** với tổ hợp đó
   → `results/E6/seed{1,2,3}/`.

Test nhỏ: với `letter_logp` cho trước và T = 2, phân phối mục tiêu đúng như tính tay; α = 0 thì loss phần chưng cất bằng 0.

---

## 3.16 Gán đôi trực tiếp (hạn CP6)

Gold 300 tiêu đề do TV2 gán tay. Để có kappa, 100 tiêu đề gán đôi cần người thứ hai gán **tay**.
Bạn sẽ gán **~50 tiêu đề mà bạn chưa từng đưa qua LLM** (TV2 đã lọc sẵn).

- Hình thức: một buổi gọi **bật camera**, ~45 phút, TV2 có mặt. TV2 gửi gói đọc (chỉ tiêu đề) lúc bắt đầu.
- Quy tắc: chỉ đọc tiêu đề, không mở bài báo, **không AI**, không mở `labeling/llm_annotations/`, không hỏi người khác.
  Lý do viết bằng lời của bạn, không chép tiêu đề. Đọc lại `labeling/GUIDE.md` trước buổi gán (10 phút).
- Hẹn giờ với TV2 trước CP6a.

---

## 3.17 Nháp chương 2 — Transformer và chưng cất (hạn CP6)

File `report/ch2_transformer.md`, tiếng Việt, 2–3 trang:

1. Transformer và self-attention (ngắn, có hình hoặc công thức), DistilBERT (chưng cất từ BERT, 6 tầng).
2. Fine-tune toàn bộ (E5) và đóng băng encoder (E7): khác nhau ở đâu, số tham số huấn luyện.
3. Nhãn giả từ LLM: prompt P2, lấy logit A/B/C; nhãn cứng (E12) và nhãn mềm + nhiệt độ (E6), công thức loss.
4. Kết quả tạm trên validation (ghi rõ "validation, không phải test").

Mỗi con số ghi nguồn (đường dẫn file trong `results/`).

---

## Mang đến CP6a (Thứ Tư 25/11)

- [ ] 3.15 đã commit
- [ ] Giờ buổi gán đôi đã hẹn với TV2
- [ ] `configs/E12.yaml` (và code nhãn giả) đã commit + push

## Mang đến CP6 ◆ (Thứ Sáu 27/11)

- [ ] 3.13: E12, 3 seed, `working_tree_dirty: false`
- [ ] 3.14: lưới 9 tổ hợp + E6 3 seed
- [ ] 3.16: ~50 nhãn gán tay
- [ ] 3.17: nháp chương 2 (Transformer)
- [ ] PR `tv3/week4` → `main`, toàn bộ test chạy qua (`python -m unittest discover -s tests`)
