# Hướng dẫn tuần 4 cho TV1 (dữ liệu & thống kê)

Giai đoạn S2: **Thứ Bảy 21/11 – Thứ Sáu 27/11**. Kế hoạch chung: [docs/WEEK4.md](../WEEK4.md).
Nhánh của bạn: `tv1/week4`, tạo từ `main` mới nhất, **một PR** trước tối Thứ Sáu 27/11.

```powershell
git switch main
git pull
git switch -c tv1/week4
```

Thứ tự làm: **1.18 → 1.16 → 1.17 → 1.19 → 1.20.** Mỗi việc xong thì commit một lần, thông điệp tiếng Anh.

---

## 1.18 Sửa ghi chép gán nhãn (hạn CP6a, Thứ Tư 25/11) — 15 phút

Nhãn gán thử 30 tiêu đề và nhãn đợt 1–2 của bạn được tạo bằng LLM. Nhóm đã chuyển file sang
`labeling/llm_annotations/` (PR #19) và ghi vào `context/decisions.md`. Ghi chép của bạn vẫn nói là gán tay.

1. Mở `plan/plan_tv1_week2.md`. Ở dòng `- [x] Tự gán nhãn độc lập 30 tiêu đề, không dùng AI…` và dòng nhật ký
   ngày 16/11 ("Three human annotation files…"), **không xóa**, thêm ngay dưới:
   `> Sửa 23/11: nhãn gán thử do LLM tạo, không phải gán tay; file đã chuyển sang labeling/llm_annotations/ (xem context/decisions.md).`
2. **Không commit** nhãn đợt 1–2 bạn đang giữ (69 + 65). Xóa chúng khỏi máy hoặc để ngoài repo.
3. Commit: `docs(tv1): correct trial labeling record (LLM-made)`.

Báo cáo trung thực không làm mất điểm; che giấu mới làm mất.

---

## 1.16 Dữ liệu rò rỉ E10 theo Phương án A (hạn CP6) — việc chính tuần này

**Quyết định (23/11, trưởng nhóm):** E10 rò rỉ vào **validation**, không đụng test. Lý do: bạn đã đo được trần
rò rỉ bằng bản gần trùng trên split đã khóa chỉ 4,13% (test) / 4,24% (validation), nên 5/10/20% bằng bản gần trùng
là không làm được (`results/feasibility/E10_frozen_split.md` — đo rất tốt).

### Thiết kế cần cài

Với mỗi mức p ∈ {5%, 10%, 20%}:

- **Tập đánh giá giữ nguyên:** toàn bộ 1.791 ID validation (giống E2 sạch, để so ghép cặp được).
- **Mục tiêu:** số ID validation "bị lộ" = `ceil(p × 1.791)` → **90 / 180 / 359**.
- Một ID validation "bị lộ" khi **chính nó** hoặc **một bản gần trùng trực tiếp của nó** (cặp trong file cặp 0,92)
  được thêm vào train.
- **Chọn donor (ID validation thêm vào train) theo hai bước, seed cố định:**
  1. Ưu tiên donor trong các cụm có ≥ 2 ID validation (mỗi donor lộ chính nó + hàng xóm trực tiếp).
  2. Khi hết cụm, thêm ID đơn lẻ (bản sao chính xác, lộ chính nó) cho đến khi **đúng** số mục tiêu.
  Nếu một donor làm vượt mục tiêu, bỏ qua donor đó và thử donor kế tiếp.
- **Lồng nhau:** donor của mức 5% nằm trong 10%, 10% nằm trong 20% (dùng cùng thứ tự ngẫu nhiên, lấy tiền tố).
  Như vậy chênh lệch giữa các mức chỉ do lượng rò rỉ.
- Train gốc **không bỏ** dòng nào; chỉ thêm donor (dùng nhãn validation của chúng — được phép vì đây là thí nghiệm
  trên validation; test vẫn không ai chạm).

### Việc cụ thể

1. Mở rộng `scripts/make_leakage.py` (hoặc thêm hàm mới bên cạnh `make_manifest`; hàm cũ giữ nguyên):
   - đầu vào: split + SHA-256, file cặp + SHA-256, mức p, seed;
   - đầu ra mỗi mức: `data/splits/leak_05.json`, `leak_10.json`, `leak_20.json`, **chỉ ID**:
     `train_add_validation_ids`, `validation_exposed_ids` (chia hai nhóm `near_duplicate` và `exact`),
     `n_exposed`, `fraction`, `seed`, SHA-256 của split và file cặp;
   - in SHA-256 của từng file đầu ra.
2. Viết test trong `tests/` với dữ liệu giả nhỏ (vài chục ID, vài cặp), kiểm tra:
   - đúng số ID bị lộ cho từng mức;
   - donor đều là ID validation; không có ID test nào xuất hiện;
   - 5% ⊂ 10% ⊂ 20%;
   - cùng seed → cùng kết quả; khác seed → khác thứ tự.
3. Chạy thật, ghi SHA-256 ba file vào `results/feasibility/E10_manifests.md` cùng số ID lộ theo từng loại.
4. Cập nhật `MAP.md` trong cùng commit (file mới).

Bạn **không** cần sửa `train/` — TV2 sẽ cho vòng huấn luyện đọc manifest (`data.leak_manifest`) ở S3.
Không đọc văn bản, không đọc nhãn test, không chạy mô hình nào trên test.

---

## 1.17 Chạy lại E0 từ commit sạch (hạn CP6) — 10 phút

`results/E0/seed1/run.json` ghi `working_tree_dirty: true`. Chạy lại khi cây làm việc sạch:

```powershell
git status --porcelain --untracked-files=no   # phải không in gì
python -m models.tfidf_lr --config configs/E0.yaml
```

(dùng đúng lệnh bạn đã chạy E0 lần trước). Kiểm tra `run.json` mới có `working_tree_dirty: false`, rồi commit
kết quả. Nếu Macro-F1 khác 0,7096, ghi số mới vào `results/rq2_validation.md` dòng E0.

---

## 1.19 Gán đôi trực tiếp (hạn CP6)

Gold 300 tiêu đề do TV2 gán tay. Để có kappa, 100 tiêu đề gán đôi cần người thứ hai gán **tay**.
Bạn sẽ gán **~33 tiêu đề mà bạn chưa từng đưa qua LLM** (TV2 đã lọc sẵn, để nhãn cũ không ảnh hưởng bạn).

- Hình thức: một buổi gọi **bật camera**, ~30 phút, TV2 có mặt. TV2 gửi gói đọc (chỉ tiêu đề) lúc bắt đầu.
- Quy tắc: chỉ đọc tiêu đề, không mở bài báo, **không AI**, không mở `labeling/llm_annotations/`, không hỏi người khác.
  Lý do viết bằng lời của bạn, không chép tiêu đề. Đọc lại `labeling/GUIDE.md` trước buổi gán (10 phút).
- Hẹn giờ với TV2 trước CP6a.

---

## 1.20 Nháp chương 3 — Dữ liệu (hạn CP6)

File `report/ch3_data.md`, tiếng Việt, 2–4 trang. Dàn ý:

1. Nguồn dữ liệu: Twitter FNS (giấy phép, số mẫu, phân bố lớp), PhraseBank, tiêu đề 2026 (RSS cho test, GDELT cho
   kho không nhãn).
2. Khử trùng lặp: MinHash LSH + Jaccard ≥ 0,75; MiniLM cosine ≥ 0,92 (vì sao đổi từ 0,85 — kiểm tra 100 cặp).
3. Split theo cụm 70/15/15, SHA-256 — vì sao split ngẫu nhiên làm phồng điểm (dẫn tới RQ1, E9, E10).
4. Thiết kế E10 (Phương án A ở trên) và giới hạn 4,1% của bản gần trùng.
5. Quy trình gán nhãn test 2026: chọn 300 ID, GUIDE, gán đôi, kappa. **Ghi trung thực sự cố nhãn LLM ngày 23/11**
   và cách nhóm xử lý (một đoạn ngắn, giọng khách quan).

Số liệu lấy từ `results/` và `context/decisions.md`; mỗi con số ghi nguồn (đường dẫn file).

---

## Mang đến CP6a (Thứ Tư 25/11)

- [ ] 1.18 đã commit
- [ ] Giờ buổi gán đôi đã hẹn với TV2
- [ ] 1.16: đã bắt đầu, có test đầu tiên

## Mang đến CP6 ◆ (Thứ Sáu 27/11)

- [ ] 1.16: ba manifest + SHA-256 + test chạy qua
- [ ] 1.17: E0 sạch
- [ ] 1.19: ~33 nhãn gán tay
- [ ] 1.20: nháp chương 3
- [ ] PR `tv1/week4` → `main`, toàn bộ test chạy qua (`python -m unittest discover -s tests`)
