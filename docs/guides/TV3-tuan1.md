# Hướng dẫn tuần 1 cho TV3 (Transformer & demo)

Hạn: **CP2 tối Thứ Tư 28/10**. Việc gốc và tiêu chí "xong": [docs/WEEK1.md](../WEEK1.md).
Mục tiêu tuần này **không phải** huấn luyện thật, mà là chứng minh môi trường GPU chạy được và
đo thời gian để ước tính ngân sách GPU cho E5–E12.

**Ba quy tắc quan trọng nhất cho TV3:**
1. `HF_TOKEN` (và token GitHub) chỉ nằm trong **Colab Secrets**, không bao giờ in ra hay lưu vào notebook.
2. Chạy thử chỉ trên **Twitter FNS**. Không chạy LLM trên tiêu đề 2026 tuần này, và không bao
   giờ chạy trên tập test 2026.
3. Commit notebook **đã xóa output** (Edit → Clear all outputs) nếu output có văn bản tiêu đề.

---

## 3.1 Notebook cài đặt — `notebooks/colab_setup.ipynb`

Nhánh: `<ten>/colab-smoke`.

1. Colab → Runtime → Change runtime type → **T4 GPU**.
2. Biểu tượng chìa khóa (Secrets) bên trái → thêm `HF_TOKEN` và `GH_TOKEN`
   (GitHub → Settings → Developer settings → Fine-grained token, chỉ quyền *Contents: Read* cho repo
   `thienandinh29/financial-sentiment-dl`). Bật "Notebook access" cho cả hai.
3. Ô 1 — clone repo private và cài thư viện:

```python
import os, subprocess
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")          # never print it
gh = userdata.get("GH_TOKEN")
subprocess.run(["git", "clone", "-q",
                f"https://x-access-token:{gh}@github.com/thienandinh29/financial-sentiment-dl.git"],
               check=True, capture_output=True)              # capture_output keeps the token out of the log
%cd financial-sentiment-dl
!python --version
!pip install -q -r requirements.lock
```

   - `requirements.lock` được tạo trên **Python 3.11**. Colab có thể đang dùng 3.12: nếu `pip install`
     báo lỗi, ghi lại lỗi đó (TV2 cần biết), rồi tạm cài `pip install -q -r requirements.txt`.
4. Ô 2 — ghi môi trường vào `results/smoke/env.json`:

```python
import json, platform, torch, transformers
env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0), "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
       "transformers": transformers.__version__}
os.makedirs("results/smoke", exist_ok=True)
json.dump(env, open("results/smoke/env.json", "w"), indent=2); print(env)
```

Xong khi: chạy từ đầu đến cuối trên runtime mới không lỗi, `gpu` là "Tesla T4", token không hiện ở đâu.

## 3.2 Chạy thử DistilBERT — 1.000 mẫu, 1 epoch

Mục đích: đo **giây/epoch** và **bộ nhớ GPU đỉnh** để ước tính E5–E10. Không phải kết quả thí nghiệm.

```python
import json, time
import numpy as np, torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED, N_TRAIN, N_VAL, MAX_LEN, BS = 1, 1000, 500, 64, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
ds = load_dataset("zeroshot/twitter-financial-news-sentiment")    # 0 bearish, 1 bullish, 2 neutral
train = ds["train"].shuffle(seed=SEED).select(range(N_TRAIN))
val = ds["validation"].shuffle(seed=SEED).select(range(N_VAL))     # validation only, never test
# FNS rows are stored roughly sorted by label: always shuffle before taking a subset.
name = "distilbert/distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3).to(DEVICE)


def batches(split, bs, shuffle):
    order = np.random.default_rng(SEED).permutation(len(split)) if shuffle else np.arange(len(split))
    for i in range(0, len(order), bs):
        b = split.select(order[i:i + bs].tolist())
        enc = tok(list(b["text"]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        yield {k: v.to(DEVICE) for k, v in enc.items()}, torch.tensor(list(b["label"]), device=DEVICE)


opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
use_amp = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
if use_amp:
    torch.cuda.reset_peak_memory_stats()
model.train()
t0 = time.time()
for x, y in batches(train, BS, shuffle=True):
    with torch.autocast(DEVICE, dtype=torch.float16, enabled=use_amp):
        loss = model(**x, labels=y).loss
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
if use_amp:
    torch.cuda.synchronize()
epoch_s = time.time() - t0

model.eval()
preds = []
with torch.no_grad():
    for x, _ in batches(val, 64, shuffle=False):
        preds += model(**x).logits.argmax(-1).tolist()
log = {"model": name, "n_train": N_TRAIN, "batch_size": BS, "max_len": MAX_LEN, "seed": SEED,
       "epoch_seconds": round(epoch_s, 1),
       "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if use_amp else None,
       "val_macro_f1_smoke": round(f1_score(list(val["label"]), preds, average="macro"), 3)}
print(log)
json.dump(log, open("results/smoke/distilbert.json", "w"), indent=2)
```

Xong khi: chạy hết không lỗi, có `results/smoke/distilbert.json`. `val_macro_f1_smoke` chỉ để xem
mô hình có học không; **không** ghi nó như kết quả. Khoảng 0,26 nghĩa là mô hình chỉ đoán "neutral"
(lớp chiếm 65%) — vẫn chấp nhận cho chạy thử, nhưng ghi lại để so ở tuần 3.
(TV2 đã chạy đoạn code này trên CPU với 256 mẫu: chạy hết, ra 0,264, ~31 giây.)

## 3.3 Chạy thử Qwen2.5-7B 4-bit, đọc logits A/B/C

Quyết định đã chốt (`context/decisions.md`): xác suất LLM lấy từ **logits token A/B/C, gộp biến thể
có/không khoảng trắng** — không đọc chữ mô hình sinh ra.

- Tải ~15 GB (lần đầu ~5–10 phút). T4 không hỗ trợ bf16 → `compute_dtype=float16`.
- Nếu hết RAM khi nạp: Runtime → Restart, chạy lại chỉ ô này (đừng nạp DistilBERT cùng lúc).

```python
import json, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

name = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
model.eval()

PROMPT = ("Classify the sentiment of this financial news headline for the main company or market it is about.\n"
          "Headline: {text}\n"
          "A) bearish\nB) neutral\nC) bullish\n"
          "Answer with one letter only.")
LETTERS = {"A": "bearish", "B": "neutral", "C": "bullish"}
# Token ids for "A" and " A" (with/without leading space); keep only single-token variants.
LETTER_IDS = {L: sorted({ids[0] for v in (L, " " + L)
                         if len(ids := tok.encode(v, add_special_tokens=False)) == 1})
              for L in LETTERS}


def letter_probs(text):
    """Return (probs over bearish/neutral/bullish, mass on the A/B/C tokens, top-10 next tokens)."""
    chat = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(text=text)}],
                                   add_generation_prompt=True, tokenize=False)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    per_letter = torch.stack([torch.logsumexp(logp[LETTER_IDS[L]], dim=0) for L in LETTERS])
    probs = torch.softmax(per_letter, dim=0)                        # renormalized over A/B/C
    top10 = [tok.decode([i]) for i in logits.topk(10).indices.tolist()]
    return dict(zip(LETTERS.values(), probs.tolist())), per_letter.exp().sum().item(), top10


texts = list(load_dataset("zeroshot/twitter-financial-news-sentiment")["train"]
             .shuffle(seed=1).select(range(20))["text"])     # FNS is sorted by label: shuffle first
t0, rows = time.time(), []
for text in texts:
    probs, mass, top10 = letter_probs(text)
    rows.append({"probs": probs, "abc_mass": round(mass, 3), "top10": top10})
seconds_per_headline = (time.time() - t0) / len(texts)
assert all(abs(sum(r["probs"].values()) - 1) < 1e-4 for r in rows)
print("letter ids:", LETTER_IDS, "| s/headline:", round(seconds_per_headline, 2))
print("min mass on A/B/C:", min(r["abc_mass"] for r in rows))
json.dump({"model": name, "quant": "nf4-4bit", "seconds_per_headline": round(seconds_per_headline, 2),
           "min_abc_mass": min(r["abc_mass"] for r in rows), "letter_ids": LETTER_IDS},
          open("results/smoke/qwen.json", "w"), indent=2)
```

Xong khi: 20 tiêu đề FNS đều có xác suất tổng = 1 (lệnh `assert` qua), và bạn ghi lại:
- `min_abc_mass` — phần xác suất mô hình đặt vào A/B/C trước khi chuẩn hóa. Nếu thấp (< 0,5),
  top-10 cho thấy mô hình muốn nói gì khác → đem đến tuần 2 (việc "prompt A/B/C, kiểm tra top-10 logits").
- `seconds_per_headline` — dùng cho 3.4.

`results/smoke/qwen.json` không chứa văn bản tiêu đề → commit được.

## 3.4 Ước tính giờ GPU T4

Điền bảng này vào biên bản CP2 bằng số đo ở 3.2 và 3.3
(`t` = `epoch_seconds` của 3.2 trên 1.000 mẫu; train thật ≈ 70% × 11.931 ≈ 8.350 mẫu → × 8,35):

| Thí nghiệm | Số lần chạy | Giờ ước tính |
| --- | --- | --- |
| E5 DistilBERT | 3 seed × tối đa N epoch | 3 × N × 8,35 × t / 3600 |
| E7 đóng băng | 3 seed | ≈ E5 hoặc ít hơn |
| E8 FinBERT | 3 seed | ≈ E5 (BERT-base lớn hơn DistilBERT ~2×) |
| E9 split ngẫu nhiên | 3 seed | ≈ E5 |
| E10 rò rỉ (3 mức) | 3 mức × 3 seed | ≈ 3 × E5 |
| E12 nhãn cứng LLM, E6 chưng cất | 3 seed mỗi cái | ≈ E5 + phần dữ liệu 2026 |
| Nhãn LLM cho kho không nhãn | ~2.200 tiêu đề × 1 lần | 2.200 × `seconds_per_headline` / 3600 |
| E11 LLM zero-shot | validation + 3 tập test (tuần 6) | (số câu) × `seconds_per_headline` / 3600 |

So tổng với hạn mức: Kaggle cho ~30 giờ GPU/tuần; Colab miễn phí không cố định (thường vài giờ/ngày).
Nếu tổng vượt, đề xuất tại CP2 theo thứ tự cắt trong SPEC: E13 → E12 → FastAPI/Docker.

---

## Mang đến CP2 (28/10)

- [ ] PR `colab-smoke` với `notebooks/colab_setup.ipynb` (output đã xóa) và `results/smoke/*.json`
- [ ] Kết quả `pip install -r requirements.lock` trên Colab (chạy được hay lỗi gì, Python mấy)
- [ ] Bảng 3.4 đã điền số
