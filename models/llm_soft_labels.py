"""Generate LLM soft labels with winning prompt P2 on the 2026 train_unlabeled pool (task 3.11).

Usage:
    python -m models.llm_soft_labels --headlines data/raw/headlines_2026/headlines.jsonl

Strict experimental integrity safeguards (AGENTS.md rule 5 & docs/WEEK3.md task 3.11):
1. Only rows with pool == "train_unlabeled" are eligible; "test_candidate" rows are never scored.
2. Any ID present in labeling/trial30_ids.csv or labeling/test300_ids.csv is strictly excluded.
3. Outputs in results/llm_soft_labels/p2_soft_labels.jsonl contain ONLY IDs and numeric probabilities/
   log-probabilities (never headline text or URLs).
4. Execution is resumable: already-scored IDs in the output JSONL are preserved and skipped on restart.
"""
import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from evaluate.metrics import read_preds
from train.data import LABELS

DEFAULT_HEADLINES = Path("data/raw/headlines_2026/headlines.jsonl")
DEFAULT_PROMPTS_CFG = Path("configs/llm_prompts.yaml")
DEFAULT_OUT_DIR = Path("results/llm_soft_labels")
DEFAULT_EXCLUSION_FILES = (
    Path("labeling/trial30_ids.csv"),
    Path("labeling/test300_ids.csv"),
)
SELECTED_PROMPT_NAME = "P2"


def load_excluded_ids(paths=DEFAULT_EXCLUSION_FILES) -> set:
    """Read all headline IDs from trial30_ids.csv and test300_ids.csv (if present) to block test/trial leakage."""
    excluded = set()
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "id" not in reader.fieldnames:
                raise ValueError(f"Exclusion CSV '{path}' must have an 'id' column")
            for row in reader:
                hid = (row.get("id") or "").strip()
                if hid:
                    excluded.add(hid)
    return excluded


def load_unlabeled_2026_pool(headlines_path=DEFAULT_HEADLINES, excluded_ids=None) -> tuple:
    """Filter exported 2026 headlines strictly to train_unlabeled and exclude any trial/test IDs.

    Input: path to headlines.jsonl, set of excluded IDs.
    Output: (eligible_items, stats) where eligible_items is a list of (id, title) tuples.
    """
    excluded = set(excluded_ids or ())
    path = Path(headlines_path)
    if not path.is_file():
        raise FileNotFoundError(f"Unlabeled headlines file not found: {path}")

    eligible, seen = [], set()
    total_rows = 0
    skipped_test_candidate = 0
    skipped_excluded = 0

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            hid = (row.get("id") or "").strip()
            title = (row.get("title") or row.get("text") or "").strip()
            pool = row.get("pool")
            if not hid or not title:
                raise ValueError(f"Missing 'id' or 'title' at line {line_no} of {path}")
            if hid in seen:
                raise ValueError(f"Duplicate headline id '{hid}' in {path}")
            seen.add(hid)

            if hid in excluded:
                skipped_excluded += 1
                continue
            if pool != "train_unlabeled":
                skipped_test_candidate += 1
                continue
            eligible.append((hid, title))

    stats = {
        "total_rows": total_rows,
        "eligible_train_unlabeled": len(eligible),
        "skipped_non_train_pool": skipped_test_candidate,
        "skipped_excluded_trial_or_test_ids": skipped_excluded,
    }
    return eligible, stats


def load_p2_prompt_template(config_path=DEFAULT_PROMPTS_CFG) -> tuple:
    """Load the winning P2 prompt template and model config from configs/llm_prompts.yaml."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    prompts = cfg.get("prompts", {})
    if SELECTED_PROMPT_NAME not in prompts:
        raise ValueError(f"Selected prompt '{SELECTED_PROMPT_NAME}' not found in {config_path}")
    return prompts[SELECTED_PROMPT_NAME], cfg


def validate_soft_label_row(row: dict) -> None:
    """Verify that a soft-label record has valid probabilities, no headline text, and finite values."""
    forbidden_keys = {"text", "title", "headline", "url"}
    overlap = forbidden_keys.intersection(row.keys())
    if overlap:
        raise ValueError(f"Soft-label record must never store text/URL fields, found: {sorted(overlap)}")
    if "id" not in row or "probs" not in row:
        raise ValueError(f"Soft-label record missing 'id' or 'probs': {row}")
    probs = row["probs"]
    if (
        not isinstance(probs, list)
        or len(probs) != len(LABELS)
        or not all(isinstance(p, (int, float)) and math.isfinite(float(p)) and 0.0 <= float(p) <= 1.0 for p in probs)
        or abs(sum(probs) - 1.0) > 1e-3
    ):
        raise ValueError(f"Invalid probabilities in soft-label row for id '{row.get('id')}': {probs}")


def build_qwen_scorer(model_cfg: dict):
    """Load Qwen2.5-7B-Instruct in 4-bit NF4 on CUDA and return a letter_probs(prompt) callable."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        raise RuntimeError(
            "CUDA GPU is required to run Qwen/Qwen2.5-7B-Instruct 4-bit NF4 for soft-label generation."
        )

    model_name = model_cfg["name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    model.eval()

    letters = {"A": "bearish", "B": "neutral", "C": "bullish"}
    letter_ids = {
        L: sorted({ids[0] for v in (L, " " + L) if len(ids := tok.encode(v, add_special_tokens=False)) == 1})
        for L in letters
    }
    for letter, ids in letter_ids.items():
        if not ids:
            raise ValueError(f"No single-token IDs found for option {letter!r}")

    def score_prompt(prompt: str):
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tok(chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        per_letter = torch.stack([torch.logsumexp(logp[letter_ids[L]], dim=0) for L in letters])
        probs = torch.softmax(per_letter, dim=0)
        top10 = [tok.decode([i]) for i in logits.topk(10).indices.tolist()]
        return (
            dict(zip(letters.values(), probs.tolist())),
            float(per_letter.exp().sum().item()),
            top10,
            [round(float(x), 6) for x in per_letter.tolist()],
        )

    return score_prompt


def generate_soft_labels(
    headlines_path=DEFAULT_HEADLINES,
    config_path=DEFAULT_PROMPTS_CFG,
    out_dir=DEFAULT_OUT_DIR,
    exclusion_files=DEFAULT_EXCLUSION_FILES,
    scorer_fn=None,
    limit: int = None,
) -> dict:
    """Run P2 soft-label generation on train_unlabeled 2026 headlines with resume support.

    Writes:
      - <out_dir>/p2_soft_labels.jsonl (ID + probs + letter_logp + abc_mass, no headline text)
      - <out_dir>/run.json (provenance and summary diagnostics)
    """
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    prompt_template, cfg = load_p2_prompt_template(config_path)
    excluded_ids = load_excluded_ids(exclusion_files)
    eligible, pool_stats = load_unlabeled_2026_pool(headlines_path, excluded_ids=excluded_ids)
    if limit is not None and limit > 0:
        eligible = eligible[:limit]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "p2_soft_labels.jsonl"

    existing_records = {}
    if out_jsonl.is_file():
        with open(out_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                validate_soft_label_row(row)
                if row["id"] in excluded_ids:
                    raise ValueError(f"Excluded trial/test ID '{row['id']}' found in existing soft labels!")
                existing_records[row["id"]] = row

    pending = [(hid, title) for hid, title in eligible if hid not in existing_records]
    if pending and scorer_fn is None:
        scorer_fn = build_qwen_scorer(cfg["model"])

    newly_scored = 0
    top1_letter_hits = 0
    with open(out_jsonl, "a", encoding="utf-8", newline="\n") as f:
        for hid, title in pending:
            prompt = prompt_template.format(text=title)
            result = scorer_fn(prompt)
            if len(result) == 4:
                probs_dict, mass, top10, letter_logp = result
            else:
                probs_dict, mass, top10 = result
                letter_logp = None
            probs_list = [
                round(float(probs_dict["bearish"]), 6),
                round(float(probs_dict["neutral"]), 6),
                round(float(probs_dict["bullish"]), 6),
            ]
            top1_is_letter = bool(top10 and top10[0].strip() in ("A", "B", "C"))
            top1_letter_hits += int(top1_is_letter)
            record = {
                "id": hid,
                "probs": probs_list,
                "abc_mass": round(float(mass), 4),
                "top1_is_letter": top1_is_letter,
            }
            if letter_logp is not None:
                record["letter_logp"] = letter_logp
            validate_soft_label_row(record)
            f.write(json.dumps(record) + "\n")
            f.flush()
            existing_records[hid] = record
            newly_scored += 1

    # Verify the written JSONL is 100% compatible with evaluate.metrics.read_preds
    verified_preds = read_preds(out_jsonl)
    all_masses = [float(r["abc_mass"]) for r in existing_records.values()]
    all_top1 = [bool(r.get("top1_is_letter", True)) for r in existing_records.values()]

    summary = {
        "prompt": SELECTED_PROMPT_NAME,
        "model": cfg["model"]["name"],
        "quantization": cfg["model"].get("quantization", "nf4-4bit"),
        "label_order": list(LABELS),
        "pool_stats": pool_stats,
        "n_scored_total": len(verified_preds),
        "n_newly_scored": newly_scored,
        "min_abc_mass": round(min(all_masses), 4) if all_masses else None,
        "mean_abc_mass": round(sum(all_masses) / len(all_masses), 4) if all_masses else None,
        "share_top1_is_letter": round(sum(all_top1) / len(all_top1), 4) if all_top1 else None,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--headlines", default=str(DEFAULT_HEADLINES))
    parser.add_argument("--config", default=str(DEFAULT_PROMPTS_CFG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of headlines to score")
    args = parser.parse_args()
    summary = generate_soft_labels(
        headlines_path=args.headlines,
        config_path=args.config,
        out_dir=args.out_dir,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
