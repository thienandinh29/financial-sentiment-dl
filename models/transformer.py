"""Transformer classifier runner for E5 (DistilBERT fine-tune), E6 (soft-label distillation),
E7 (frozen encoder), E8 (FinBERT), and E12 (hard pseudo-labels).

Usage:
    python -m models.transformer --config configs/E5.yaml [--seeds 1 2 3]
    python -m models.transformer --config configs/E7.yaml [--seeds 1 2 3]
    python -m models.transformer --config configs/E12.yaml [--seeds 1 2 3]
    python -m models.transformer --config configs/E6.yaml --e6-grid
    python -m models.transformer --config configs/E6.yaml [--seeds 1 2 3]

Reads the frozen ID-only split via train.data.load_split_data (verifying SHA-256), trains with
weighted cross-entropy (class weights computed on FNS train only), AdamW, linear warmup + decay
schedule, FP16 AMP on CUDA, and early stopping on validation Macro-F1. Writes per seed:
  - results/<E>/seed<k>/preds_validation.jsonl
  - results/<E>/seed<k>/run.json
  - checkpoints/<E>/seed<k>/best.pt (gitignored)
Never accesses or evaluates any test split (AGENTS.md rule 1).
"""
import argparse
import copy
import importlib.metadata
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score

from models.llm_soft_labels import (
    DEFAULT_EXCLUSION_FILES,
    DEFAULT_HEADLINES,
    EXPECTED_HEADLINES_SHA256,
    load_excluded_ids,
    validate_soft_label_row,
)
from train.data import DEFAULT_RAW, LABELS, load_split, load_split_data, sha256_file

NUM_CLASSES = len(LABELS)
HEAD_PREFIXES = ("pre_classifier", "classifier", "score", "out_proj")
NO_DECAY_TOKENS = ("bias", "LayerNorm.weight", "layer_norm.weight")
VALID_PSEUDO_MODES = (None, "hard", "soft")
E6_GRID_TEMPERATURES = (1.0, 2.0, 4.0)
E6_GRID_ALPHAS = (0.3, 0.5, 0.7)


def is_classifier_head_param(param_name: str) -> bool:
    """Return True if a parameter name belongs to the sequence classification head rather than the encoder."""
    parts = param_name.split(".")
    return any(p in HEAD_PREFIXES for p in parts)


def linear_warmup_lr_multiplier(current_step: int, total_steps: int, warmup_ratio: float = 0.1) -> float:
    """Compute the LR multiplier in [0.0, 1.0] for linear warmup followed by linear decay.

    Input: current_step (0-indexed or 1-indexed step count), total_steps (> 0), warmup_ratio in [0, 1].
    Output: float multiplier applied to base learning rate.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not (0.0 <= warmup_ratio <= 1.0):
        raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")
    step = max(0, min(current_step, total_steps))
    warmup_steps = int(round(total_steps * warmup_ratio))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    denom = max(1, total_steps - warmup_steps)
    return max(0.0, float(total_steps - step) / float(denom))


def get_distill_params(cfg: dict) -> tuple:
    """Extract (distill_alpha, distill_temperature) from cfg['data'] or cfg['train']."""
    dcfg, tcfg = cfg.get("data", {}), cfg.get("train", {})
    alpha = dcfg.get("distill_alpha", tcfg.get("distill_alpha"))
    temp = dcfg.get("distill_temperature", tcfg.get("distill_temperature"))
    return alpha, temp


def softmax_temperature(letter_logp, temperature: float = 1.0) -> list:
    """Compute temperature-scaled teacher distribution softmax(letter_logp / T) (docs/guides/TV3-tuan4.md).

    Input: 3-element sequence of teacher log-probabilities/logits for [bearish, neutral, bullish], T > 0.
    Output: list of 3 floats in (0, 1) summing to 1.0.
    """
    if temperature is None or not isinstance(temperature, (int, float)) or float(temperature) <= 0.0:
        raise ValueError(f"Temperature T must be positive, got {temperature}")
    if not isinstance(letter_logp, (list, tuple, np.ndarray)) or len(letter_logp) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} values in letter_logp, got {letter_logp}")
    t = float(temperature)
    scaled = []
    for x in letter_logp:
        val = float(x)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite value in letter_logp: {letter_logp}")
        scaled.append(val / t)
    max_val = max(scaled)
    exps = [math.exp(v - max_val) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def distillation_kl_loss(student_logits, teacher_target_T, temperature: float = 1.0, alpha: float = 0.5):
    """Compute alpha * T^2 * KL(target_T || softmax(student_logits / T)) for E6 distillation.

    Works on both PyTorch tensors (for GPU/CPU training) and Python lists / NumPy arrays (for unit tests).
    When alpha == 0, the distillation loss is exactly 0.
    """
    if temperature is None or float(temperature) <= 0.0:
        raise ValueError(f"Temperature T must be positive, got {temperature}")
    if alpha is None or float(alpha) < 0.0:
        raise ValueError(f"Distillation alpha must be non-negative, got {alpha}")

    t = float(temperature)
    a = float(alpha)

    # Check if student_logits is a PyTorch tensor without importing torch at top level if unnecessary
    if hasattr(student_logits, "device") and hasattr(student_logits, "dtype"):
        import torch
        from torch.nn import functional as F

        if a == 0.0 or student_logits.numel() == 0:
            return student_logits.sum() * 0.0
        log_p_student = F.log_softmax(student_logits.float() / t, dim=-1)
        target = teacher_target_T.to(device=student_logits.device, dtype=torch.float32)
        kl = F.kl_div(log_p_student, target, reduction="batchmean")
        return (a * (t ** 2)) * kl

    # Pure-Python / NumPy path for lightweight unit testing without torch
    logits_arr = np.asarray(student_logits, dtype=np.float64)
    target_arr = np.asarray(teacher_target_T, dtype=np.float64)
    if logits_arr.ndim == 1:
        logits_arr = logits_arr[None, :]
        target_arr = target_arr[None, :]
    if logits_arr.shape != target_arr.shape or logits_arr.shape[-1] != NUM_CLASSES:
        raise ValueError(f"Shape mismatch in distillation_kl_loss: {logits_arr.shape} vs {target_arr.shape}")
    if a == 0.0 or len(logits_arr) == 0:
        return 0.0

    scaled = logits_arr / t
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    log_p = scaled - np.log(np.exp(scaled).sum(axis=-1, keepdims=True))
    # KL(target || student) = sum_c target_c * (log(target_c) - log_p_c)
    safe_target = np.clip(target_arr, 1e-12, 1.0)
    kl_per_row = np.sum(target_arr * (np.log(safe_target) - log_p), axis=-1)
    return float(a * (t ** 2) * np.mean(kl_per_row))


def validate_transformer_config(cfg: dict) -> None:
    """Validate required fields and experimental integrity rules in a Transformer YAML config."""
    for section in ("experiment", "rq", "data", "model", "train", "seeds"):
        if section not in cfg:
            raise ValueError(f"Missing required config section '{section}'")
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    if not dcfg.get("split_file") or not dcfg.get("split_sha256"):
        raise ValueError("Config 'data' must specify both 'split_file' and 'split_sha256' (AGENTS.md rule 2)")
    if mcfg.get("num_labels", NUM_CLASSES) != NUM_CLASSES:
        raise ValueError(f"Expected num_labels={NUM_CLASSES} for {LABELS}, got {mcfg.get('num_labels')}")
    if not mcfg.get("pretrained_name"):
        raise ValueError("Config 'model' must specify 'pretrained_name'")
    if not isinstance(mcfg.get("freeze_encoder", False), bool):
        raise ValueError("Config 'model.freeze_encoder' must be a boolean")
    if tcfg.get("early_stopping", {}).get("metric", "val_macro_f1") != "val_macro_f1":
        raise ValueError("Early stopping metric must be 'val_macro_f1' (never test)")

    pseudo_path = dcfg.get("pseudo_labels")
    pseudo_mode = dcfg.get("pseudo_mode")
    if pseudo_mode not in VALID_PSEUDO_MODES:
        raise ValueError(f"Invalid data.pseudo_mode {pseudo_mode!r}; expected one of {VALID_PSEUDO_MODES}")
    if (pseudo_path is None) != (pseudo_mode is None):
        raise ValueError("Config 'data.pseudo_labels' and 'data.pseudo_mode' must be set together")
    if pseudo_mode == "soft":
        alpha, temp = get_distill_params(cfg)
        if alpha is None or not isinstance(alpha, (int, float)) or not (0.0 <= float(alpha) <= 10.0):
            raise ValueError(f"Soft distillation requires valid distill_alpha, got {alpha!r}")
        if temp is None or not isinstance(temp, (int, float)) or float(temp) <= 0.0:
            raise ValueError(f"Soft distillation requires distill_temperature > 0, got {temp!r}")


def load_pseudo_labeled_examples(
    pseudo_jsonl_path,
    headlines_path=DEFAULT_HEADLINES,
    *,
    pseudo_mode: str = "hard",
    temperature: float = 1.0,
    excluded_ids=None,
    fns_split_ids=None,
    expected_headlines_sha256: str = EXPECTED_HEADLINES_SHA256,
) -> tuple:
    """Load and validate 2026 pseudo-labeled examples for E12 (hard argmax) or E6 (soft temperature-scaled).

    Enforces strict leakage guards (docs/guides/TV3-tuan4.md tasks 3.13 & 3.14):
      1. Verifies SHA-256 of data/raw/headlines_2026/headlines.jsonl when expected_headlines_sha256 is set.
      2. Rejects duplicate IDs in pseudo_labels JSONL.
      3. Rejects any ID in the 330 excluded trial/test IDs (trial30_ids.csv + assignment.csv).
      4. Rejects any ID belonging to the FNS dataset (train, validation, or test splits).
      5. Rejects any ID missing from headlines.jsonl or with pool != 'train_unlabeled'.

    Returns:
      (examples, stats) where examples is a list of tuples:
        - hard mode: (id, text, hard_label_int)
        - soft mode: (id, text, hard_label_int, target_T_list)
    """
    if pseudo_mode not in ("hard", "soft"):
        raise ValueError(f"pseudo_mode must be 'hard' or 'soft', got {pseudo_mode!r}")
    if pseudo_mode == "soft" and (temperature is None or float(temperature) <= 0.0):
        raise ValueError(f"temperature must be > 0 for soft pseudo_mode, got {temperature!r}")

    pseudo_path = Path(pseudo_jsonl_path)
    if not pseudo_path.is_file():
        raise FileNotFoundError(f"Pseudo-label file not found: {pseudo_path}")

    raw_path = Path(headlines_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw 2026 headlines file not found: {raw_path}")

    actual_raw_sha = sha256_file(raw_path)
    if expected_headlines_sha256 and actual_raw_sha != expected_headlines_sha256:
        raise ValueError(
            f"Headlines SHA-256 mismatch for {raw_path}: expected {expected_headlines_sha256}, got {actual_raw_sha}"
        )

    excluded = set(excluded_ids) if excluded_ids is not None else load_excluded_ids(DEFAULT_EXCLUSION_FILES)
    fns_ids = set(fns_split_ids or ())

    headlines_by_id = {}
    with open(raw_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            hid = (row.get("id") or "").strip()
            title = (row.get("title") or row.get("text") or "").strip()
            pool = row.get("pool")
            if not hid or not title:
                raise ValueError(f"Missing id or title at line {line_no} of {raw_path}")
            if hid in headlines_by_id:
                raise ValueError(f"Duplicate id '{hid}' in {raw_path}")
            headlines_by_id[hid] = (title, pool)

    examples = []
    seen_pseudo_ids = set()
    class_counts = {label: 0 for label in LABELS}
    confidences = []
    soft_entropies = []

    with open(pseudo_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_soft_label_row(row)
            hid = row["id"].strip()
            if hid in seen_pseudo_ids:
                raise ValueError(f"Duplicate id '{hid}' in pseudo-label file {pseudo_path} at line {line_no}")
            seen_pseudo_ids.add(hid)

            if hid in excluded:
                raise ValueError(f"Excluded trial/test 2026 ID '{hid}' found in pseudo-labels!")
            if hid in fns_ids:
                raise ValueError(f"FNS split ID '{hid}' found in 2026 pseudo-labels!")
            if hid not in headlines_by_id:
                raise ValueError(f"Pseudo-label ID '{hid}' not found in {raw_path}")

            text, pool = headlines_by_id[hid]
            if pool != "train_unlabeled":
                raise ValueError(f"Pseudo-label ID '{hid}' belongs to pool {pool!r}, expected 'train_unlabeled'")

            probs = [float(p) for p in row["probs"]]
            hard_label = int(max(range(NUM_CLASSES), key=lambda k: probs[k]))
            conf = float(probs[hard_label])
            class_counts[LABELS[hard_label]] += 1
            confidences.append(conf)

            if pseudo_mode == "hard":
                examples.append((hid, text, hard_label))
            else:
                if "letter_logp" in row and row["letter_logp"] is not None:
                    logp = row["letter_logp"]
                else:
                    logp = [math.log(max(p, 1e-12)) for p in probs]
                target_t = softmax_temperature(logp, temperature=float(temperature))
                ent = -sum(p * math.log(max(p, 1e-12)) for p in target_t)
                soft_entropies.append(ent)
                examples.append((hid, text, hard_label, target_t))

    if not examples:
        raise ValueError(f"No pseudo-labeled examples loaded from {pseudo_path}")

    stats = {
        "pseudo_labels_path": str(pseudo_path),
        "pseudo_labels_sha256": sha256_file(pseudo_path),
        "headlines_sha256": actual_raw_sha,
        "pseudo_mode": pseudo_mode,
        "n_retained": len(examples),
        "n_excluded_by_threshold": 0,
        "hard_class_counts": class_counts,
        "confidence_summary": {
            "min": round(min(confidences), 4),
            "mean": round(sum(confidences) / len(confidences), 4),
            "median": round(float(statistics.median(confidences)), 4),
            "max": round(max(confidences), 4),
        },
    }
    if pseudo_mode == "soft":
        stats["distill_temperature"] = float(temperature)
        stats["mean_target_entropy_nats"] = round(sum(soft_entropies) / len(soft_entropies), 4)
    return examples, stats


def write_preds(path: Path, ids, probs) -> None:
    """Write {"id", "probs": [bearish, neutral, bullish]} per line (docs/WEEK2.md contract)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, p in zip(ids, probs):
            f.write(json.dumps({"id": i, "probs": [round(float(x), 6) for x in p]}) + "\n")


def macro_f1_np(labels, preds) -> float:
    """Macro-F1 over the 3 canonical classes; a class never predicted counts as F1 = 0."""
    return float(f1_score(labels, preds, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0))


def configure_encoder_freezing(model, freeze_encoder: bool = False) -> dict:
    """Freeze encoder parameters when freeze_encoder=True, keeping classification head trainable.

    Returns parameter statistics dict: total_params, trainable_params, frozen_params, freeze_encoder.
    """
    for name, param in model.named_parameters():
        if freeze_encoder:
            param.requires_grad = is_classifier_head_param(name)
        else:
            param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise ValueError("No trainable parameters remain after configuring freeze_encoder")
    if freeze_encoder and trainable == total:
        raise ValueError("freeze_encoder=True did not freeze any encoder parameters")
    return {
        "freeze_encoder": bool(freeze_encoder),
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
    }


def build_transformer_model(model_cfg: dict, model_loader=None):
    """Instantiate a sequence classification Transformer and apply dropout / encoder freezing."""
    pretrained_name = model_cfg["pretrained_name"]
    num_labels = model_cfg.get("num_labels", NUM_CLASSES)
    dropout = model_cfg.get("dropout")

    if model_loader is not None:
        model = model_loader(model_cfg)
    else:
        from transformers import AutoConfig, AutoModelForSequenceClassification

        hf_cfg = AutoConfig.from_pretrained(pretrained_name, num_labels=num_labels)
        if dropout is not None:
            if hasattr(hf_cfg, "seq_classif_dropout"):
                hf_cfg.seq_classif_dropout = float(dropout)
            if hasattr(hf_cfg, "classifier_dropout"):
                hf_cfg.classifier_dropout = float(dropout)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, config=hf_cfg)

    param_stats = configure_encoder_freezing(model, freeze_encoder=model_cfg.get("freeze_encoder", False))
    return model, param_stats


def build_optimizer_and_scheduler(model, train_cfg: dict, steps_per_epoch: int):
    """Build AdamW optimizer (with decoupled weight decay) and linear warmup + decay scheduler."""
    import torch

    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    max_epochs = int(train_cfg["max_epochs"])
    total_steps = max(1, steps_per_epoch * max_epochs)

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in NO_DECAY_TOKENS):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    opt_name = train_cfg.get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer '{train_cfg.get('optimizer')}'")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: linear_warmup_lr_multiplier(step, total_steps, warmup_ratio=warmup_ratio),
    )
    return optimizer, scheduler


def make_transformer_loader(examples, tokenizer, max_len: int, batch_size: int, shuffle: bool, seed: int):
    """Return a seeded DataLoader that tokenizes batches into PyTorch tensors.

    Supports standard 3-tuples (id, text, label) for E5/E7/E8/E12 and 5-tuples
    (id, text, label, soft_target_3, is_pseudo_bool) for E6 soft distillation.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    class RawTextDataset(Dataset):
        def __init__(self, rows):
            self.rows = list(rows)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    def collate_fn(batch):
        first = batch[0]
        if len(first) == 3:
            ids, texts, labels = zip(*batch)
            enc = tokenizer(
                list(texts),
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            return list(ids), dict(enc), torch.tensor(labels, dtype=torch.long)
        elif len(first) == 5:
            ids, texts, labels, soft_targets, is_pseudo = zip(*batch)
            enc = tokenizer(
                list(texts),
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            return (
                list(ids),
                dict(enc),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(soft_targets, dtype=torch.float32),
                torch.tensor(is_pseudo, dtype=torch.bool),
            )
        else:
            raise ValueError(f"Unexpected example tuple length: {len(first)}")

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        RawTextDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_fn,
    )


def evaluate_transformer(model, loader, device, loss_fn=None):
    """Evaluate a Transformer on validation loader; return (ids, probs, labels, mean_loss)."""
    import torch

    model.eval()
    ids, probs, labels, total_loss, total_n = [], [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch_ids, enc, y = batch[0], batch[1], batch[2]
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = (out.logits if hasattr(out, "logits") else out[0]).float()
            if loss_fn is not None:
                total_loss += loss_fn(logits, y.to(device)).item() * len(y)
                total_n += len(y)
            ids.extend(batch_ids)
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            labels.append(y.numpy())
    mean_loss = (total_loss / total_n) if total_n > 0 else None
    return ids, np.concatenate(probs, axis=0), np.concatenate(labels, axis=0), mean_loss


def compute_batch_loss(
    logits,
    y,
    loss_fn,
    *,
    soft_targets=None,
    is_pseudo=None,
    distill_alpha: float = None,
    distill_temperature: float = None,
):
    """Compute batch loss for standard CE/weighted_ce (E5/E7/E12) or mixed FNS CE + soft KL distillation (E6)."""
    if is_pseudo is None or soft_targets is None:
        return loss_fn(logits, y)

    fns_mask = ~is_pseudo
    pseudo_mask = is_pseudo
    n_fns = int(fns_mask.sum().item())
    n_pseudo = int(pseudo_mask.sum().item())
    n_total = max(1, n_fns + n_pseudo)

    loss_total = logits.sum() * 0.0
    if n_fns > 0:
        loss_fns = loss_fn(logits[fns_mask], y[fns_mask])
        loss_total = loss_total + (float(n_fns) / float(n_total)) * loss_fns
    if n_pseudo > 0 and distill_alpha is not None and float(distill_alpha) > 0.0:
        loss_pseudo = distillation_kl_loss(
            logits[pseudo_mask],
            soft_targets[pseudo_mask],
            temperature=float(distill_temperature),
            alpha=float(distill_alpha),
        )
        loss_total = loss_total + (float(n_pseudo) / float(n_total)) * loss_pseudo
    return loss_total


def train_transformer(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    *,
    max_epochs: int,
    patience: int,
    device,
    amp: bool = False,
    grad_clip: float = 1.0,
    distill_alpha: float = None,
    distill_temperature: float = None,
):
    """Train Transformer with FP16 AMP, linear warmup scheduler, and early stopping on val Macro-F1."""
    import torch

    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = loss_fn.to(device)
    history, best_f1, best_epoch, best_state, waited = [], -1.0, 0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        start, total_loss, total_n = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            if len(batch) == 3:
                _, enc, y = batch
                soft_targets, is_pseudo = None, None
            else:
                _, enc, y, soft_targets, is_pseudo = batch
                soft_targets = soft_targets.to(device)
                is_pseudo = is_pseudo.to(device)
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(**enc)
                logits = (out.logits if hasattr(out, "logits") else out[0]).float()
                loss = compute_batch_loss(
                    logits,
                    y,
                    loss_fn,
                    soft_targets=soft_targets,
                    is_pseudo=is_pseudo,
                    distill_alpha=distill_alpha,
                    distill_temperature=distill_temperature,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=float(grad_clip),
                )
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * len(y)
            total_n += len(y)

        if use_amp:
            torch.cuda.synchronize()
        _, probs, labels, val_loss = evaluate_transformer(model, val_loader, device, loss_fn)
        f1 = macro_f1_np(labels, probs.argmax(axis=1))
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_n),
            "val_loss": val_loss,
            "val_macro_f1": f1,
            "seconds": round(time.perf_counter() - start, 3),
        })
        if f1 > best_f1:
            best_f1, best_epoch, waited = f1, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch


def prepare_training_examples_with_pseudo(cfg: dict, fns_train_ex: list, root: Path) -> tuple:
    """Combine FNS train examples with optional E12 (hard) or E6 (soft) 2026 pseudo-labeled examples.

    Guarantees that FNS train class weights are computed BEFORE calling this function.
    """
    dcfg = cfg["data"]
    pseudo_rel = dcfg.get("pseudo_labels")
    pseudo_mode = dcfg.get("pseudo_mode")
    if not pseudo_rel or not pseudo_mode:
        return list(fns_train_ex), None

    all_splits = load_split(root / dcfg["split_file"], dcfg["split_sha256"])
    fns_all_ids = {i for ids in all_splits.values() for i in ids}
    excluded_2026_ids = load_excluded_ids([root / p for p in DEFAULT_EXCLUSION_FILES])
    headlines_path = root / dcfg.get("pseudo_headlines_file", DEFAULT_HEADLINES)
    expected_sha = dcfg.get("pseudo_headlines_sha256", EXPECTED_HEADLINES_SHA256)

    alpha, temp = get_distill_params(cfg)
    pseudo_rows, pseudo_stats = load_pseudo_labeled_examples(
        root / pseudo_rel,
        headlines_path,
        pseudo_mode=pseudo_mode,
        temperature=float(temp) if temp is not None else 1.0,
        excluded_ids=excluded_2026_ids,
        fns_split_ids=fns_all_ids,
        expected_headlines_sha256=expected_sha,
    )
    pseudo_stats["fns_train_count"] = len(fns_train_ex)
    pseudo_stats["combined_train_count"] = len(fns_train_ex) + len(pseudo_rows)

    if pseudo_mode == "hard":
        combined = list(fns_train_ex) + list(pseudo_rows)
        return combined, pseudo_stats

    # Soft distillation mode (E6): attach 5-tuples (id, text, label, soft_target_3, is_pseudo_bool)
    pseudo_stats["distill_alpha"] = float(alpha)
    fns_5tuples = []
    for hid, text, y in fns_train_ex:
        one_hot = [1.0 if c == int(y) else 0.0 for c in range(NUM_CLASSES)]
        fns_5tuples.append((hid, text, int(y), one_hot, False))
    pseudo_5tuples = [
        (hid, text, int(hard_y), target_t, True) for hid, text, hard_y, target_t in pseudo_rows
    ]
    return fns_5tuples + pseudo_5tuples, pseudo_stats


def run_transformer_seed(
    cfg: dict,
    data: dict,
    seed: int,
    root=".",
    results_dir="results",
    checkpoint_dir="checkpoints",
    device=None,
    model_loader=None,
    tokenizer_loader=None,
) -> dict:
    """Train one seed for a Transformer experiment and write run.json + preds_validation.jsonl."""
    import torch
    from train.losses import build_loss
    from train.loop import set_seed
    from train.runlog import run_metadata, write_json

    validate_transformer_config(cfg)
    root = Path(root)
    started = datetime.now(timezone.utc)
    set_seed(seed)

    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    fns_train_ex, val_ex = data["train"], data["validation"]

    if tokenizer_loader is not None:
        tokenizer = tokenizer_loader(mcfg)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(mcfg["pretrained_name"])

    model, param_stats = build_transformer_model(mcfg, model_loader=model_loader)
    model = model.to(device)

    # Docs/guides/TV3-tuan4.md rule: class weights are ALWAYS computed on FNS train only (never pseudo-labels)
    loss_fn = build_loss(tcfg["loss"], [y for _, _, y in fns_train_ex], gamma=tcfg.get("focal_gamma", 2.0))
    train_ex, pseudo_stats = prepare_training_examples_with_pseudo(cfg, fns_train_ex, root)
    distill_alpha, distill_temperature = get_distill_params(cfg)

    train_loader = make_transformer_loader(
        train_ex, tokenizer, dcfg["max_len"], tcfg["batch_size"], shuffle=True, seed=seed
    )
    val_loader = make_transformer_loader(
        val_ex, tokenizer, dcfg["max_len"], tcfg["batch_size"], shuffle=False, seed=seed
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, tcfg, steps_per_epoch=len(train_loader))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best_epoch = train_transformer(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        max_epochs=int(tcfg["max_epochs"]),
        patience=int(tcfg["early_stopping"]["patience"]),
        device=device,
        amp=bool(tcfg.get("mixed_precision", False)),
        grad_clip=tcfg.get("grad_clip", 1.0),
        distill_alpha=float(distill_alpha) if distill_alpha is not None else None,
        distill_temperature=float(distill_temperature) if distill_temperature is not None else None,
    )

    ids, probs, labels, _ = evaluate_transformer(model, val_loader, device)
    out = Path(results_dir) / cfg["experiment"] / f"seed{seed}"
    write_preds(out / "preds_validation.jsonl", ids, probs)

    ckpt = Path(checkpoint_dir) / cfg["experiment"] / f"seed{seed}" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)

    meta = run_metadata(cfg, seed, started, cwd=root)
    try:
        meta["libraries"]["transformers"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        meta["libraries"]["transformers"] = None

    peak_gpu_gb = (
        round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None
    )
    summary = {
        "best_epoch": best_epoch,
        "val_macro_f1": macro_f1_np(labels, probs.argmax(axis=1)),
        "peak_gpu_mem_gb": peak_gpu_gb,
    }
    run_payload = {
        **meta,
        "label_order": list(LABELS),
        "model_stats": param_stats,
        "history": history,
        **summary,
    }
    if pseudo_stats is not None:
        run_payload["pseudo_stats"] = pseudo_stats
    write_json(out / "run.json", run_payload)
    return {"seed": seed, **param_stats, **summary}


def run_transformer_experiment(
    cfg: dict,
    root=".",
    seeds=None,
    results_dir="results",
    checkpoint_dir="checkpoints",
    device=None,
    model_loader=None,
    tokenizer_loader=None,
) -> list:
    """Load the frozen split once and train all requested seeds for E5/E6/E7/E8/E12."""
    import torch

    validate_transformer_config(cfg)
    root = Path(root)
    dcfg = cfg["data"]
    data = load_split_data(
        root / dcfg["split_file"],
        dcfg["split_sha256"],
        [root / p for p in dcfg.get("raw_files", DEFAULT_RAW)],
    )
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return [
        run_transformer_seed(
            cfg,
            data,
            seed,
            root=root,
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            device=resolved_device,
            model_loader=model_loader,
            tokenizer_loader=tokenizer_loader,
        )
        for seed in (seeds or cfg["seeds"])
    ]


def format_e6_grid_markdown(grid_rows: list, selected: dict) -> str:
    """Format the 9-point E6 validation grid search (T in {1,2,4} x alpha in {0.3,0.5,0.7}, seed 1) as Markdown."""
    lines = [
        "# Kết quả Dò Lưới Siêu tham số Chưng cất Nhãn mềm `E6` trên Validation (`seed = 1`)",
        "",
        "Lưới thực nghiệm theo `docs/guides/TV3-tuan4.md` (Mục 3.14): $T \\in \\{1, 2, 4\\} \\times \\alpha \\in \\{0.3, 0.5, 0.7\\}$ (9 tổ hợp, chạy trên `seed = 1`, đánh giá trên `1,791` mẫu `validation` của `data/splits/cluster_split.jsonl`, tuyệt đối không dùng tập `test`).",
        "",
        "| Thứ tự | Nhiệt độ ($T$) | Hệ số ($\\alpha$) | Best Epoch | Val Macro-F1 | Bearish F1 | Neutral F1 | Bullish F1 | Thời gian (s) | Peak VRAM (GB) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for idx, r in enumerate(grid_rows, start=1):
        is_best = (
            float(r["temperature"]) == float(selected["temperature"])
            and float(r["alpha"]) == float(selected["alpha"])
        )
        b_tag = "**" if is_best else ""
        lines.append(
            f"| {idx} | {b_tag}`{r['temperature']:.1f}`{b_tag} | {b_tag}`{r['alpha']:.1f}`{b_tag} | "
            f"{r['best_epoch']} | {b_tag}`{r['macro_f1']:.4f}`{b_tag} | "
            f"`{r['f1']['bearish']:.4f}` | `{r['f1']['neutral']:.4f}` | `{r['f1']['bullish']:.4f}` | "
            f"`{r['elapsed_seconds']:.2f}` | `{r.get('peak_gpu_mem_gb')}` |"
        )
    lines.extend([
        "",
        "## Tổ hợp được chọn (theo quy tắc Val Macro-F1 cao nhất trên `seed = 1`)",
        "",
        f"- **Nhiệt độ chọn ($T^*$):** `{selected['temperature']:.1f}`",
        f"- **Trọng số chưng cất chọn ($\\alpha^*$):** `{selected['alpha']:.1f}`",
        f"- **Validation Macro-F1 (`seed = 1`):** `{selected['macro_f1']:.4f}` "
        f"(`bearish = {selected['f1']['bearish']:.4f}`, `neutral = {selected['f1']['neutral']:.4f}`, "
        f"`bullish = {selected['f1']['bullish']:.4f}`)",
        "",
    ])
    return "\n".join(lines)


def run_e6_grid_search(
    base_cfg: dict,
    root=".",
    results_dir="results",
    checkpoint_dir="checkpoints",
    temperatures=E6_GRID_TEMPERATURES,
    alphas=E6_GRID_ALPHAS,
    seed: int = 1,
    device=None,
    model_loader=None,
    tokenizer_loader=None,
) -> dict:
    """Run the 9-point E6 validation grid (T in {1,2,4} x alpha in {0.3,0.5,0.7}, seed 1) with resume support.

    Writes each grid cell to results/E6/grid/T<T>_alpha<alpha>/seed1/{run.json,preds_validation.jsonl}
    and the full table to results/E6/grid.md. Returns the winning configuration and all 9 rows.
    """
    import torch
    from evaluate.metrics import read_preds, score, validation_gold

    root = Path(root)
    results_root = Path(results_dir)
    dcfg = base_cfg["data"]
    data = load_split_data(
        root / dcfg["split_file"],
        dcfg["split_sha256"],
        [root / p for p in dcfg.get("raw_files", DEFAULT_RAW)],
    )
    gold = {i: y for i, _, y in data["validation"]}
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    grid_rows = []
    for temp in temperatures:
        for alpha in alphas:
            tag = f"T{float(temp):.1f}_alpha{float(alpha):.1f}".replace(".", "p")
            cell_exp = f"E6/grid/{tag}"
            cell_out = results_root / cell_exp / f"seed{seed}"
            run_path = cell_out / "run.json"
            pred_path = cell_out / "preds_validation.jsonl"

            if not (run_path.is_file() and pred_path.is_file()):
                cell_cfg = copy.deepcopy(base_cfg)
                cell_cfg["experiment"] = cell_exp
                cell_cfg["data"]["pseudo_mode"] = "soft"
                cell_cfg["data"]["distill_temperature"] = float(temp)
                cell_cfg["data"]["distill_alpha"] = float(alpha)
                run_transformer_seed(
                    cell_cfg,
                    data,
                    seed=seed,
                    root=root,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir,
                    device=resolved_device,
                    model_loader=model_loader,
                    tokenizer_loader=tokenizer_loader,
                )

            run_data = json.loads(run_path.read_text(encoding="utf-8"))
            scored = score(read_preds(pred_path), gold)
            grid_rows.append({
                "tag": tag,
                "temperature": float(temp),
                "alpha": float(alpha),
                "seed": seed,
                "best_epoch": run_data["best_epoch"],
                "macro_f1": scored["macro_f1"],
                "val_macro_f1_exact": float(run_data["val_macro_f1"]),
                "f1": scored["f1"],
                "elapsed_seconds": float(run_data["elapsed_seconds"]),
                "peak_gpu_mem_gb": run_data.get("peak_gpu_mem_gb"),
            })

    selected = max(grid_rows, key=lambda r: (r["macro_f1"], r["val_macro_f1_exact"]))
    e6_dir = results_root / "E6"
    e6_dir.mkdir(parents=True, exist_ok=True)
    grid_md_path = e6_dir / "grid.md"
    grid_md_path.write_text(format_e6_grid_markdown(grid_rows, selected), encoding="utf-8")
    (e6_dir / "grid.json").write_text(
        json.dumps({"selected": selected, "grid": grid_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"selected": selected, "grid": grid_rows, "grid_md": str(grid_md_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Path to configs/E5.yaml, E6.yaml, E7.yaml, or E12.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the config's seeds")
    parser.add_argument("--e6-grid", action="store_true", help="Run the 9-point E6 (T, alpha) grid on seed 1 first")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.e6_grid:
        grid_res = run_e6_grid_search(cfg)
        print(json.dumps(grid_res["selected"]))
    else:
        for summary in run_transformer_experiment(cfg, seeds=args.seeds):
            print(json.dumps(summary))


if __name__ == "__main__":
    main()
