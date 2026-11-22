"""Transformer classifier runner for E5 (DistilBERT fine-tune), E7 (frozen encoder), and E8 (FinBERT).

Usage:
    python -m models.transformer --config configs/E5.yaml [--seeds 1 2 3]
    python -m models.transformer --config configs/E7.yaml [--seeds 1 2 3]

Reads the frozen ID-only split via train.data.load_split_data (verifying SHA-256), trains with
weighted cross-entropy (computed on train only), AdamW, linear warmup + decay schedule, FP16 AMP
on CUDA, and early stopping on validation Macro-F1. Writes per seed:
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
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score

from train.data import DEFAULT_RAW, LABELS, load_split_data

NUM_CLASSES = len(LABELS)
HEAD_PREFIXES = ("pre_classifier", "classifier", "score", "out_proj")
NO_DECAY_TOKENS = ("bias", "LayerNorm.weight", "layer_norm.weight")


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
    """Return a seeded DataLoader that tokenizes batches into PyTorch tensors."""
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
        ids, texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        return list(ids), dict(enc), torch.tensor(labels, dtype=torch.long)

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
        for batch_ids, enc, y in loader:
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
        for _, enc, y in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(**enc)
                logits = (out.logits if hasattr(out, "logits") else out[0]).float()
                loss = loss_fn(logits, y)
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
    started = datetime.now(timezone.utc)
    set_seed(seed)

    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    train_ex, val_ex = data["train"], data["validation"]

    if tokenizer_loader is not None:
        tokenizer = tokenizer_loader(mcfg)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(mcfg["pretrained_name"])

    model, param_stats = build_transformer_model(mcfg, model_loader=model_loader)
    model = model.to(device)

    loss_fn = build_loss(tcfg["loss"], [y for _, _, y in train_ex], gamma=tcfg.get("focal_gamma", 2.0))
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
    write_json(
        out / "run.json",
        {
            **meta,
            "label_order": list(LABELS),
            "model_stats": param_stats,
            "history": history,
            **summary,
        },
    )
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
    """Load the frozen split once and train all requested seeds for E5/E7/E8."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Path to configs/E5.yaml or configs/E7.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the config's seeds")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for summary in run_transformer_experiment(cfg, seeds=args.seeds):
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
