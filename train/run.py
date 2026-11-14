"""Run one experiment config for every seed: data -> vocab -> embeddings -> model -> train -> files.

Usage:
    python -m train.run --config configs/E2.yaml [--seeds 1]

Writes, per seed, results/<E>/seed<k>/{run.json, vocab.json, preds_validation.jsonl} and the best
weights to checkpoints/<E>/seed<k>/best.pt (gitignored). Only validation is predicted.
"""
import argparse
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import yaml

import models.bilstm_attention  # noqa: F401  (registers "bilstm_attention")
import models.textcnn  # noqa: F401  (registers "textcnn")
from models.embeddings import build_vocab, glove_embeddings, random_embeddings, unk_rate
from models.registry import MODELS, register_model  # noqa: F401  (register_model re-exported for tests)
from train.data import DEFAULT_RAW, LABELS, load_split_data
from train.losses import build_loss
from train.loop import evaluate, make_loader, macro_f1, set_seed, train_model
from train.runlog import run_metadata, write_json


def build_embeddings(model_cfg, vocab, seed, root):
    """Return (matrix, stats) from GloVe ('glove.6B.100d' -> data/raw/glove/glove.6B.100d.txt) or 'random'."""
    name = model_cfg["embedding"]
    if name == "random":
        dim = model_cfg.get("embedding_dim", 100)
        return random_embeddings(vocab, dim, seed), {"embedding": "random", "dim": dim}
    matrix, stats = glove_embeddings(Path(root) / "data/raw/glove" / f"{name}.txt", vocab, seed)
    return matrix, {"embedding": name, **stats}


def write_preds(path, ids, probs):
    """Write {"id", "probs": [bearish, neutral, bullish]} per line (docs/WEEK2.md contract)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, p in zip(ids, probs):
            f.write(json.dumps({"id": i, "probs": [round(float(x), 6) for x in p]}) + "\n")


def run_seed(cfg, data, seed, root, results_dir, checkpoint_dir, device):
    """Train one seed and write its files; return a short summary dict."""
    started = datetime.now(timezone.utc)
    set_seed(seed)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    train_ex, val_ex = data["train"], data["validation"]
    vocab = build_vocab([t for _, t, _ in train_ex], min_freq=dcfg.get("min_freq", 1))
    encode = partial(vocab.encode, max_len=dcfg["max_len"])
    matrix, emb_stats = build_embeddings(mcfg, vocab, seed, root)
    if mcfg["name"] not in MODELS:
        raise ValueError(f"Model '{mcfg['name']}' is not registered yet (known: {sorted(MODELS)})")
    model = MODELS[mcfg["name"]](mcfg, matrix).to(device)
    loss_fn = build_loss(tcfg["loss"], [y for _, _, y in train_ex], gamma=tcfg.get("focal_gamma", 2.0))
    if tcfg.get("optimizer", "adam") != "adam":
        raise ValueError(f"Unsupported optimizer '{tcfg['optimizer']}'")
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    train_loader = make_loader(train_ex, encode, tcfg["batch_size"], shuffle=True, seed=seed)
    val_loader = make_loader(val_ex, encode, tcfg["batch_size"], shuffle=False, seed=seed)
    history, best_epoch = train_model(
        model, train_loader, val_loader, loss_fn, optimizer,
        max_epochs=tcfg["max_epochs"], patience=tcfg["early_stopping"]["patience"], device=device,
        amp=tcfg.get("mixed_precision", False), grad_clip=tcfg.get("grad_clip"))
    ids, probs, labels, _ = evaluate(model, val_loader, device)
    out = Path(results_dir) / cfg["experiment"] / f"seed{seed}"
    write_preds(out / "preds_validation.jsonl", ids, probs)
    vocab.save(out.parent / f"seed{seed}" / "vocab.json")
    ckpt = Path(checkpoint_dir) / cfg["experiment"] / f"seed{seed}" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    summary = {"best_epoch": best_epoch, "val_macro_f1": macro_f1(labels, probs.argmax(1)),
               "val_unk_rate": unk_rate(vocab, [t for _, t, _ in val_ex])}
    write_json(out / "run.json", {**run_metadata(cfg, seed, started, cwd=root), "label_order": list(LABELS),
                                  "embedding_stats": emb_stats, "history": history, **summary})
    return {"seed": seed, **summary}


def run_experiment(cfg, root=".", seeds=None, results_dir="results", checkpoint_dir="checkpoints", device=None):
    """Load the frozen split once, then train every seed. Output: list of per-seed summaries."""
    root = Path(root)
    dcfg = cfg["data"]
    data = load_split_data(root / dcfg["split_file"], dcfg["split_sha256"],
                           [root / p for p in dcfg.get("raw_files", DEFAULT_RAW)])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return [run_seed(cfg, data, seed, root, results_dir, checkpoint_dir, device)
            for seed in (seeds or cfg["seeds"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", help="override the config's seeds")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for summary in run_experiment(cfg, seeds=args.seeds):
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
