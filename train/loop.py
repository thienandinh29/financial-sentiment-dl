"""Shared training loop for the RNN/CNN models: seeding, batching, early stopping on val Macro-F1.

A model is any nn.Module with forward(tokens, lengths) -> logits of shape (batch, 3), where
tokens is a padded LongTensor (batch, time) and lengths a LongTensor (batch,). Nothing here
touches the test sets: evaluate() is only called on validation (AGENTS.md rule 1).
"""
import copy
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from models.embeddings import PAD_ID, UNK_ID

NUM_CLASSES = 3


def set_seed(seed):
    """Seed Python, NumPy and torch (CPU and CUDA); make cuDNN choose deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextDataset(Dataset):
    """Examples (id, text, label) encoded once; an empty encoding becomes [UNK_ID]."""

    def __init__(self, examples, encode):
        self.items = [(i, encode(text) or [UNK_ID], label) for i, text, label in examples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    """Pad a batch: returns (ids, tokens (B, T), lengths (B,), labels (B,))."""
    ids, seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    tokens = torch.full((len(seqs), int(lengths.max())), PAD_ID, dtype=torch.long)
    for row, seq in enumerate(seqs):
        tokens[row, :len(seq)] = torch.tensor(seq)
    return list(ids), tokens, lengths, torch.tensor(labels)


def make_loader(examples, encode, batch_size, shuffle, seed):
    """Return a DataLoader with a seeded shuffle order (same seed -> same batches)."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TextDataset(examples, encode), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, collate_fn=collate)


def macro_f1(labels, preds):
    """Macro-F1 over the 3 classes; a class never predicted counts as F1 = 0."""
    return float(f1_score(labels, preds, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0))


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None):
    """Return (ids, probs (n, 3) array, labels array, mean loss or None) in loader order."""
    model.eval()
    ids, probs, labels, losses, n = [], [], [], 0.0, 0
    for batch_ids, tokens, lengths, y in loader:
        logits = model(tokens.to(device), lengths).float()
        if loss_fn is not None:
            losses += loss_fn(logits, y.to(device)).item() * len(y)
            n += len(y)
        ids += batch_ids
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.append(y.numpy())
    return ids, np.concatenate(probs), np.concatenate(labels), (losses / n if n else None)


def train_model(model, train_loader, val_loader, loss_fn, optimizer, *, max_epochs, patience,
                device, amp=False, grad_clip=None):
    """Train with early stopping on validation Macro-F1 and restore the best weights.

    Output: (history, best_epoch). history has one dict per epoch: train_loss, val_loss,
    val_macro_f1, seconds. Training stops after `patience` epochs without improvement.
    """
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = loss_fn.to(device)
    history, best_f1, best_epoch, best_state, waited = [], -1.0, 0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        start, total, n = time.perf_counter(), 0.0, 0
        for _, tokens, lengths, y in train_loader:
            tokens, y = tokens.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(tokens, lengths).float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        _, probs, labels, val_loss = evaluate(model, val_loader, device, loss_fn)
        f1 = macro_f1(labels, probs.argmax(1))
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                        "val_macro_f1": f1, "seconds": round(time.perf_counter() - start, 3)})
        if f1 > best_f1:
            best_f1, best_epoch, waited = f1, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return history, best_epoch
