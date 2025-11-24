# train.py
import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import yaml
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from model import get_model
try:
    from utils import set_seed, accuracy as acc_fn
except ImportError:
    set_seed = None
    acc_fn = None


# -------------------------
# 1) Dataset
# -------------------------
class ImageCsvDataset(Dataset):
    """
    CSV columns: path, label
    path is relative to img_dir
    """
    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.df = pd.read_csv(csv_path)
        if "path" not in self.df.columns or "label" not in self.df.columns:
            # fallback if csv has no header
            self.df.columns = ["path", "label"]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        rel_path = str(self.df.iloc[idx]["path"])
        label = int(self.df.iloc[idx]["label"])

        img_path = os.path.join(self.img_dir, rel_path)
        img = Image.open(img_path).convert("RGB")

        # PIL -> Tensor [C,H,W] float32 in [0,1]
        img = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
             .view(img.size[1], img.size[0], 3)
             .numpy())
        )  # H,W,C uint8

        img = img.permute(2, 0, 1).float() / 255.0  # C,H,W float

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# -------------------------
# 2) Pad Collate
# -------------------------
def pad_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Right/bottom zero-padding to [B,C,H_max,W_max]
    """
    imgs, labels = zip(*batch)

    # Find max H/W in batch
    H_max = max(im.shape[1] for im in imgs)
    W_max = max(im.shape[2] for im in imgs)

    padded_imgs = []
    for im in imgs:
        C, H, W = im.shape
        canvas = torch.zeros((C, H_max, W_max), dtype=im.dtype)
        canvas[:, :H, :W] = im
        padded_imgs.append(canvas)

    images = torch.stack(padded_imgs, dim=0)  # B,C,H_max,W_max
    labels = torch.tensor(labels, dtype=torch.long)  # B

    return images, labels


# -------------------------
# 3) Train / Eval loops
# -------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module,
                    device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total += bs

    avg_loss = total_loss / total
    avg_acc = total_correct / total
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        bs = X.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total += bs

    avg_loss = total_loss / total
    avg_acc = total_correct / total
    return avg_loss, avg_acc


# -------------------------
# 4) Checkpoint / Resume
# -------------------------
def save_checkpoint(state: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt.get("best_acc", 0.0)
    return start_epoch, best_acc


# -------------------------
# 5) Main (config-based)
# -------------------------
def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed fix
    if set_seed is not None:
        set_seed(seed)
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # datasets
    train_ds = ImageCsvDataset(
        csv_path=data_cfg["train_csv"],
        img_dir=data_cfg["img_dir"],
        transform=None
    )
    val_ds = ImageCsvDataset(
        csv_path=data_cfg["val_csv"],
        img_dir=data_cfg["img_dir"],
        transform=None
    )

    # loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=pad_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=pad_collate_fn,
        pin_memory=True
    )

    # model / optim / loss
    num_classes = model_cfg["num_classes"]
    model = get_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0)
    )

    # resume
    start_epoch = 0
    best_acc = 0.0
    resume_path = train_cfg.get("resume_path", "")
    if resume_path:
        start_epoch, best_acc = load_checkpoint(resume_path, model, optimizer)
        print(f"[Resume] Loaded checkpoint from {resume_path} (start_epoch={start_epoch}, best_acc={best_acc:.4f})")

    save_dir = train_cfg.get("save_dir", "ckpts")
    best_path = os.path.join(save_dir, "best.pth")

    patience = train_cfg.get("early_stop_patience", 999999)
    bad_epochs = 0

    epochs = train_cfg["epochs"]
    for epoch in range(start_epoch, epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  train loss: {tr_loss:.4f} | train acc: {tr_acc:.4f}")
        print(f"  val   loss: {va_loss:.4f} | val   acc: {va_acc:.4f}")

        # best ckpt
        if va_acc > best_acc:
            best_acc = va_acc
            bad_epochs = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": cfg
                },
                best_path
            )
            print(f"  best acc updated -> saved {best_path}")
        else:
            bad_epochs += 1
            print(f"  no improvement ({bad_epochs}/{patience})")

        # early stopping
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Training done. Best val acc = {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
