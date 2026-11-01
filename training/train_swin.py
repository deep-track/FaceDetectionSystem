"""
Train the Swin Transformer image detector and write data/best_swin.pth.

Data layout (made by scripts/prepare_image_dataset.py, or arranged by hand):

    data_dir/train/real/*.jpg   data_dir/train/fake/*.jpg
    data_dir/val/real/*.jpg     data_dir/val/fake/*.jpg
    data_dir/test/real/*.jpg    data_dir/test/fake/*.jpg      (test is optional)

Class order is fixed to Real = 0, Fake = 1, the same order the API uses.

    python training/train_swin.py --data_dir data/image_dataset --epochs 12 --batch 32

Per-group error rates (the fairness check):
    Make a CSV with columns  path,group  (path relative to data_dir, e.g. test/real/x.jpg;
    group e.g. a Monk or Fitzpatrick skin-tone label) and pass --groups_csv. The test split is
    then reported per group: false-positive rate (a REAL face called FAKE), false-negative rate,
    and the gap between the best and worst group. scripts/prepare_image_dataset.py writes a
    manifest.csv with an empty "group" column you can fill in and pass directly.

A GPU is strongly recommended (Swin-S has ~50M parameters). Free options include Google Colab and
Kaggle notebooks; copy the finished best_swin.pth back into data/.
"""
import argparse
import csv
import io
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
# same architecture + same eval preprocessing the API uses, so nothing can drift apart
from backend.core.image_predictor import ARCH, CLASSES, SwinTransformer, preprocess  # noqa: E402

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ── data ──────────────────────────────────────────────────────────────────────

class FolderDataset(Dataset):
    """split_dir/real -> 0, split_dir/fake -> 1 (explicit, NOT alphabetical like ImageFolder)."""

    def __init__(self, data_dir, split, transform):
        self.data_dir, self.transform = data_dir, transform
        self.items = []   # (relative path, label)
        for label, cls in enumerate(("real", "fake")):
            folder = os.path.join(data_dir, split, cls)
            if not os.path.isdir(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(IMAGE_EXT):
                    self.items.append((f"{split}/{cls}/{f}", label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        img = Image.open(os.path.join(self.data_dir, rel)).convert("RGB")
        return self.transform(img), label, i

    def labels(self):
        return np.array([l for _, l in self.items])


class RandomJPEG:
    """Re-compress at a random quality. Real-world uploads are compressed; the model must not
    rely on pristine pixels."""

    def __init__(self, p=0.4, quality=(40, 95)):
        self.p, self.quality = p, quality

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def train_transform():
    # No hue jitter on purpose: shifting hue would alter skin colour. Brightness/contrast
    # jitter helps with dim or unevenly lit photos, which matter for darker skin tones.
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.1, hue=0.0),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.2),
        RandomJPEG(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ── training / evaluation ─────────────────────────────────────────────────────

def video_key(rel_path):
    """'test/fake/Deepfakes__000_003__f00012.jpg' -> 'Deepfakes__000_003' (all frames of one video)."""
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    parts = stem.split("__")
    return "__".join(parts[:-1]) if len(parts) >= 3 else stem


def video_level(ds, probs, labels, idxs, threshold=0.5):
    """Average P(fake) over the frames of each video, then score videos instead of frames."""
    per_video, label_of = defaultdict(list), {}
    for p, l, i in zip(probs, labels, idxs):
        k = video_key(ds.items[i][0])
        per_video[k].append(p)
        label_of[k] = int(l)
    keys = sorted(per_video)
    return summarize(np.array([np.mean(per_video[k]) for k in keys]), np.array([label_of[k] for k in keys]),
                     threshold=threshold)


def choose_threshold(probs, labels, target_fpr=None):
    """
    Pick the P(fake) cut-off on the VALIDATION set (never on test), so the API's Real/Fake call
    is not stuck at an arbitrary 0.5.
      default      : best balanced accuracy (Youden's J = detection rate - false-positive rate)
      --target_fpr : catch as many fakes as possible while calling at most this share of real faces fake
    """
    if not ((labels == 0).any() and (labels == 1).any()):
        return 0.5, "default 0.5 (validation set lacks a class)"
    fpr, tpr, thr = roc_curve(labels, probs)
    ok = np.isfinite(thr)
    fpr, tpr, thr = fpr[ok], tpr[ok], thr[ok]
    if target_fpr is not None:
        allowed = fpr <= target_fpr
        if not allowed.any():
            return float(np.clip(thr.max(), 0.01, 0.99)), f"strictest cut-off (val FPR could not reach {target_fpr:.0%})"
        i = int(np.argmax(np.where(allowed, tpr, -1.0)))
        method = f"most fakes caught with val FPR <= {target_fpr:.0%}"
    else:
        i = int(np.argmax(tpr - fpr))
        method = "best balanced accuracy on val (Youden's J)"
    return float(np.clip(thr[i], 0.01, 0.99)), method


def count_videos(ds):
    vids = {(video_key(rel), label) for rel, label in ds.items}
    return sum(1 for _, l in vids if l == 0), sum(1 for _, l in vids if l == 1)


def run_eval(model, loader, device, criterion):
    model.eval()
    probs, labels, idxs, loss_sum, n = [], [], [], 0.0, 0
    with torch.no_grad():
        for x, y, i in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss_sum += criterion(out, y).item() * len(y)
            n += len(y)
            probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
            labels.append(y.cpu().numpy())
            idxs.append(i.numpy())
    probs, labels, idxs = np.concatenate(probs), np.concatenate(labels), np.concatenate(idxs)
    m = summarize(probs, labels, loss_sum / max(n, 1))
    m["video"] = video_level(loader.dataset, probs, labels, idxs)
    return m, probs, labels, idxs


def summarize(probs, labels, loss=None, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    real, fake = labels == 0, labels == 1
    out = {
        "n": int(len(labels)),
        "n_real": int(real.sum()),
        "n_fake": int(fake.sum()),
        "acc": float((preds == labels).mean()) if len(labels) else float("nan"),
        # false-positive rate: REAL called FAKE. This is the harm to a person wrongly accused.
        "fpr": float(preds[real].mean()) if real.any() else float("nan"),
        # false-negative rate: FAKE called REAL
        "fnr": float(1 - preds[fake].mean()) if fake.any() else float("nan"),
    }
    out["auc"] = float(roc_auc_score(labels, probs)) if real.any() and fake.any() else float("nan")
    if loss is not None:
        out["loss"] = float(loss)
    return out


def fmt(m):
    parts = [f"n={m['n']} ({m['n_real']} real/{m['n_fake']} fake)", f"acc={m['acc']:.4f}",
             f"auc={m['auc']:.4f}", f"FPR(real->fake)={m['fpr']:.4f}", f"FNR={m['fnr']:.4f}"]
    if "loss" in m:
        parts.insert(1, f"loss={m['loss']:.4f}")
    return "  ".join(parts)


def load_groups(csv_path):
    """
    CSV with a "group" column plus either "path" (one image, relative to data_dir) or
    "video" (a whole video, e.g. "real__000" or "Deepfakes__000_003", as the frame file names start).
    The manifest.csv from prepare_image_dataset.py works as is once its "group" column is filled.
    """
    groups = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            g = (row.get("group") or "").strip()
            if not g:
                continue
            if row.get("path"):
                groups[row["path"].replace("\\", "/")] = g
            elif row.get("video"):
                groups[row["video"].strip()] = g
    return groups


def clean_nan(obj):
    """NaN is not valid JSON; write null instead."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def group_report(ds, probs, labels, idxs, groups, threshold=0.5):
    by_group = defaultdict(list)
    for p, l, i in zip(probs, labels, idxs):
        rel = ds.items[i][0]
        g = groups.get(rel) or groups.get(video_key(rel))
        if g:
            by_group[g].append((p, l))
    report = {}
    for g, rows in sorted(by_group.items()):
        arr = np.array(rows)
        report[g] = summarize(arr[:, 0], arr[:, 1].astype(int), threshold=threshold)
    return report


def main():
    ap = argparse.ArgumentParser(description="Train the Swin image detector")
    ap.add_argument("--data_dir",   required=True)
    ap.add_argument("--output",     default="data/best_swin.pth")
    ap.add_argument("--epochs",     type=int,   default=12)
    ap.add_argument("--batch",      type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-4, help="backbone learning rate (head uses 10x)")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--freeze_epochs", type=int, default=1,
                    help="epochs to train only the new classifier head before unfreezing the backbone")
    ap.add_argument("--patience",   type=int,   default=4, help="stop after this many epochs without val AUC gain")
    ap.add_argument("--no_pretrained", action="store_true",
                    help="random init instead of ImageNet weights (only for quick tests)")
    ap.add_argument("--save_fp16", action="store_true",
                    help="store the weights in half precision (~half the file size, ~98 MB instead of ~197 MB)")
    ap.add_argument("--target_fpr", type=float, default=None,
                    help="choose the decision threshold so that at most this share of REAL validation faces "
                         "is called fake (e.g. 0.05). Default: best balanced accuracy")
    ap.add_argument("--groups_csv", default=None, help="CSV with columns path,group for per-group test metrics")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda" else "   (CPU training will be very slow)"))

    train_ds = FolderDataset(args.data_dir, "train", train_transform())
    val_ds   = FolderDataset(args.data_dir, "val",   preprocess)
    test_ds  = FolderDataset(args.data_dir, "test",  preprocess)
    if len(train_ds) == 0 or len(val_ds) == 0:
        sys.exit(f"Need images in {args.data_dir}/train/{{real,fake}} and {args.data_dir}/val/{{real,fake}}. "
                 f"See scripts/prepare_image_dataset.py.")

    y_train = train_ds.labels()
    n_real, n_fake = int((y_train == 0).sum()), int((y_train == 1).sum())
    if n_real == 0 or n_fake == 0:
        sys.exit("The training set needs both real and fake images.")
    print(f"train: {len(train_ds)} images (real {n_real}, fake {n_fake}) | val: {len(val_ds)} | test: {len(test_ds)}")
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        r, f = count_videos(ds)
        print(f"  {name:<5} videos: real {r}, fake {f}")
    vr, _ = count_videos(val_ds)
    if vr < 20:
        print(f"  [WARN] only {vr} real videos in val: validation scores will be noisy.")

    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_dl   = DataLoader(val_ds,   args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_dl  = DataLoader(test_ds,  args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=pin) if len(test_ds) else None

    model = SwinTransformer(num_classes=2, pretrained=not args.no_pretrained).to(device)

    # class-balanced loss: weights make the (smaller) real class count as much as the fake class overall
    total = n_real + n_fake
    weights = torch.tensor([total / (2 * n_real), total / (2 * n_fake)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    opt = torch.optim.AdamW([
        {"params": model.base_model.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr * 10},
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    best_auc, best_epoch, bad = -1.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        for p in model.base_model.parameters():
            p.requires_grad = not frozen
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out, _ = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
            seen += len(y)
        sched.step()

        val_m, *_ = run_eval(model, val_dl, device, criterion)
        print(f"epoch {epoch:>2}/{args.epochs}{' (head only)' if frozen else ''}  "
              f"train_loss={run_loss / max(seen, 1):.4f}  [{time.time() - t0:.0f}s]\n"
              f"   val frames: {fmt(val_m)}\n"
              f"   val videos: {fmt(val_m['video'])}", flush=True)

        score = val_m["auc"] if not np.isnan(val_m["auc"]) else val_m["acc"]
        if score > best_auc:
            best_auc, best_epoch, bad = score, epoch, 0
            state = model.state_dict()
            if args.save_fp16:
                state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
            torch.save({
                "model_state_dict": state,
                "classes": CLASSES, "arch": ARCH, "img_size": 224,
                "epoch": epoch, "val": val_m,
            }, args.output)
            print(f"  saved best model -> {args.output}")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  no improvement for {args.patience} epochs, stopping early")
                break

    print(f"\nBest epoch: {best_epoch} (val frame AUC {best_auc:.4f}; model selection uses this)")

    # reload the best checkpoint
    ckpt = torch.load(args.output, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # calibrate the Real/Fake cut-off on validation and store it in the model file (the API reads it)
    val_m, vprobs, vlabels, vidx = run_eval(model, val_dl, device, criterion)
    threshold, method = choose_threshold(vprobs, vlabels, args.target_fpr)
    ckpt["threshold"], ckpt["threshold_method"] = threshold, method
    torch.save(ckpt, args.output)
    val_t = summarize(vprobs, vlabels, threshold=threshold)
    val_t["video"] = video_level(val_ds, vprobs, vlabels, vidx, threshold)
    print(f"\nDecision threshold: P(fake) >= {threshold:.3f}  [{method}]  (saved in the model file)")
    if not 0.1 <= threshold <= 0.9:
        print("   [WARN] this cut-off is far from 0.5, which usually means the validation set is small "
              "or the model is poorly calibrated. Check it on the full run before trusting it.")
    print(f"   val frames @0.5:   {fmt(val_m)}")
    print(f"   val frames @{threshold:.3f}: {fmt(val_t)}")
    report = {"best_epoch": best_epoch, "threshold": threshold, "threshold_method": method,
              "val_at_threshold": val_t}

    # final test with the best checkpoint, at 0.5 and at the calibrated threshold
    if test_dl is not None:
        test_m, probs, labels, idxs = run_eval(model, test_dl, device, criterion)
        test_t = summarize(probs, labels, threshold=threshold)
        test_t["video"] = video_level(test_ds, probs, labels, idxs, threshold)
        print(f"\nTEST @0.5    frames: {fmt(test_m)}")
        print(f"TEST @0.5    videos: {fmt(test_m['video'])}")
        print(f"TEST @{threshold:.3f}  frames: {fmt(test_t)}   <- what the API will do (one image at a time)")
        print(f"TEST @{threshold:.3f}  videos: {fmt(test_t['video'])}")
        report["test"] = test_m
        report["test_at_threshold"] = test_t
        if args.groups_csv:
            groups = load_groups(args.groups_csv)
            per_group = group_report(test_ds, probs, labels, idxs, groups, threshold)
            if per_group:
                print(f"\nPer-group test results at threshold {threshold:.3f} (FPR = real faces wrongly called fake):")
                for g, m in per_group.items():
                    print(f"  {g:<14} {fmt(m)}")
                fprs = [m["fpr"] for m in per_group.values() if not np.isnan(m["fpr"])]
                if len(fprs) > 1:
                    print(f"  FPR gap (worst - best group): {max(fprs) - min(fprs):.4f}")
                report["per_group"] = per_group
            else:
                print("\n[WARN] --groups_csv matched none of the test images (check the path column).")
    report_path = os.path.splitext(args.output)[0] + "_report.json"
    with open(report_path, "w") as f:
        json.dump(clean_nan(report), f, indent=2)
    print(f"\nReport saved -> {report_path}")
    print(f"Model saved -> {args.output}")


if __name__ == "__main__":
    main()
