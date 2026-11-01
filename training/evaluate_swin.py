"""
Re-evaluate a trained image detector and (optionally) reset its Real/Fake cut-off, without retraining.

The cut-off is always chosen on the VALIDATION split; the test split is only reported.

  # call at most 3% of real validation faces fake, write the result to a new file
  python training/evaluate_swin.py --model data/best_swin.pth --data_dir data/image_dataset ^
      --target_fpr 0.03 --output data/best_swin_fpr3.pth

  # or set the cut-off by hand
  python training/evaluate_swin.py --model data/best_swin.pth --data_dir data/image_dataset --threshold 0.5

  # only report (keep the stored cut-off), with per-skin-tone results
  python training/evaluate_swin.py --model data/best_swin.pth --data_dir data/image_dataset --groups_csv groups.csv

--groups_csv: columns "group" and either "path" (image, relative to data_dir) or "video"
(e.g. "real__000", the part of the frame file name before "__f00012").

If --output is not given and the cut-off changes, the model file is updated in place and the
previous version is kept next to it as <name>.before_recalibration.pth.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from backend.core.image_predictor import _load_weights, preprocess          # noqa: E402
from train_swin import (FolderDataset, choose_threshold, clean_nan, count_videos,  # noqa: E402
                        fmt, group_report, load_groups, run_eval, summarize, video_level)


def predict_split(model, ds, device, batch, workers):
    """Run the model once over a split. Returns (probs, labels, idxs)."""
    dl = DataLoader(ds, batch, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    _, probs, labels, idxs = run_eval(model, dl, device, nn.CrossEntropyLoss())
    return probs, labels, idxs


def summarize_at(ds, probs, labels, idxs, thresholds):
    out = {}
    for name, t in thresholds.items():
        m = summarize(probs, labels, threshold=t)
        m["video"] = video_level(ds, probs, labels, idxs, t)
        out[name] = m
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate / recalibrate the Swin image detector")
    ap.add_argument("--model",      required=True, help="best_swin.pth to evaluate")
    ap.add_argument("--data_dir",   required=True, help="dataset made by scripts/prepare_image_dataset.py")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--target_fpr", type=float, default=None,
                       help="new cut-off: catch as many fakes as possible while calling at most this "
                            "share of REAL validation images fake (e.g. 0.03)")
    group.add_argument("--balanced", action="store_true",
                       help="new cut-off: best balanced accuracy on validation (what training does by default)")
    group.add_argument("--threshold", type=float, default=None, help="new cut-off set by hand (0-1)")
    ap.add_argument("--output",     default=None, help="where to write the recalibrated model (default: in place)")
    ap.add_argument("--groups_csv", default=None, help="per-group (e.g. skin tone) test results")
    ap.add_argument("--batch",      type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        ap.error("--threshold must be between 0 and 1")
    if args.target_fpr is not None and not 0.0 < args.target_fpr < 1.0:
        ap.error("--target_fpr must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, stored = _load_weights(args.model, device)
    print(f"Model: {args.model}  (stored cut-off: {stored:.3f})")

    val_ds  = FolderDataset(args.data_dir, "val",  preprocess)
    test_ds = FolderDataset(args.data_dir, "test", preprocess)
    if len(val_ds) == 0:
        sys.exit(f"No validation images in {args.data_dir}/val.")
    for name, ds in (("val", val_ds), ("test", test_ds)):
        r, f = count_videos(ds)
        print(f"  {name:<4} {len(ds)} images, videos: real {r}, fake {f}")

    # 1. choose the new cut-off on validation
    preds = {"val": predict_split(model, val_ds, device, args.batch, args.num_workers)}
    if len(test_ds):
        preds["test"] = predict_split(model, test_ds, device, args.batch, args.num_workers)
    vprobs, vlabels, _ = preds["val"]
    if args.target_fpr is not None or args.balanced:
        new_t, method = choose_threshold(vprobs, vlabels, args.target_fpr)
    elif args.threshold is not None:
        new_t, method = float(args.threshold), "set by hand"
    else:
        new_t, method = stored, "unchanged"

    thresholds = {"stored": stored, "0.5": 0.5, "new": new_t}
    print(f"\nCut-offs compared: stored {stored:.3f} | 0.5 | new {new_t:.3f} [{method}]")

    # 2. report val and test at each cut-off (frames = what the image API does per upload)
    report = {"model": args.model, "stored_threshold": stored, "new_threshold": new_t, "method": method}
    for name, ds in (("val", val_ds), ("test", test_ds)):
        if name not in preds:
            continue
        probs, labels, idxs = preds[name]
        res = summarize_at(ds, probs, labels, idxs, thresholds)
        report[name] = res
        print(f"\n{name.upper()}")
        for tname, m in res.items():
            print(f"  @{thresholds[tname]:.3f} ({tname:<6}) frames: {fmt(m)}")
            print(f"  @{thresholds[tname]:.3f} ({tname:<6}) videos: {fmt(m['video'])}")
        if name == "test" and args.groups_csv:
            per_group = group_report(ds, probs, labels, idxs, load_groups(args.groups_csv), new_t)
            if per_group:
                print(f"\nPer-group TEST results at cut-off {new_t:.3f} (FPR = real faces wrongly called fake):")
                for g, m in per_group.items():
                    print(f"  {g:<14} {fmt(m)}")
                fprs = [m["fpr"] for m in per_group.values() if not np.isnan(m["fpr"])]
                if len(fprs) > 1:
                    print(f"  FPR gap (worst - best group): {max(fprs) - min(fprs):.4f}")
                print("  Groups with few real videos give unreliable rates; check n_real before drawing conclusions.")
                report["per_group"] = per_group
            else:
                print("\n[WARN] --groups_csv matched none of the test images (check the path/video column).")

    # 3. save the new cut-off into the model file
    if method != "unchanged" and abs(new_t - stored) > 1e-9:
        out = args.output or args.model
        if os.path.abspath(out) == os.path.abspath(args.model):
            backup = os.path.splitext(args.model)[0] + ".before_recalibration.pth"
            shutil.copy2(args.model, backup)
            print(f"\nPrevious model kept as {backup}")
        ckpt = torch.load(args.model, map_location="cpu")
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            ckpt = {"model_state_dict": ckpt}
        ckpt["threshold"], ckpt["threshold_method"] = new_t, method
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        torch.save(ckpt, out)
        print(f"Saved model with cut-off {new_t:.3f} -> {out}")
        report["output"] = out
    else:
        out = args.model
        print("\nCut-off unchanged; model file not modified.")

    report_path = os.path.splitext(out)[0] + "_evaluation.json"
    if os.path.abspath(os.path.dirname(report_path)) == os.path.abspath(os.path.dirname(args.model)) and \
            not os.access(os.path.dirname(os.path.abspath(report_path)), os.W_OK):
        report_path = os.path.basename(report_path)   # read-only input folder (e.g. on Kaggle)
    with open(report_path, "w") as f:
        json.dump(clean_nan(report), f, indent=2)
    print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()
