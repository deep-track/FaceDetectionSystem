"""
Build the face-crop image dataset used to train the Swin image detector.

Input folders can contain videos (.mp4 .avi .mov .mkv), images (.jpg .jpeg .png .webp .bmp) or both.
Every item goes through the SAME face crop the API applies at inference time
(backend/core/face_crop.py), so training and serving see identical inputs.

  python scripts/prepare_image_dataset.py ^
      --real_dir  "D:/data/FF++/original" ^
      --fake_dirs "D:/data/FF++/Deepfakes" "D:/data/FF++/FaceSwap" "D:/data/FF++/Face2Face" ^
      --out_dir   data/image_dataset ^
      --limit 300 --fake_limit 100 --frames_per_video 8 --workers 4

Output:
  out_dir/train|val|test/real|fake/<source>__<video>__f00012.jpg
  out_dir/manifest.csv    path, split, label, source, video, identity, group
                          ("group" is empty on purpose: fill it with a skin-tone label,
                           e.g. Monk/Fitzpatrick, to get per-group error rates from train_swin.py)

How the split works (important):
  A FaceForensics++ fake named 000_003 is real video 000 with the face of person 003 swapped in.
  If real 000 sat in the test set while 000_003 sat in train, the model could learn "this scene
  is fake" and score real test videos as fake. So videos are grouped by the people/scenes in
  their names (000, 003, 000_003, 003_000 -> one group, across ALL folders), and each whole
  group goes to one split. Files whose names do not follow the FF++ / DeepFakeDetection naming
  are their own group.
"""
import argparse
import csv
import os
import random
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from backend.core.face_crop import FaceCropper                           # noqa: E402
from backend.core.model_files import resolve_model_file, LANDMARKER_URL  # noqa: E402

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SPLITS    = ("train", "val", "test")
MANIFEST_COLUMNS = ["path", "split", "label", "source", "video", "identity", "group"]

GENERIC_FOLDER_NAMES = {"videos", "video", "images", "image", "frames", "c0", "c23", "c40", "raw",
                        "compressed", "data", "train", "val", "test"}

# FaceForensics++:   original "000", manipulated "000_003"
# DeepFakeDetection: original "01__walking...", manipulated "01_02__walking..."
FF_REAL  = re.compile(r"^(\d{3})$")
FF_FAKE  = re.compile(r"^(\d{3})_(\d{3})$")
DFD_REAL = re.compile(r"^(\d{2})__")
DFD_FAKE = re.compile(r"^(\d{2})_(\d{2})__")


# ── naming / grouping ─────────────────────────────────────────────────────────

def source_name(folder, taken):
    """
    Short unique label for a fake folder. FaceForensics++ keeps every generator in a folder
    called ".../manipulated_sequences/<Generator>/c23/videos", so the last path part is useless;
    use the nearest meaningful parent ("Deepfakes") instead.
    """
    parts = [p for p in os.path.normpath(folder).replace("\\", "/").split("/") if p]
    name = next((p for p in reversed(parts) if p.lower() not in GENERIC_FOLDER_NAMES), "fake")
    unique, i = name, 2
    while unique in taken or unique == "real":
        unique, i = f"{name}{i}", i + 1
    taken.add(unique)
    return unique


def identity_keys(source, stem):
    """People/scenes a file shows, taken from its name. Unknown naming -> the file is its own group."""
    m = FF_FAKE.match(stem)
    if m:
        return ["ff:" + m.group(1), "ff:" + m.group(2)]
    m = FF_REAL.match(stem)
    if m:
        return ["ff:" + m.group(1)]
    m = DFD_FAKE.match(stem)
    if m:
        return ["dfd:" + m.group(1), "dfd:" + m.group(2)]
    m = DFD_REAL.match(stem)
    if m:
        return ["dfd:" + m.group(1)]
    return [f"file:{source}/{stem}"]


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def plan_splits(items, ratios, seed):
    """
    items: list of dicts with "source", "name", "label".
    Adds "identity" (group id) and "split" to every item. Whole groups go to one split.
    """
    uf = UnionFind()
    for it in items:
        keys = identity_keys(it["source"], it["name"])
        it["_keys"] = keys
        for k in keys[1:]:
            uf.union(keys[0], k)
    for it in items:
        it["identity"] = uf.find(it["_keys"][0])
        del it["_keys"]

    size = Counter(it["identity"] for it in items)
    groups = sorted(size)
    random.Random(seed).shuffle(groups)
    total, done, split_of = len(items), 0, {}
    for g in groups:
        frac = done / total
        split_of[g] = "train" if frac < ratios[0] else ("val" if frac < ratios[0] + ratios[1] else "test")
        done += size[g]
    for it in items:
        it["split"] = split_of[it["identity"]]
    return size


def list_media(folder, recursive):
    found = []
    if recursive:
        for dirpath, _, files in os.walk(folder):
            found += [os.path.join(dirpath, f) for f in files]
    else:
        found = [os.path.join(folder, f) for f in os.listdir(folder)]
    return sorted(p for p in found if p.lower().endswith(VIDEO_EXT + IMAGE_EXT) and os.path.isfile(p))


# ── frame extraction (runs in worker processes) ───────────────────────────────

def sample_video_frames(path, n_frames):
    """Yield (frame_index, BGR frame) for ~n_frames evenly spaced frames, reading sequentially."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # unknown length: take every 15th frame of the first n_frames * 15
        wanted = set(range(0, n_frames * 15, 15))
    else:
        wanted = set(np.linspace(0, max(total - 1, 0), min(n_frames, total)).astype(int).tolist())
    idx, last = 0, max(wanted)
    while idx <= last:
        if not cap.grab():
            break
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok:
                yield idx, frame
        idx += 1
    cap.release()


_cropper = None
_cfg = None


def _init_worker(landmarker_path, margin, cfg):
    global _cropper, _cfg
    cv2.setNumThreads(1)   # one decode thread per process; parallelism comes from the processes
    _cropper = FaceCropper(landmarker_path, margin=margin)
    _cfg = cfg


def process_item(it):
    """Crop faces from one video/image and save them. Returns (manifest rows, frames without a face)."""
    path, kind = it["path"], ("real" if it["label"] == 0 else "fake")
    crops, no_face = [], 0
    try:
        if path.lower().endswith(VIDEO_EXT):
            frames = ((idx, Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                      for idx, f in sample_video_frames(path, _cfg["frames_per_video"]))
        else:
            img = Image.open(path)
            img.load()
            frames = [(0, img)]
        for idx, img in frames:
            crop, found = _cropper.crop(img)
            if found:
                crops.append((f"f{idx:05d}", crop))
            else:
                no_face += 1
    except Exception as e:
        print(f"  [ERROR] {path}: {e}", flush=True)
        return [], no_face

    rows = []
    for tag, crop in crops:
        rel = f"{it['split']}/{kind}/{it['source']}__{it['name']}__{tag}.jpg"
        out_path = os.path.join(_cfg["out_dir"], *rel.split("/"))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        size = _cfg["size"]
        crop.convert("RGB").resize((size, size), Image.LANCZOS).save(out_path, "JPEG", quality=95)
        rows.append([rel, it["split"], it["label"], it["source"], it["name"], it["identity"], ""])
    return rows, no_face


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Prepare face-crop image dataset for the Swin detector")
    ap.add_argument("--real_dir",   required=True, help="folder with real videos and/or images")
    ap.add_argument("--fake_dirs",  required=True, nargs="+",
                    help="one or more folders of fake videos/images (one folder per generator)")
    ap.add_argument("--out_dir",    default="data/image_dataset")
    ap.add_argument("--frames_per_video", type=int, default=8)
    ap.add_argument("--size",       type=int, default=224, help="saved crop size (model input is 224)")
    ap.add_argument("--split",      type=float, nargs=3, default=(0.70, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--limit",      type=int, default=None, help="max videos/images from the real folder "
                                                                 "(and from each fake folder unless --fake_limit); "
                                                                 "a seeded random sample")
    ap.add_argument("--fake_limit", type=int, default=None,
                    help="max videos/images per fake folder; use it to keep real and fake totals balanced")
    ap.add_argument("--recursive",  action="store_true")
    ap.add_argument("--margin",     type=float, default=0.3, help="context around the face (fraction)")
    ap.add_argument("--workers",    type=int, default=1, help="parallel processes (e.g. 4 on Kaggle)")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--model",      default=None, help="face_landmarker.task (default: data/ or auto-download)")
    args = ap.parse_args()

    if abs(sum(args.split) - 1.0) > 1e-6:
        ap.error("--split ratios must add up to 1")
    fake_limit = args.fake_limit if args.fake_limit is not None else args.limit

    # 1. collect every file first, so the split can be planned across all folders at once
    items, taken = [], set()
    for kind, label, folder, limit in ([("real", 0, args.real_dir, args.limit)] +
                                       [("fake", 1, d, fake_limit) for d in args.fake_dirs]):
        source = "real" if kind == "real" else source_name(folder, taken)
        media = list_media(folder, args.recursive)
        if limit and limit < len(media):
            # a seeded random sample per folder (not the first N files): every generator then covers
            # different scenes, which gives the model more variety. Same seed -> same sample.
            media = sorted(random.Random(f"{args.seed}:{source}").sample(media, limit))
        if not media:
            print(f"[WARN] nothing found in {folder}")
        for p in media:
            items.append({"path": p, "label": label, "source": source,
                          "name": os.path.splitext(os.path.basename(p))[0]})
    if not any(it["label"] == 0 for it in items) or not any(it["label"] == 1 for it in items):
        sys.exit("Need both real and fake files.")

    group_size = plan_splits(items, args.split, args.seed)

    if any(DFD_FAKE.match(it["name"]) for it in items if it["label"] == 1) and \
            not any(DFD_REAL.match(it["name"]) for it in items if it["label"] == 0):
        print("[WARN] DeepFakeDetection-style fakes are included but none of their original videos are. "
              "The model can then learn 'these actors = fake' instead of learning manipulation. "
              "Consider leaving that folder out.")
    largest = max(group_size.values())
    print(f"{len(items)} files in {len(group_size)} person/scene groups (largest group: {largest} files)")
    if largest > 0.05 * len(items):
        print("[WARN] one group holds more than 5% of the files, so the split ratios will be uneven.")

    plan = Counter((it["split"], it["label"]) for it in items)
    print("Planned videos/images per split:")
    for s in SPLITS:
        print(f"  {s:<5} real={plan[(s, 0)]:>5}  fake={plan[(s, 1)]:>5}")

    # 2. resume support, refusing to mix in a dataset that was split differently
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    planned = {(it["source"], it["name"]): it["split"] for it in items}
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != MANIFEST_COLUMNS:
                sys.exit(f"{manifest_path} was made by an older version of this script with a different split. "
                         f"Delete {args.out_dir} and run again.")
            for r in reader:
                key = (r["source"], r["video"])
                if key in planned and planned[key] != r["split"]:
                    sys.exit(f"{args.out_dir} was made with different settings (split, seed, limits or folders). "
                             f"Delete it or use another --out_dir.")
                done.add(key)
        print(f"Resuming: {len(done)} videos/images already done")
    todo = [it for it in items if (it["source"], it["name"]) not in done]

    new_file = not os.path.exists(manifest_path)
    landmarker = args.model or resolve_model_file("face_landmarker.task", LANDMARKER_URL)
    cfg = {"out_dir": args.out_dir, "frames_per_video": args.frames_per_video, "size": args.size}

    written = Counter()            # (split, kind) -> images
    per_source = defaultdict(lambda: [0, 0, 0])   # source -> [files, images, frames without face]
    with open(manifest_path, "a", newline="") as mf:
        writer = csv.writer(mf)
        if new_file:
            writer.writerow(MANIFEST_COLUMNS)
        if args.workers > 1:
            pool = Pool(args.workers, initializer=_init_worker, initargs=(landmarker, args.margin, cfg))
            results = pool.imap(process_item, todo, chunksize=2)
        else:
            pool = None
            _init_worker(landmarker, args.margin, cfg)
            results = map(process_item, todo)
        try:
            for n, (it, (rows, no_face)) in enumerate(zip(todo, results), 1):
                writer.writerows(rows)
                mf.flush()
                kind = "real" if it["label"] == 0 else "fake"
                written[(it["split"], kind)] += len(rows)
                st = per_source[it["source"]]
                st[0] += 1; st[1] += len(rows); st[2] += no_face
                if n % 100 == 0:
                    print(f"  {n}/{len(todo)}", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            elif _cropper is not None:
                _cropper.close()

    print("\nDone. Images written this run:")
    for s in SPLITS:
        print(f"  {s:<5} real={written[(s, 'real')]:>6}  fake={written[(s, 'fake')]:>6}")
    print("\nPer folder:  files  images  frames-without-face")
    for src, (nf, ni, nn) in sorted(per_source.items()):
        print(f"  {src:<18} {nf:>5}  {ni:>6}  {nn:>6}")
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
