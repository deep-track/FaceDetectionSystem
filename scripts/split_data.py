import os
import argparse
import numpy as np
from collections import defaultdict


def split_npz(path, is_maps, out_dir, seed, label):
    print(f"\n{'─'*50}")
    print(f"  Loading {path} ...")
    data   = np.load(path, allow_pickle=True)
    X      = data['X'].astype(np.float32)
    y      = data['y'].astype(np.int32)
    names  = list(data['video_names'])
    rng    = np.random.default_rng(seed)

    print(f"  Total segments : {len(X)}")
    print(f"  Real           : {int((y==0).sum())}")
    print(f"  Fake           : {int((y==1).sum())}")

    # ── identify generators from name prefix ─────────────────────────────
    # names look like "FaceSwap/000.mp4" or "real/abc.mp4"
    # for older files without a prefix we fall back to the label
    generators = defaultdict(list)   # generator_name → [indices]
    real_indices = []

    for i, name in enumerate(names):
        name = str(name)
        if '/' in name:
            prefix = name.split('/')[0]
        elif '\\' in name:
            prefix = name.split('\\')[0]
        else:
            # no prefix saved — use label as fallback
            prefix = 'real' if y[i] == 0 else 'unknown_fake'

        if y[i] == 0:
            real_indices.append(i)
        else:
            generators[prefix].append(i)

    real_indices = np.array(real_indices)
    print(f"\n  Real segments  : {len(real_indices)}")
    print(f"  Fake generators found:")
    for g, idxs in sorted(generators.items()):
        print(f"    {g:30s} {len(idxs):>6} segments")

    if not generators:
        print("  [WARN] No fake generator prefixes found in video_names.")
        print("         Re-run extraction with the updated script to get prefixes.")
        return []

    # ── per-generator balanced split ─────────────────────────────────────
    saved = []
    for gen_name, fake_idx in sorted(generators.items()):
        fake_idx = np.array(fake_idx)
        n        = min(len(real_indices), len(fake_idx))   # balance to minority

        # undersample whichever side is larger
        real_sampled = rng.choice(real_indices, size=n, replace=False)
        fake_sampled = rng.choice(fake_idx,     size=n, replace=False)

        keep = np.concatenate([real_sampled, fake_sampled])
        perm = rng.permutation(len(keep))
        keep = keep[perm]

        X_out = X[keep]
        y_out = y[keep]
        n_out = np.array(names)[keep]

        out_name = f"{label}_{gen_name}.npz"
        out_path = os.path.join(out_dir, out_name)
        np.savez_compressed(out_path, X=X_out, y=y_out, video_names=n_out)

        print(f"\n  [{gen_name}]")
        print(f"    Real : {int((y_out==0).sum()):>5}  |  Fake : {int((y_out==1).sum()):>5}  "
              f"|  Total : {len(y_out):>5}")
        print(f"    → {out_path}")
        saved.append((gen_name, out_path))

    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps',     default='data/final_ppg_maps.npz')
    ap.add_argument('--features', default='data/final_features.npz')
    ap.add_argument('--out_dir',  default='data/split')
    ap.add_argument('--seed',     type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    map_pairs  = split_npz(args.maps,     is_maps=True,  out_dir=args.out_dir,
                            seed=args.seed, label='maps')
    feat_pairs = split_npz(args.features, is_maps=False, out_dir=args.out_dir,
                            seed=args.seed, label='features')

    print(f"\n{'='*50}")
    print("  All done. Files saved:")
    for _, p in map_pairs + feat_pairs:
        print(f"    {p}")

    if map_pairs:
        print(f"\n  Next steps — train one model per generator:")
        for gen, path in map_pairs:
            print(f"    python 4_train_cnn.py --data {path} "
                  f"--out data/model_{gen}.keras")
        for gen, path in feat_pairs:
            print(f"    python 3_train_svm.py --data {path} "
                  f"--out data/svm_{gen}.pkl")


if __name__ == '__main__':
    main()