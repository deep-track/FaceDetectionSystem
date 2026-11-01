"""
Video-level train/test splitting shared by train_cnn.py and train_svm.py.

Every video is cut into 50%-overlapping segments, so segments of one video are
near-duplicates of each other. Splitting at the segment level therefore puts
almost the same data in both train and test and inflates every metric.
These helpers split by video instead: all segments of a video land on the same side.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def group_split_indices(groups, test_size, seed=42):
    """Return (train_idx, test_idx) such that no group (video) appears in both."""
    groups = np.asarray(groups)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(groups)), groups=groups))
    return train_idx, test_idx


def describe_split(name, y, groups):
    """One-line summary: segments, videos and class balance."""
    y = np.asarray(y)
    print(f"  {name:<5}: {len(y):>6} segments | {len(set(groups)):>5} videos | "
          f"real={int((y == 0).sum())} fake={int((y == 1).sum())}")
