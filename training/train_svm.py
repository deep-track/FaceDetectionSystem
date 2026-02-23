import argparse
import pickle
import numpy as np
from collections import defaultdict

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# load data
def load_features(npz_path):
    data  = np.load(npz_path, allow_pickle=True)
    X     = data['X'].astype(np.float32)
    y     = data['y'].astype(np.int32)
    names = data['video_names']

    # Replace NaN / Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Loaded : {len(X)} segments, {X.shape[1]} features")
    print(f"  Real : {np.sum(y==0)}  |  Fake : {np.sum(y==1)}")
    return X, y, names

# aggregate feats to video-level
def video_level_predictions(probs, y_seg, names):
    """
    Average per-segment probabilities within each video.
    Threshold at 0.5 to get video-level prediction.

    Returns (video_preds, video_labels, video_ids)
    """
    vid_probs  = defaultdict(list)
    vid_labels = {}

    for prob, label, name in zip(probs, y_seg, names):
        vid_probs[name].append(prob)
        vid_labels[name] = label

    v_preds, v_labels, v_names = [], [], []
    for name, ps in vid_probs.items():
        avg_prob = np.mean(ps)
        v_preds.append(1 if avg_prob >= 0.5 else 0)
        v_labels.append(vid_labels[name])
        v_names.append(name)

    return np.array(v_preds), np.array(v_labels), np.array(v_names)

# train model
def train_svm(X_train, y_train):
    """
    RBF-kernel SVM with probability calibration (SVR in paper parlance).
    """
    print("\nTraining SVM (RBF kernel, probability=True) …")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale',
              probability=True, class_weight='balanced',
              random_state=42)
    svm.fit(X_train, y_train)
    print("  Done.")
    return svm

def evaluate(svm, scaler, X_test, y_test, names_test, label="Test"):
    X_s   = scaler.transform(X_test)
    probs = svm.predict_proba(X_s)[:, 1]    # P(fake)
    preds = (probs >= 0.5).astype(int)

    print(f"\n─── Segment-level [{label}] ───")
    print(f"  Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
    print(classification_report(y_test, preds,
                                target_names=['Real','Fake'], digits=4))
    print("Confusion matrix (rows=actual, cols=pred):")
    print(confusion_matrix(y_test, preds))

    # Video-level aggregation
    v_preds, v_labels, _ = video_level_predictions(probs, y_test, names_test)
    print(f"\n─── Video-level [{label}] ───")
    print(f"  Accuracy : {accuracy_score(v_labels, v_preds):.4f}")
    print(classification_report(v_labels, v_preds,
                                target_names=['Real','Fake'], digits=4))

    return accuracy_score(y_test, preds), accuracy_score(v_labels, v_preds)

def main():
    parser = argparse.ArgumentParser(description="FakeCatcher SVM Training")
    parser.add_argument('--features',   required=True,
                        help='.npz file from 1_signal_feature_extraction.py')
    parser.add_argument('--test_split', type=float, default=0.4,
                        help='Fraction for test set (default 0.4)')
    parser.add_argument('--output',     default='data/svm_model.pkl',
                        help='Where to save trained model + scaler')
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  FakeCatcher — SVM Training")
    print(f"  features={args.features} | test_split={args.test_split}")
    print(f"{'='*55}\n")

    X, y, names = load_features(args.features)

    # split take 60/40 split from paper

    X_train, X_test, y_train, y_test, n_train, n_test = train_test_split(
        X, y, names,
        test_size=args.test_split,
        stratify=y,
        random_state=42
    )
    print(f"\nTrain segments : {len(X_train)}")
    print(f"Test  segments : {len(X_test)}")

    # Scale
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Train
    svm = train_svm(X_train_s, y_train)

    # Evaluate on train
    evaluate(svm, scaler, X_train, y_train, n_train, label="Train")

    # Evaluate on test
    seg_acc, vid_acc = evaluate(svm, scaler, X_test, y_test, n_test, label="Test")

    print(f"\n{'='*55}")
    print(f"  Final Results")
    print(f"  Segment accuracy : {seg_acc:.4f}")
    print(f"  Video   accuracy : {vid_acc:.4f}")
    print(f"{'='*55}")

    # save
    bundle = {'svm': svm, 'scaler': scaler}
    with open(args.output, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\n✓ Model saved → {args.output}")

if __name__ == '__main__':
    main()