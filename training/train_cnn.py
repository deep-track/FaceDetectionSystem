import argparse
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load data
def load_maps(npz_path):
    data  = np.load(npz_path, allow_pickle=True)
    X     = data['X'].astype(np.float32)    # (N, omega, 64, 1)
    y     = data['y'].astype(np.int32)
    names = data['video_names']

    # Data is already normalised to [0, 1] by the extraction script
    # Verify range and warn if unexpected
    print(f"Loaded : {len(X)} maps, shape {X.shape}")
    print(f"  Real : {np.sum(y==0)}  |  Fake : {np.sum(y==1)}")
    print(f"  Value range: [{X.min():.3f}, {X.max():.3f}]  mean={X.mean():.3f}")
    if X.max() > 1.5:
        print("  [WARN] Values exceed 1.0 — normalising by 255")
        X = X / 255.0
    return X, y, names

# cnn architecture
def build_cnn(input_shape):
    """
    Three convolutional blocks with BatchNormalization.
    GlobalAveragePooling2D to keep parameter count low (~130K).
    LR=3e-4: middle ground between 1e-4 (too slow) and 1e-3 (too fast).
    L2 regularization on conv layers to penalise large weights.
    """
    reg = keras.regularizers.l2(1e-4)

    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                      kernel_regularizer=reg, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # Block 3
        layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Global average pooling
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ], name='FakeCatcher_CNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),  # between 1e-4 and 1e-3
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model

# video level aggregation
def video_level_predictions(probs, y_seg, names):
    """Average P(fake) across all segments of a video; threshold at 0.5."""
    vid_probs  = defaultdict(list)
    vid_labels = {}

    for prob, label, name in zip(probs, y_seg, names):
        vid_probs[name].append(float(prob))
        vid_labels[name] = int(label)

    v_preds, v_labels, v_names = [], [], []
    for name, ps in vid_probs.items():
        avg = np.mean(ps)
        v_preds.append(1 if avg >= 0.5 else 0)
        v_labels.append(vid_labels[name])
        v_names.append(name)

    return np.array(v_preds), np.array(v_labels), np.array(v_names)

def evaluate(model, X_test, y_test, names_test, label="Test"):
    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    print(f"\n─── Segment-level [{label}] ───")
    seg_acc = accuracy_score(y_test, preds)
    print(f"  Accuracy : {seg_acc:.4f}")
    try:
        print(f"  ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
    except Exception:
        pass
    print(classification_report(y_test, preds,
                                target_names=['Real', 'Fake'], digits=4))
    print("Confusion matrix (rows=actual, cols=pred):")
    print(confusion_matrix(y_test, preds))

    # Video-level
    v_preds, v_labels, _ = video_level_predictions(probs, y_test, names_test)
    vid_acc = accuracy_score(v_labels, v_preds)
    print(f"\n─── Video-level [{label}] ───")
    print(f"  Accuracy : {vid_acc:.4f}")
    print(classification_report(v_labels, v_preds,
                                target_names=['Real', 'Fake'], digits=4))
    return seg_acc, vid_acc

# plot training curve history
def plot_history(history, save_path='training_history.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, metric, title in zip(
        axes,
        ['loss', 'accuracy', 'auc'],
        ['Loss', 'Accuracy', 'AUC']
    ):
        ax.plot(history.history[metric],          label='Train')
        ax.plot(history.history[f'val_{metric}'], label='Val',  linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"✓ Training curves saved → {save_path}")

def main():
    parser = argparse.ArgumentParser(description="FakeCatcher CNN Training")
    parser.add_argument('--maps',       required=True,
                        help='.npz file from PPG map extraction')
    parser.add_argument('--output',     default='cnn_model.keras',
                        help='Saved Keras model path')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch',      type=int,   default=32)
    parser.add_argument('--test_split', type=float, default=0.4)
    parser.add_argument('--val_split',  type=float, default=0.15,
                        help='Fraction of TRAIN set used for validation')
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  FakeCatcher — CNN Training")
    print(f"  maps={args.maps} | epochs={args.epochs} | batch={args.batch}")
    print(f"{'='*55}\n")

    # load
    X, y, names = load_maps(args.maps)

    # train-test split, 60/40 as stipulated in the paper
    X_train, X_test, y_train, y_test, n_train, n_test = train_test_split(
        X, y, names,
        test_size=args.test_split,
        stratify=y,
        random_state=42
    )
    print(f"\nTrain segments : {len(X_train)}")
    print(f"Test  segments : {len(X_test)}")

    # build model
    input_shape = X_train.shape[1:]    # (omega, 64, 1)
    model = build_cnn(input_shape)
    model.summary()

    # callbacks
    # IMPORTANT: all three callbacks monitor val_loss so they agree on what "best" means.
    # Previously ModelCheckpoint monitored val_accuracy while EarlyStopping monitored
    # val_loss — they saved different epochs. val_accuracy can be misleadingly high
    # early on (model predicts everything REAL, gets ~50% accuracy from majority class)
    # while val_loss correctly shows the model hasn't learned anything useful yet.
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=7,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=2, verbose=1, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(args.output, monitor='val_loss',
                                        save_best_only=True, mode='min', verbose=1),
    ]

    # class weights (handle class imbalance like Celeb-DF)
    n_real = int(np.sum(y_train == 0))
    n_fake = int(np.sum(y_train == 1))
    total  = n_real + n_fake
    class_weight = {0: total / (2 * n_real), 1: total / (2 * n_fake)}
    print(f"\nClass weights: real={class_weight[0]:.3f}, fake={class_weight[1]:.3f}")

    # train
    print("\nTraining …")
    history = model.fit(
        X_train, y_train,
        epochs          = args.epochs,
        batch_size      = args.batch,
        validation_split= args.val_split,
        class_weight    = class_weight,
        callbacks       = callbacks,
        verbose         = 1
    )

    # plot
    plot_history(history)

    # load best checkpoint for evaluation
    best_model = keras.models.load_model(args.output)

    # evaluate
    evaluate(best_model, X_train, y_train, n_train, label="Train")
    seg_acc, vid_acc = evaluate(best_model, X_test,  y_test,  n_test,  label="Test")

    print(f"\n{'='*55}")
    print(f"  Final Results")
    print(f"  Segment accuracy : {seg_acc:.4f}")
    print(f"  Video   accuracy : {vid_acc:.4f}")
    print(f"{'='*55}")
    print(f"\n✓ Best model saved → {args.output}")

if __name__ == '__main__':
    main()