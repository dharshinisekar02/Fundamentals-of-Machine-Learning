import numpy as np
import json

# ─────────────────────────────────────────
# STEP 5: Evaluation
# ─────────────────────────────────────────
# Metrics we calculate (all from scratch):
#  1. Overall Accuracy
#  2. Per-class Precision, Recall, F1
#  3. Confusion Matrix
#  4. Training curve summary
# ─────────────────────────────────────────

from model import MLP


def precision_recall_f1(y_true, y_pred, num_classes):
    """Calculate per-class Precision, Recall, F1 without sklearn"""
    results = {}
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        results[c] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   int(np.sum(y_true == c))
        }
    return results


def confusion_matrix(y_true, y_pred, num_classes):
    """Build confusion matrix from scratch"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm


def print_confusion_matrix(cm, labels):
    """Pretty print confusion matrix"""
    short = [l[:8] for l in labels]   # shorten label names
    col_w = 10
    header = " " * 12 + "".join(f"{s:>{col_w}}" for s in short)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{short[i]:<12}" + "".join(f"{v:>{col_w}}" for v in row)
        print(row_str)


def evaluate():
    # 1. Load data & model
    X_test  = np.load("X_test.npy")
    y_test  = np.load("y_test.npy")
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")

    with open("label_map.json") as f:
        label_map = json.load(f)                        # {"0": "availability_check", ...}

    num_classes = len(label_map)
    labels      = [label_map[str(i)] for i in range(num_classes)]

    model = MLP(X_test.shape[1], 128, 64, num_classes, lr=0.01)
    model.load("model_weights.npz")

    # 2. Predictions
    y_pred_test  = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # 3. Accuracy
    train_acc = np.mean(y_pred_train == y_train)
    test_acc  = np.mean(y_pred_test  == y_test)

    print("=" * 55)
    print("            MODEL EVALUATION REPORT")
    print("=" * 55)
    print(f"  Train Accuracy : {train_acc*100:.2f}%")
    print(f"  Test  Accuracy : {test_acc*100:.2f}%")

    # 4. Per-class metrics
    metrics = precision_recall_f1(y_test, y_pred_test, num_classes)

    print("\n  Per-Class Metrics (on Test Set):")
    print(f"  {'Intent':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("  " + "-" * 60)

    total_support = 0
    weighted_f1   = 0

    for i, label in enumerate(labels):
        m = metrics[i]
        print(f"  {label:<22} {m['precision']:>10.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['support']:>9}")
        weighted_f1   += m['f1'] * m['support']
        total_support += m['support']

    macro_f1    = np.mean([metrics[i]['f1'] for i in range(num_classes)])
    weighted_f1 = weighted_f1 / total_support

    print("  " + "-" * 60)
    print(f"  {'Macro F1':<22} {macro_f1:>27.3f}")
    print(f"  {'Weighted F1':<22} {weighted_f1:>27.3f}")

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test, num_classes)
    print("\n  Confusion Matrix (rows=Actual, cols=Predicted):")
    print()
    print_confusion_matrix(cm, labels)

    # 6. Training curve summary
    with open("train_history.json") as f:
        history = json.load(f)

    print("\n  Training Curve Summary:")
    epochs     = len(history["val_acc"])
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_val   = max(history["val_acc"])
    print(f"  Total epochs   : {epochs}")
    print(f"  Best Val Acc   : {best_val*100:.2f}%  (epoch {best_epoch})")
    print(f"  Final Loss     : {history['train_loss'][-1]}")
    print("=" * 55)


# ─────────────────────────────────────────
# Run: python evaluate.py
# ─────────────────────────────────────────
if __name__ == "__main__":
    evaluate()
