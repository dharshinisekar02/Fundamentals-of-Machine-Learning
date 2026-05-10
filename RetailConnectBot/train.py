import numpy as np
import json
import time
from model import MLP, cross_entropy_loss

# ─────────────────────────────────────────
# STEP 4: Training
# ─────────────────────────────────────────

def one_hot(y, num_classes):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def get_batches(X, y, batch_size, shuffle=True):
    n   = X.shape[0]
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def train(epochs=200, batch_size=64, lr=0.01, dropout_rate=0.3):

    # 1. Load data
    X_train = np.load("X_train.npy")
    X_test  = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test  = np.load("y_test.npy")

    num_classes = len(np.unique(y_train))
    input_size  = X_train.shape[1]

    print(f"Training samples : {X_train.shape[0]}")
    print(f"Test samples     : {X_test.shape[0]}")
    print(f"Input features   : {input_size}")
    print(f"Classes          : {num_classes}")
    print(f"Epochs           : {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print("-" * 55)

    # 2. Init model
    model = MLP(
        input_size=input_size,
        hidden1=128, hidden2=64,
        output_size=num_classes,
        lr=lr, dropout_rate=dropout_rate
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    start   = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches  = 0

        # Mini-batch training
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            y_batch_oh = one_hot(y_batch, num_classes)
            probs      = model.forward(X_batch, training=True)
            total_loss += cross_entropy_loss(probs, y_batch_oh)
            model.backward(y_batch_oh)
            n_batches  += 1

        avg_loss = total_loss / n_batches

        # Evaluate on FULL train + test set (no dropout)
        train_acc = accuracy(model.predict(X_train), y_train)
        val_acc   = accuracy(model.predict(X_test),  y_test)

        history["train_loss"].append(round(avg_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save("model_weights.npz")

        if epoch % 20 == 0 or epoch == 1:
            saved = " ← best saved" if val_acc == best_val_acc else ""
            print(f"Epoch {epoch:>3}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train: {train_acc*100:.1f}% | "
                  f"Val: {val_acc*100:.1f}%{saved}")

    elapsed = time.time() - start
    print("-" * 55)
    print(f"Done in {elapsed:.1f}s  |  Best Val Accuracy: {best_val_acc*100:.2f}%")

    with open("train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("History saved -> train_history.json")

    return model, history


# ─────────────────────────────────────────
# Run: python train.py
# ─────────────────────────────────────────
if __name__ == "__main__":
    train(epochs=200, batch_size=64, lr=0.01, dropout_rate=0.3)
