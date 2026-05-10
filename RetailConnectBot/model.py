import numpy as np
import os

# ─────────────────────────────────────────
# STEP 3: MLP Model — Built From Scratch
# No PyTorch, No TensorFlow, No Sklearn MLP
# Pure NumPy only
# ─────────────────────────────────────────
#
# Architecture:
#
#  Input (vocab size)
#      ↓
#  Dense Layer 1 (128 neurons) + ReLU
#      ↓
#  Dropout (rate=0.3)
#      ↓
#  Dense Layer 2 (64 neurons) + ReLU
#      ↓
#  Dropout (rate=0.3)
#      ↓
#  Output Layer (9 neurons) + Softmax
#      ↓
#  Predicted Intent (0–8)
# ─────────────────────────────────────────


# ── Activation Functions ──────────────────

def relu(x):
    """ReLU: max(0, x) — kills negative values, keeps positive"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Gradient of ReLU: 1 if x > 0, else 0"""
    return (x > 0).astype(float)

def softmax(x):
    """
    Softmax: converts raw scores to probabilities (sum = 1)
    Numerically stable: subtract max before exp
    """
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Loss Function ─────────────────────────

def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy: measures how wrong predictions are.
    Lower = better.
    """
    n      = y_true.shape[0]
    clipped = np.clip(y_pred, 1e-9, 1.0)
    return -np.sum(y_true * np.log(clipped)) / n


# ── Weight Initialization ─────────────────

def init_weights(input_size, hidden1, hidden2, output_size, seed=42):
    """
    He initialization: designed for ReLU.
    Prevents gradients from vanishing or exploding.
    """
    np.random.seed(seed)
    W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden1))
    W2 = np.random.randn(hidden1, hidden2)    * np.sqrt(2.0 / hidden1)
    b2 = np.zeros((1, hidden2))
    W3 = np.random.randn(hidden2, output_size) * np.sqrt(2.0 / hidden2)
    b3 = np.zeros((1, output_size))
    return W1, b1, W2, b2, W3, b3


# ── MLP Class ─────────────────────────────

class MLP:
    def __init__(self, input_size, hidden1=128, hidden2=64,
                 output_size=9, lr=0.01, dropout_rate=0.3):
        self.lr           = lr
        self.dropout_rate = dropout_rate
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = \
            init_weights(input_size, hidden1, hidden2, output_size)
        self.cache = {}

    # ── Forward Pass ──────────────────────
    def forward(self, X, training=True):
        """
        training=True  -> apply dropout (use during training)
        training=False -> no dropout    (use during prediction)
        """
        # Layer 1
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        if training:
            mask1 = (np.random.rand(*a1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            a1 = a1 * mask1
        else:
            mask1 = np.ones_like(a1)

        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        if training:
            mask2 = (np.random.rand(*a2.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            a2 = a2 * mask2
        else:
            mask2 = np.ones_like(a2)

        # Output
        z3     = a2 @ self.W3 + self.b3
        output = softmax(z3)

        self.cache = {
            'X': X, 'z1': z1, 'a1': a1, 'mask1': mask1,
            'z2': z2, 'a2': a2, 'mask2': mask2,
            'z3': z3, 'output': output
        }
        return output

    # ── Backward Pass ─────────────────────
    def backward(self, y_true_onehot):
        n = y_true_onehot.shape[0]
        c = self.cache

        dz3 = (c['output'] - y_true_onehot) / n
        dW3 = c['a2'].T @ dz3
        db3 = dz3.sum(axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T * c['mask2']
        dz2 = da2 * relu_derivative(c['z2'])
        dW2 = c['a1'].T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T * c['mask1']
        dz1 = da1 * relu_derivative(c['z1'])
        dW1 = c['X'].T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    # ── Predict ───────────────────────────
    def predict(self, X):
        """Returns predicted class index for each sample"""
        probs = self.forward(X, training=False)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """Returns probability of each class"""
        return self.forward(X, training=False)

    # ── Save / Load ───────────────────────
    def save(self, path="model_weights.npz"):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        print(f"Model saved -> {path}")

    def load(self, path="model_weights.npz"):
        data = np.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
        self.W3, self.b3 = data['W3'], data['b3']
        print(f"Model loaded <- {path}")


# ─────────────────────────────────────────
# Run sanity check: python model.py
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Load X_train.npy generated by vectorizer.py (same folder)
    X_train = np.load("X_train.npy")

    input_size  = X_train.shape[1]  # depends on vocab size
    output_size = 9                  # number of intents

    model = MLP(
        input_size=input_size,
        hidden1=128,
        hidden2=64,
        output_size=output_size,
        lr=0.01,
        dropout_rate=0.3
    )

    # Quick forward pass sanity check
    sample = X_train[:1]
    probs  = model.forward(sample, training=False)

    print("=== Model Architecture ===")
    print(f"  Input    : {input_size} features")
    print(f"  Hidden 1 : 128 neurons  (ReLU + Dropout 0.3)")
    print(f"  Hidden 2 : 64  neurons  (ReLU + Dropout 0.3)")
    print(f"  Output   : {output_size} neurons (Softmax)")
    print(f"\n=== Parameter Count ===")
    print(f"  W1 : {model.W1.shape} = {model.W1.size:,} params")
    print(f"  W2 : {model.W2.shape} = {model.W2.size:,} params")
    print(f"  W3 : {model.W3.shape} = {model.W3.size:,} params")
    print(f"  TOTAL : {model.W1.size + model.W2.size + model.W3.size:,} params")
    print(f"\n=== Sanity Check ===")
    print(f"  Output probs sum : {probs.sum():.4f}  (must be 1.0)")
    print(f"  Predicted class  : {np.argmax(probs)}")
    print("\nmodel.py is working correctly!")
