import os
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy
import pandas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE_DIR)
sys.path.append(PARENT)

from preprocessing import preprocess_for_regression


class MLPRegressor:
    def __init__(self, layer_sizes: List[int], lr: float = 0.01, seed: int = 42):
        self.layer_sizes = layer_sizes
        self.lr = lr
        rng = numpy.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            w = rng.normal(0, 1 / numpy.sqrt(n_in), size=(n_in, n_out))
            b = numpy.zeros((1, n_out))
            self.weights.append(w)
            self.biases.append(b)


    def _relu(x: numpy.ndarray) -> numpy.ndarray:
        return numpy.maximum(0, x)

    def _relu_grad(x: numpy.ndarray) -> numpy.ndarray:
        return (x > 0).astype(float)

    def _forward(self, X: numpy.ndarray):
        activations = [X]
        zs = []
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            zs.append(z)
            if idx == len(self.weights) - 1:
                a = z
            else:
                a = self._relu(z)
            activations.append(a)
        return activations, zs

    def fit(self, X: numpy.ndarray, y: numpy.ndarray, epochs: int = 200, batch_size: int = 32):
        n_samples = X.shape[0]
        losses = []
        for epoch in range(epochs):
            indices = numpy.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices].reshape(-1, 1)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                activations, zs = self._forward(xb)
                preds = activations[-1]
                error = preds - yb

                # Backpropagation
                delta = 2 * error / len(xb)
                grads_w = []
                grads_b = []
                for i in reversed(range(len(self.weights))):
                    a_prev = activations[i]
                    dw = a_prev.T @ delta
                    db = numpy.sum(delta, axis=0, keepdims=True)
                    grads_w.insert(0, dw)
                    grads_b.insert(0, db)
                    if i != 0:
                        delta = (delta @ self.weights[i].T) * self._relu_grad(zs[i - 1])

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * grads_w[i]
                    self.biases[i] -= self.lr * grads_b[i]

            # Track epoch loss on full data for smoother curves.
            preds_epoch = self.predict(X)
            epoch_loss = numpy.mean((preds_epoch - y.reshape(-1, 1)) ** 2)
            losses.append(epoch_loss)
        return losses

    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        activations, _ = self._forward(X)
        return activations[-1]


def main():
    data_path = os.path.join(PARENT, "students", "student-mat.csv")
    plots = os.path.join(BASE_DIR, "plots")

    prep = preprocess_for_regression(
        data_path=data_path,
        plot_dir=plots,
        top_k=30,
        corr_threshold=0.9,
    )

    X_train = prep["X_train"]
    y_train = prep["y_train"]
    X_test = prep["X_test"]
    y_test = prep["y_test"]

    input_dim = X_train.shape[1]
    model = MLPRegressor(layer_sizes=[input_dim, 30, 30, 1], lr=0.01)

    start = time.time()
    losses = train_model(model, X_train, y_train, X_test, y_test, epochs=2000, batch_size=32, log_interval=100)
    elapsed = time.time() - start

    preds = model.predict(X_test).reshape(-1)
    eval_metrics = _metrics(y_test, preds)

    _plot_losses(losses, os.path.join(plots, "loss.png"))
    _plot_predictions(y_test, preds, os.path.join(plots, "scatter.png"))
    _save_comparison_table(y_test, preds, os.path.join(plots, "comparison_table.csv"))

    print("#" * 40)
    print("Custom MLP")
    print(f"Epochs: {len(losses)}, Time: {elapsed:.2f}s")
    print(f"MAE: {eval_metrics['mae']:.3f}")
    print(f"RMSE: {eval_metrics['rmse']:.3f}")
    print(f"R2: {eval_metrics['r2']:.3f}")


def _metrics(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> dict:
    y_true = y_true
    y_pred = y_pred
    mae = float(numpy.mean(numpy.abs(y_true - y_pred)))
    rmse = float(numpy.sqrt(numpy.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def _plot_losses(losses: List[float], path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses)+1), losses, label="Train MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training process by epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_predictions(y_true: numpy.ndarray, y_pred: numpy.ndarray, path: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("True G3")
    plt.ylabel("Predicted G3")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="ideal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_model(model, X_train, y_train, X_val, y_val, epochs=2000, batch_size=32, log_interval=100):
    losses = []
    for epoch in range(epochs):
        model.fit(X_train, y_train, epochs=1, batch_size=batch_size)
        preds_train = model.predict(X_train).reshape(-1)
        mse_epoch = float(numpy.mean((y_train - preds_train) ** 2))
        losses.append(mse_epoch)
        if (epoch + 1) % log_interval == 0:
            mae_train = float(numpy.mean(numpy.abs(y_train - preds_train)))
            preds_val = model.predict(X_val).reshape(-1)
            mae_val = float(numpy.mean(numpy.abs(y_val - preds_val)))
            print(f"Epoch {epoch+1}: train MAE={mae_train:.3f}, test MAE={mae_val:.3f}")
    return losses


def _save_comparison_table(y_true: numpy.ndarray, y_pred: numpy.ndarray, path: str) -> None:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    df = pandas.DataFrame({
        "Index": range(len(y_true)),
        "True G3": y_true,
        "Predicted G3": y_pred,
        "Difference": y_true - y_pred
    })
    df.to_csv(path, index=False)
    print(f"Comparison table saved to {path}")

if __name__ == "__main__":
    main()

