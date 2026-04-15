import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE_DIR)
sys.path.append(PARENT)

from preprocessing import preprocess_for_regression

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

    model = MLPRegressor(
        hidden_layer_sizes=(30, 30),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    train_losses = train_model(model, X_train, y_train, X_test, y_test, epochs=2000)

    preds = model.predict(X_test)
    eval_metrics = _metrics(y_test, preds)

    _plot_predictions(y_test, preds, os.path.join(plots, "scatter.png"))
    _plot_losses(train_losses, os.path.join(plots, "loss.png"))
    _save_comparison_table(y_test, preds, os.path.join(plots, "comparison_table.csv"))

    print("#" * 40)
    print("Library MLP (sklearn)")
    print(f"MAE: {eval_metrics['mae']:.3f}")
    print(f"RMSE: {eval_metrics['rmse']:.3f}")

def train_model(model, X_train, y_train, X_val, y_val, epochs=2000):
    train_losses = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for epoch in range(epochs):
            model.fit(X_train, y_train)
            preds_train = model.predict(X_train)
            mse_epoch = np.mean((y_train - preds_train) ** 2)
            train_losses.append(mse_epoch)
            if (epoch + 1) % 100 == 0:
                mae_train = float(np.mean(np.abs(y_train - preds_train)))
                preds_val = model.predict(X_val)
                mae_val = float(np.mean(np.abs(y_val - preds_val)))
                print(f"Epoch {epoch+1}: train MAE={mae_train:.3f}, test MAE={mae_val:.3f}")
    return train_losses

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true
    y_pred = y_pred
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}

def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
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

def _plot_losses(losses, path: str) -> None:
    if not losses:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses)+1), losses, label="Train MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training process by epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _save_comparison_table(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    df = pd.DataFrame({
        "Index": range(len(y_true)),
        "True G3": y_true,
        "Predicted G3": y_pred,
        "Difference": y_true - y_pred
    })
    df.to_csv(path, index=False)
    print(f"Comparison table saved to {path}")

if __name__ == "__main__":
    main()

