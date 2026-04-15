import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import preprocess_for_rnn

from rnn import CustomRNN
from gru import CustomGRU
from lstm import CustomLSTM

def train_and_eval_custom(model, X_train, y_train, X_test, y_test, epochs=20, max_samples=None):
    # Если max_samples не указан, используем все данные для обучения
    if max_samples is None:
        X_tr = X_train
        y_tr = y_train
    else:
        X_tr = X_train[:max_samples]
        y_tr = y_train[:max_samples]

    # Обучение
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in zip(X_tr, y_tr):
            epoch_loss += model.train_step(x, y, lr=0.01)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(X_tr):.4f}")

    # Тестирование на всех тестовых данных
    preds = []
    for x in X_test:
        preds.append(model.forward(x))

    preds = np.array(preds)
    
    # Вычисление метрик
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Custom Model - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return preds
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "steel_industry", "Steel_industry_data.csv")
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

    seq_length = 10
    hidden_size = 16
    epochs = 10
    learning_rate = 0.005
    
    data = preprocess_for_rnn(data_path, seq_length=seq_length, test_size=0.2, plot_dir=plots_dir)
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    
    print("\n--- Training Custom RNN (hidden_size=32, seq_length=10, epochs=20) ---")
    rnn_model = CustomRNN(input_size=X_train.shape[2], hidden_size=hidden_size)
    preds_rnn = train_and_eval_custom(rnn_model, X_train, y_train, X_test, y_test, 
                                     epochs=epochs)
    
    print("\n--- Training Custom GRU (hidden_size=32, seq_length=10, epochs=20) ---")
    gru_model = CustomGRU(input_size=X_train.shape[2], hidden_size=hidden_size)
    preds_gru = train_and_eval_custom(gru_model, X_train, y_train, X_test, y_test, 
                                     epochs=epochs)

    print("\n--- Training Custom LSTM (hidden_size=32, seq_length=10, epochs=20) ---")
    lstm_model = CustomLSTM(input_size=X_train.shape[2], hidden_size=hidden_size)
    preds_lstm = train_and_eval_custom(lstm_model, X_train, y_train, X_test, y_test, 
                                      epochs=epochs)

    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(16, 7), dpi=150)

    plt.plot(y_test[:200], label="True", color="black", linestyle="--", linewidth=2)
    plt.plot(preds_rnn[:200], label="RNN", linewidth=1.5)
    plt.plot(preds_gru[:200], label="GRU", linewidth=1.5)
    plt.plot(preds_lstm[:200], label="LSTM", linewidth=1.5)

    plt.xlabel("Time step")
    plt.ylabel("Energy consumption (kWh)")
    plt.title("Energy Consumption Prediction: RNN vs GRU vs LSTM", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "predictions.png"), dpi=300)
