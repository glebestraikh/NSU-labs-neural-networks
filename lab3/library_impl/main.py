import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import preprocess_for_rnn

from rnn import RNNNet
from gru import GRUNet
from lstm import LSTMNet

def train_and_eval(model_type, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training {model_type} ---")
    
    input_size = X_train.shape[2]
    hidden_size = 32
    num_layers = 1
    
    if model_type == "RNN":
        model = RNNNet(input_size, hidden_size, num_layers).to(device)
    elif model_type == "GRU":
        model = GRUNet(input_size, hidden_size, num_layers).to(device)
    elif model_type == "LSTM":
        model = LSTMNet(input_size, hidden_size, num_layers).to(device)
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # объединяет X_train и y_train в пары
    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    # Разбивает данные на батчи
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            # Перенос на устройство
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # модель делает прогноз
            outputs = model(batch_X).squeeze()
            # сравниваем прогноз и правильный ответ
            loss = criterion(outputs, batch_y)

            # очищаем старые градиенты
            optimizer.zero_grad()
            # считаем ошибки
            loss.backward()
            # обновляем веса
            optimizer.step()

            # Накопление ошибки эпохи
            # берётся ошибка батча
            # умножается на размер батча
            # складывается в общую ошибку эпохи
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    # переводит модель в режим тестирования
    model.eval()
    with torch.no_grad():
        test_X = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(test_X).squeeze().cpu().numpy()
        
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"{model_type} Test MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return preds

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "steel_industry", "Steel_industry_data.csv")
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    
    data = preprocess_for_rnn(data_path, seq_length=10, test_size=0.2, plot_dir=plots_dir)

    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    
    preds_rnn = train_and_eval("RNN", X_train, y_train, X_test, y_test, epochs=10)
    preds_gru = train_and_eval("GRU", X_train, y_train, X_test, y_test, epochs=10)
    preds_lstm = train_and_eval("LSTM", X_train, y_train, X_test, y_test, epochs=10)

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
    plt.savefig(os.path.join(plots_dir, "predictions.png"), dpi=300)
