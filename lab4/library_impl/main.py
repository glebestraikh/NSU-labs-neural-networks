import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import get_data
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from metrics import accuracy, macro_avg
from plotting import save_all_plots


def build_lenet5():
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Tanh(),
        nn.Linear(120, 84),
        nn.Tanh(),
        nn.Linear(84, 10),
    )


def make_loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, criterion, device):
    # Отключает режим обучения
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_all = []
    with torch.inference_mode():
        # Берём батчи
        for data, target in loader:
            # перенос на устройство
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)
            preds_all.append(preds.cpu().numpy())
    return total_loss / total, correct / total, np.concatenate(preds_all)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MNIST-master')
    x_train, y_train, x_test, y_test = get_data(data_dir)

    x_train, y_train = x_train[:30000], y_train[:30000]
    x_test, y_test = x_test[:5000], y_test[:5000]

    train_loader = make_loader(x_train, y_train, batch_size=32, shuffle=True)
    test_loader = make_loader(x_test, y_test, batch_size=128, shuffle=False)
    train_eval_loader = make_loader(x_train, y_train, batch_size=128, shuffle=False)

    model = build_lenet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 5
    history = {
        # loss на каждом батче
        'step_losses': [],

        # средний loss на train после эпохи
        'train_losses': [],
        # точность на train после эпохи
        'train_accs': [],
        # ошибка на тестовых данных после эпохи
        'test_losses': [],
        # точность на тесте после эпохи
        'test_accs': [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss, seen = 0.0, 0
        # tqdm для показа прогресса
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, target = data.to(device), target.to(device)
            # обнуление градиентов
            optimizer.zero_grad()
            # прогон модели
            output = model(data)

            # считаем ошибку
            loss = criterion(output, target)

            # backpropagation
            # куда и как поправить веса, чтобы ошибка уменьшилась
            loss.backward()

            # обновление весов
            optimizer.step()

            # Записывает ошибку каждого батча.
            history['step_losses'].append(loss.item())

            # накопление loss эпохи
            running_loss += loss.item() * data.size(0)

            #  считаем количество обработанных примеров для корректного усреднения
            seen += data.size(0)

        eval_train_loss, eval_train_acc, _ = evaluate(model, train_eval_loader, criterion, device)
        test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)

        history['train_losses'].append(eval_train_loss)
        history['train_accs'].append(eval_train_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc)

        print(f"Epoch {epoch+1} - "
              f"train_loss: {eval_train_loss:.4f} train_acc: {eval_train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}")

    _, _, final_test_preds = evaluate(model, test_loader, criterion, device)

    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    precision, recall, f1 = save_all_plots(
        history, y_test, final_test_preds, x_test, plots_dir,
        n_classes=10, title_prefix='[Library] ')

    print("\n=== Library impl — итоговые метрики на тесте ===")
    print(f"Accuracy: {accuracy(y_test, final_test_preds):.4f}")
    print(f"Macro precision: {macro_avg(precision):.4f}")
    print(f"Macro recall:    {macro_avg(recall):.4f}")
    print(f"Macro F1:        {macro_avg(f1):.4f}")
    print("\nPer-class:")
    print("class | precision | recall  | f1")
    for c in range(10):
        print(f"  {c}   | {precision[c]:.4f}    | {recall[c]:.4f}  | {f1[c]:.4f}")
    print(f"\nPlots saved to {plots_dir}")


if __name__ == '__main__':
    train()