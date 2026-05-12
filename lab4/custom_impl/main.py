import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import get_data
from lenet import LeNet5, softmax_cross_entropy
import numpy as np
from tqdm import tqdm

from metrics import accuracy, macro_avg
from plotting import save_all_plots


def predict(model, x, batch_size=64):
    # Прогон по батчам, чтобы не держать весь датасет в forward сразу
    preds = []
    n = 0
    for i in range(0, len(x), batch_size):
        logits = model.forward(x[i:i + batch_size])
        # argmax по классам — берём наиболее вероятный класс
        preds.append(np.argmax(logits, axis=1))
        n += logits.shape[0]
    return np.concatenate(preds)

# Оценивает модель на тестовых данных: считает средний loss и предсказания
def eval_loss_and_preds(model, x, y, batch_size=64):
    # Считаем loss и предсказания за один проход по тесту
    preds = []
    total_loss = 0.0
    n = 0
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model.forward(xb)
        loss, _, probs = softmax_cross_entropy(logits, yb)
        preds.append(np.argmax(probs, axis=1))
        # Взвешиваем по размеру батча - последний батч может быть меньше
        total_loss += loss * xb.shape[0]
        n += xb.shape[0]
    return total_loss / n, np.concatenate(preds)


def train():
    # MNIST уже padded до 32x32 в препроцессинге
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MNIST-master')
    x_train, y_train, x_test, y_test = get_data(data_dir)

    # x_train, y_train = x_train[:20000], y_train[:20000]
    # x_test, y_test = x_test[:5000], y_test[:5000]

    model = LeNet5()
    # Гиперпараметры обучения
    epochs = 5
    batch_size = 32
    lr = 0.001

    # История метрик по эпохам — для графиков
    history = {
        'step_losses': [],
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
    }

    for epoch in range(epochs):
        # Перемешивание выборки каждую эпоху, чтобы батчи отличались
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]

        epoch_loss_sum = 0.0
        seen = 0
        for i in tqdm(range(0, len(x_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            xb = x_train[i:i + batch_size]
            yb = y_train[i:i + batch_size]

            # Прямой проход
            logits = model.forward(xb)

            # Loss + градиент
            loss, grad, _ = softmax_cross_entropy(logits, yb)

            # # Обратный проход + обновление весов
            model.backward(grad, lr)

            history['step_losses'].append(loss)
            # Взвешиваем по размеру батча, чтобы получить среднее по эпохе
            epoch_loss_sum += loss * xb.shape[0]
            seen += xb.shape[0]

        # Метрики после эпохи: и на train, и на test
        train_loss = epoch_loss_sum / seen
        train_preds = predict(model, x_train)
        train_acc = accuracy(y_train, train_preds)

        test_loss, test_preds = eval_loss_and_preds(model, x_test, y_test)
        test_acc = accuracy(y_test, test_preds)

        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc)

        print(f"Epoch {epoch+1} - "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}")

    # Финальные предсказания на тесте — для метрик и графиков
    _, final_test_preds = eval_loss_and_preds(model, x_test, y_test)

    # Сохраняем все графики (loss/acc по эпохам, per-class метрики, примеры)
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    precision, recall, f1 = save_all_plots(
        history, y_test, final_test_preds, x_test, plots_dir,
        n_classes=10, title_prefix='[Custom] ')

    print("\n=== Custom impl — итоговые метрики на тесте ===")
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