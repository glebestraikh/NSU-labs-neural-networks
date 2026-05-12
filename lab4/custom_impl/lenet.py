import numpy as np

class Tanh:
    def __init__(self):
        self.input_data = None
        self.output = None

    def forward(self, input_data):
        self.input_data = input_data
        # y = tanh(x)
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_error, _learning_rate):
        # dL/dx = (1 - tanh(x)^2) * dL/dy
        # производная tanh
        return (1 - self.output ** 2) * output_error

class Linear:
    def __init__(self, input_size, output_size):
        # Матрица весов случайных чисел
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size)
        self.input_data = None

    def forward(self, input_data):
        self.input_data = input_data
        #  матричное умножение XW + be,t
        return np.dot(self.input_data, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        # weights.T - Транспонирование матрицы
        # dL/dX = dL/∂Y * W^T
        input_error = np.dot(output_error, self.weights.T)

        # dL/dW = X^T * dL/dY
        weights_error = np.dot(self.input_data.T, output_error)
        # dL/db = Σ(dL/dY)
        bias_error = np.sum(output_error, axis=0)

        # W := W -  learning_rate * dL /dW
        self.weights -= learning_rate * weights_error
        # b := b - learning_rate * dL/db
        self.bias -= learning_rate * bias_error

        return input_error

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Ядра свёртки: (out_channels, in_channels, K, K)
        # Инициализация He: scale = sqrt(2 / fan_in), fan_in = in_channels * K * K
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        # Один bias на каждый выходной канал
        self.bias = np.zeros(out_channels)
        self.input_data = None
        self.output = None

    def forward(self, input_data):
        self.input_data = input_data
        batch_size, in_c, h, w = input_data.shape
        # H_out = H_in - K + 1
        out_h = h - self.kernel_size + 1
        # W_out = W_in - K + 1
        out_w = w - self.kernel_size + 1

        self.output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                # Вырезаем окно
                region = self.input_data[:, np.newaxis, :, i:i+self.kernel_size, j:j+self.kernel_size]
                # Формула для одного элемента выхода:
                # output[b, c, i, j] = Σ_{m=0}^{in_c-1} Σ_{p=0}^{k-1} Σ_{q=0}^{k-1}
                #                      input[b, m, i+p, j+q] * weight[c, m, p, q]
                #                      + bias[c]
                self.output[:, :, i, j] = np.sum(region * self.weights[np.newaxis, :, :, :, :], axis=(2, 3, 4)) + self.bias

        return self.output

    def backward(self, output_error, learning_rate):
        batch_size, in_c, h, w = self.input_data.shape
        # H_out = H_in - K + 1
        out_h = h - self.kernel_size + 1
        # W_out = W_in - K + 1
        out_w = w - self.kernel_size + 1

        input_error = np.zeros_like(self.input_data)  # Градиент по входу (dL/dX)
        weights_error = np.zeros_like(self.weights)  # Градиент по весам (dL/dW)
        bias_error = np.sum(output_error, axis=(0, 2, 3))  # Градиент по bias (dL/db)

        for i in range(out_h):
            for j in range(out_w):
                err = output_error[:, :, i, j]
                # Вырезаем то же самое окно
                region = self.input_data[:, :, i:i+self.kernel_size, j:j+self.kernel_size]

                # Накопление градиента для весов
                # dL/dw[c, m, p, q] += Σ_batch  err[b, c, i, j] * input[b, m, i+p, j+q]
                weights_error += np.sum(
                    err[:, :, np.newaxis, np.newaxis, np.newaxis] * region[:, np.newaxis, :, :, :],
                    axis=0
                )


                # Накопление градиента по входу
                # dL/dx[b, m, i+p, j+q] += Σ_c err[b, c, i, j] * w[c, m, p, q]
                input_error[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += np.sum(
                    err[:, :, np.newaxis, np.newaxis, np.newaxis] * self.weights[np.newaxis, :, :, :, :],
                    axis=1
                )

        # W := W - lr * dL/dW
        self.weights -= learning_rate * weights_error
        # b := b - lr * dL/db
        self.bias -= learning_rate * bias_error

        return input_error

class AvgPool2d:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_data = None
        self.output = None

    def forward(self, input_data):
        self.input_data = input_data
        batch_size, c, h, w = input_data.shape
        # Вычисляем размер выходной карты
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        self.output = np.zeros((batch_size, c, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                # Вычисляем координаты левого верхнего угла окна
                h_start = i * self.stride
                w_start = j * self.stride
                # Вырезаем окно K x K
                region = input_data[:, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                # output[b, c, i, j] = (1 / K^2) * Σ region
                self.output[:, :, i, j] = np.mean(region, axis=(2, 3))

        return self.output

    def backward(self, output_error, _learning_rate):
        out_h, out_w = output_error.shape[2:]
        input_error = np.zeros_like(self.input_data)

        # При усреднении градиент равномерно делится между K*K элементами окна
        n_elements = self.kernel_size * self.kernel_size

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                # dL/dx для каждого элемента окна = dL/dy / K^2
                err = output_error[:, :, i, j] / n_elements
                # Раскидываем градиент обратно в окно входа
                input_error[:, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size] += err[:, :, np.newaxis, np.newaxis]

        return input_error

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input_data):
        self.input_shape = input_data.shape
        # (B, C, H, W) -> (B, C*H*W)
        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, output_error, _learning_rate):
        # Возвращаем градиент к 4D форме входа
        return output_error.reshape(self.input_shape)

def softmax_cross_entropy(logits, labels):
    batch_size = logits.shape[0]

    # Вычитаем максимум для численной стабильности (exp не уходит в бесконечность)
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    # превращаем в вероятности
    # softmax: p_i = exp(z_i) / Σ exp(z_j)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # функция потерь: L = -(1/N) Σ log(p_{y_i})
    # 1e-9 — защита от log(0)
    log_likelihood = -np.log(probs[range(batch_size), labels] + 1e-9)
    loss = float(np.sum(log_likelihood)) / batch_size

    # grad сначала равен probs вероятностям
    # Градиент = (предсказанная вероятность — правильный ответ) / batch_size
    grad = probs.copy()
    grad[range(batch_size), labels] -= 1
    grad /= batch_size

    # loss - одно число. Показывает, насколько плохо сеть работает
    # grad - градиент. Этот массив идёт дальше в backward() последнего слоя
    # probs - вероятности
    return loss, grad, probs

class LeNet5:
    def __init__(self):
        # вход 32x32, выход 10 классов
        self.layers = [
            # Свёртка: 6 фильтров (ядер) размером 5×5
            Conv2d(1, 6, 5),     # Output: (6, 28, 28) from 32x32 input
            # Функция активации
            Tanh(),
            # Усредняющий пулинг 2×2 с шагом 2
            AvgPool2d(2, 2),     # Output: (6, 14, 14)
            # Вторая свёртка: 16 фильтров размером 5×5
            Conv2d(6, 16, 5),    # Output: (16, 10, 10)
            # Функция активации
            Tanh(),
            # Усредняющий пулинг 2×2
            AvgPool2d(2, 2),     # Output: (16, 5, 5)
            # Выпрямляет карту в вектор
            Flatten(),
            # полносвязный слой
            Linear(16 * 5 * 5, 120),
            # Функция активации
            Tanh(),
            # полносвязный слой
            Linear(120, 84),
            # Функция активации
            Tanh(),
            # полносвязный слой
            Linear(84, 10)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad, lr)



