import numpy as np

class CustomRNN:
    def __init__(self, input_size, hidden_size):
        # матрицу случайных чисел из нормального распределения h_t = W_xh * x_t
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1
        # Передаёт информацию из прошлого шага: h_t = W_h * h_t-1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        # y = W_hy * h_t
        self.Why = np.random.randn(1, hidden_size) * 0.1
        # bios для памяти
        self.bh = np.zeros((hidden_size, 1))
        #  bias выхода
        self.by = np.zeros((1, 1))
        
        self.hidden_size = hidden_size
        
    def forward(self, inputs):
        # Сохраняем всю последовательность входных данных
        self.last_inputs = inputs
        self.last_hs = {0: np.zeros((self.hidden_size, 1))}
        
        for i, x in enumerate(inputs):
            x = x.reshape(-1, 1)
            # h_t = tanh(W_xh * x_t + W_hh * h_t-1 + b)
            self.last_hs[i + 1] = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.last_hs[i]) + self.bh)

        # Берём последнюю память: h_T
        # и считаем y = W_hy * h_T + by
        y = np.dot(self.Why, self.last_hs[len(inputs)]) + self.by
        return y.item()

    def train_step(self, x_seq, y_true, lr):
        y_pred = self.forward(x_seq)
        # MSE
        loss = (y_pred - y_true)**2
        
        # dL/dy
        # Градиент ошибки по выходу
        dy = 2 * (y_pred - y_true)
        
        # Начинаем считать градиенты для последнего слоя
        # насколько каждый элемент памяти повлиял на ошибку выхода
        dWhy = dy * self.last_hs[len(x_seq)].T
        # Градиент bias выхода
        dby = dy

        # Создание пустых градиентов
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        dh_next = np.dot(self.Why.T, dy)
        
        for t in reversed(range(1, len(x_seq) + 1)):
            # Производная tanh для функции усиления
            dtanh = (1 - self.last_hs[t] ** 2) * dh_next

            # Градиент bias
            dbh += dtanh
            dWhh += np.dot(dtanh, self.last_hs[t - 1].T)
            dWxh += np.dot(dtanh, x_seq[t - 1].reshape(-1, 1).T)

            # Передача ошибки дальше назад
            dh_next = np.dot(self.Whh.T, dtanh)
            
        # Ограничение градиентов
        dWxh = np.clip(dWxh, -1, 1)
        dWhh = np.clip(dWhh, -1, 1)
        dWhy = np.clip(dWhy, -1, 1)
        dbh = np.clip(dbh, -1, 1)
        dby = np.clip(dby, -1, 1)

        # Обновление весов градиентным спуском
        self.Wxh -= lr * dWxh
        self.Whh -= lr * dWhh
        self.Why -= lr * dWhy
        self.bh -= lr * dbh
        self.by -= lr * dby
        
        return loss

