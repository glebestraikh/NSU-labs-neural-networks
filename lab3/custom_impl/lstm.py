import numpy as np
from utils import sigmoid, d_sigmoid

class CustomLSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates parameters
        # Матрица весов для forget gate f_t = sigma(W_f [h_t-1, x_t]
        # решает, какую часть старой памяти удалить
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        # Bias forget gate
        self.bf = np.zeros((hidden_size, 1))

        #  Input gate веса
        # сколько новой информации записать
        # i_t = sigma(W_i [h_t-1, x_t])
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        # сдвиг для входного gate
        self.bi = np.zeros((hidden_size, 1))

        # Candidate memory веса
        # c_t = tanh(W_c [h_t-1, x_t])
        # что можно записать в память
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        # Candidate bias
        self.bc = np.zeros((hidden_size, 1))

        # что из памяти показать наружу
        # какую часть внутренней памяти модели сделать выходом на текущем шаге
        # Output gate веса
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.bo = np.zeros((hidden_size, 1))

        # y = W_hy * h_t
        # Выходной слой веса
        self.Why = np.random.randn(1, hidden_size) * 0.1
        self.by = np.zeros((1, 1))

    def forward(self, inputs):
        self.xs = {}
        self.hs = {-1: np.zeros((self.hidden_size, 1))}
        self.cs = {-1: np.zeros((self.hidden_size, 1))}
        self.fs, self.ins, self.c_bars, self.os = {}, {}, {}, {}
        
        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1)
            # Объединение входа и прошлое состояниe
            concat = np.vstack((self.hs[t-1], x))

            # что забыть из старой памяти
            f = sigmoid(np.dot(self.Wf, concat) + self.bf)
            # сколько новой информации записать
            i = sigmoid(np.dot(self.Wi, concat) + self.bi)
            # Создаёт новую возможную память
            c_bar = np.tanh(np.dot(self.Wc, concat) + self.bc)
            # что показать наружу
            o = sigmoid(np.dot(self.Wo, concat) + self.bo)

            # Обновление памяти
            # забываем часть старого (f)
            # добавляем новое (i * c_bar)
            c = f * self.cs[t-1] + i * c_bar

            # Выход hidden state
            # решаем, какую часть памяти показать наружу (o) и применяем tanh к внутренней памяти (c)
            h = o * np.tanh(c)
            
            self.fs[t], self.ins[t], self.c_bars[t], self.os[t] = f, i, c_bar, o
            self.cs[t], self.hs[t] = c, h
            self.xs[t] = concat

        # y = W_hy * h_T + b
        y = np.dot(self.Why, self.hs[len(inputs)-1]) + self.by
        return y.item()

    def train_step(self, x_seq, y_true, lr):
        y_pred = self.forward(x_seq)
        loss = (y_pred - y_true)**2
        
        dy = 2 * (y_pred - y_true)
        dWhy = dy * self.hs[len(x_seq)-1].T
        dby = dy
        
        dWf, dWi, dWc, dWo = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo)
        dbf, dbi, dbc, dbo = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo)
        
        dh_next = np.dot(self.Why.T, dy)
        dc_next = np.zeros_like(self.cs[0])
        
        for t in reversed(range(len(x_seq))):
            dh = dh_next

            # o gate
            do = dh * np.tanh(self.cs[t])
            do_sig = self.os[t] * (1 - self.os[t]) * do
            dWo += np.dot(do_sig, self.xs[t].T)
            dbo += do_sig
            
            # dc
            dc = dc_next + dh * self.os[t] * (1 - np.tanh(self.cs[t])**2)
            
            # c_bar
            dc_bar = dc * self.ins[t]
            dtanh_c_bar = (1 - self.c_bars[t]**2) * dc_bar
            dWc += np.dot(dtanh_c_bar, self.xs[t].T)
            dbc += dtanh_c_bar
            
            # i gate
            di = dc * self.c_bars[t]
            di_sig = self.ins[t] * (1 - self.ins[t]) * di
            dWi += np.dot(di_sig, self.xs[t].T)
            dbi += di_sig
            
            # f gate
            df = dc * self.cs[t-1]
            df_sig = self.fs[t] * (1 - self.fs[t]) * df
            dWf += np.dot(df_sig, self.xs[t].T)
            dbf += df_sig
            
            # Передача ошибки назад
            dconcat = np.dot(self.Wf.T, df_sig) + np.dot(self.Wi.T, di_sig) + np.dot(self.Wc.T, dtanh_c_bar) + np.dot(self.Wo.T, do_sig)
            dh_next = dconcat[:self.hidden_size]
            dc_next = dc * self.fs[t]

        # ограничиваем градиенты
        dWf = np.clip(dWf, -1, 1)
        dWi = np.clip(dWi, -1, 1)
        dWc = np.clip(dWc, -1, 1)
        dWo = np.clip(dWo, -1, 1)
        dbf = np.clip(dbf, -1, 1)
        dbi = np.clip(dbi, -1, 1)
        dbc = np.clip(dbc, -1, 1)
        dbo = np.clip(dbo, -1, 1)
        dWhy = np.clip(dWhy, -1, 1)
        dby = np.clip(dby, -1, 1)

        # Обновление весов градиентным спуском
        self.Wf -= lr * dWf
        self.Wi -= lr * dWi
        self.Wc -= lr * dWc
        self.Wo -= lr * dWo
        self.bf -= lr * dbf
        self.bi -= lr * dbi
        self.bc -= lr * dbc
        self.bo -= lr * dbo
        self.Why -= lr * dWhy
        self.by -= lr * dby
        
        return loss


