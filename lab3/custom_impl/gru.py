import numpy as np
from utils import sigmoid, d_sigmoid

class CustomGRU:
    def __init__(self, input_size, hidden_size):
        # We will keep GRU small for the sake of BPTT implementation limits.
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Update gate
        # решает, сколько новой информации записать в память
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.bz = np.zeros((hidden_size, 1))

        # Reset gate
        # решает, сколько старой памяти использовать
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.br = np.zeros((hidden_size, 1))

        # Candidate hidden state
        # веса для создания новой памяти
        # h_t = tanh(W_h [r * h_t-1, x_t])
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.bh = np.zeros((hidden_size, 1))

        # переводит скрытое состояние в число
        # y = W_hy * h_t
        self.Why = np.random.randn(1, hidden_size) * 0.1
        self.by = np.zeros((1, 1))
        
    def forward(self, inputs):
        self.last_inputs = inputs
        self.hs = {0: np.zeros((self.hidden_size, 1))}
        self.zs = {}
        self.rs = {}
        self.hs_bar = {}

        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1)
            concat_input = np.vstack((self.hs[t], x))

            # сколько новой информации записать
            z = sigmoid(np.dot(self.Wz, concat_input) + self.bz)
            # сколько старой памяти забыть
            r = sigmoid(np.dot(self.Wr, concat_input) + self.br)

            # 	уменьшаем влияние прошлого через r
            concat_reset = np.vstack((r * self.hs[t], x))

            # возможное новое состояние памяти
            h_bar = np.tanh(np.dot(self.Wh, concat_reset) + self.bh)

            self.zs[t] = z
            self.rs[t] = r
            self.hs_bar[t] = h_bar
            self.hs[t+1] = (1 - z) * self.hs[t] + z * h_bar

        # h_t = (1 - z)h_t-1 + z * h_t
        y = np.dot(self.Why, self.hs[len(inputs)]) + self.by
        return y.item()

    def train_step(self, x_seq, y_true, lr):
        y_pred = self.forward(x_seq)
        loss = (y_pred - y_true)**2
        
        dy = 2 * (y_pred - y_true)
        
        dWhy = dy * self.hs[len(x_seq)].T
        dby = dy
        
        dWz, dWr, dWh = np.zeros_like(self.Wz), np.zeros_like(self.Wr), np.zeros_like(self.Wh)
        dbz, dbr, dbh = np.zeros_like(self.bz), np.zeros_like(self.br), np.zeros_like(self.bh)
        
        dh_next = np.dot(self.Why.T, dy)
        
        for t in reversed(range(len(x_seq))):
            x = x_seq[t].reshape(-1, 1)
            h_prev = self.hs[t]
            h_bar = self.hs_bar[t]
            z = self.zs[t]
            r = self.rs[t]
            
            concat_input = np.vstack((h_prev, x))
            concat_reset = np.vstack((r * h_prev, x))
            
            # Gradients через hidden state
            dh = dh_next
            dh_prev = (1 - z) * dh
            
            # Gradients через candidate state
            dh_bar = z * dh
            dtanh = (1 - h_bar**2) * dh_bar
            
            dWh += np.dot(dtanh, concat_reset.T)
            dbh += dtanh
            
            dconcat_reset = np.dot(self.Wh.T, dtanh)
            dr_h_prev = dconcat_reset[:self.hidden_size]
            
            dh_prev += r * dr_h_prev
            
            # Gradients через update gate (z)
            dz = (h_bar - h_prev) * dh
            dz_sig = z * (1 - z) * dz
            
            dWz += np.dot(dz_sig, concat_input.T)
            dbz += dz_sig
            dconcat_input_z = np.dot(self.Wz.T, dz_sig)
            dh_prev += dconcat_input_z[:self.hidden_size]
            
            # Gradients через r
            dr = dr_h_prev * h_prev
            dr_sig = r * (1 - r) * dr
            
            dWr += np.dot(dr_sig, concat_input.T)
            dbr += dr_sig
            dconcat_input_r = np.dot(self.Wr.T, dr_sig)
            dh_prev += dconcat_input_r[:self.hidden_size]
            
            dh_next = dh_prev
            
        dWz = np.clip(dWz, -1, 1)
        dWr = np.clip(dWr, -1, 1)
        dWh = np.clip(dWh, -1, 1)
        dbz = np.clip(dbz, -1, 1)
        dbr = np.clip(dbr, -1, 1)
        dbh = np.clip(dbh, -1, 1)
        dWhy = np.clip(dWhy, -1, 1)
        dby = np.clip(dby, -1, 1)
            
        self.Wz -= lr * dWz
        self.Wr -= lr * dWr
        self.Wh -= lr * dWh
        self.bz -= lr * dbz
        self.br -= lr * dbr
        self.bh -= lr * dbh
        self.Why -= lr * dWhy
        self.by -= lr * dby
        
        return loss

