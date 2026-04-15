import torch.nn as nn

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNNet, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

