import numpy as np

# реализация функций активации и их производных

# Сжимает любое число в диапазон: (0, 1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# как сильно изменится sigmoid при изменении входа
def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

# Сжимает значения в диапазон: (-1, 1)
def tanh(x):
    return np.tanh(x)

#  нужна для backpropagation
def d_tanh(x):
    return 1 - np.tanh(x)**2

