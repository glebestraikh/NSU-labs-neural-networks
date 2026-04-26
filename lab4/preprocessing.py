import gzip
import struct
import numpy as np
import os

def read_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)

def read_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def get_data(data_dir):
    x_train = read_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = read_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    x_test = read_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = read_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    x_train_padded = np.pad(x_train, ((0,0), (0,0), (2,2), (2,2)), mode='constant')
    x_test_padded = np.pad(x_test, ((0,0), (0,0), (2,2), (2,2)), mode='constant')

    return x_train_padded, y_train, x_test_padded, y_test

