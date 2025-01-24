import warnings
from config import np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim > 2: warnings.warn('Probably invalid dimension. Use 1D for data point or 2D for batch data.')
    x -= np.max(x, axis=-1, keepdims=True)   # overflow 방지 (계산 결과는 동일)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
