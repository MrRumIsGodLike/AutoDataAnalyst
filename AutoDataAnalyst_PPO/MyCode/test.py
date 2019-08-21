import numpy as np


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


x = 30
re = sigmoid(x)
print(re)
