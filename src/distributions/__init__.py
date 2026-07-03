import numpy as np


def raised_cosine(value, length, position, sigma=1, offset=0):
    x = (np.arange(length) - position) / sigma
    return value * 0.5 * (1 + np.cos(np.pi * x)) * np.where(np.abs(x) < 1, 1, 0) + offset


def spike(value, length, position):
    arr = np.zeros(length)
    arr[position] = value
    return arr


def homogeneous(value, length):
    return np.ones(length) * value
