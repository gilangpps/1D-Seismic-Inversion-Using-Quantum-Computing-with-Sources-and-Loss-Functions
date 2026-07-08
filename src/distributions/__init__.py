"""
src/distributions/__init__.py
Medium property generators for 1-D elastic wave simulation.
"""

import numpy as np


def raised_cosine(value: float, length: int, position: float,
                  sigma: float = 1, offset: float = 0) -> np.ndarray:
    """
    Raised-cosine profile centred at position with half-width sigma.

        f(x) = value * 0.5 * (1 + cos(π(x-position)/sigma)) + offset
                for |x - position| < sigma, else offset.

    Parameters
    ----------
    value    : peak amplitude above offset
    length   : number of grid points
    position : centre of the cosine bell
    sigma    : half-width (in grid indices)
    offset   : baseline value

    Returns
    -------
    np.ndarray, shape (length,)
    """
    x = (np.arange(length) - position) / sigma
    bell = value * 0.5 * (1.0 + np.cos(np.pi * x))
    return np.where(np.abs(x) < 1.0, bell, 0.0) + offset


def spike(value: float, length: int, position: int) -> np.ndarray:
    """
    Single-point spike array: arr[position] = value, all others zero.
    """
    arr           = np.zeros(length)
    arr[int(position)] = value
    return arr


def homogeneous(value: float, length: int) -> np.ndarray:
    """
    Constant (homogeneous) array of given value and length.
    """
    return np.ones(int(length)) * value
