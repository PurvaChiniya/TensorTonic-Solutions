import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x, dtype = np.float64)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))