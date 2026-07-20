import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
   
    x = np.array(x, dtype = np.float64)
    axis = 0 if x.ndim == 1 else 1
    max_x = x - np.max(x, keepdims = True,axis =axis)
    return np.exp(max_x) / np.sum(np.exp(max_x), keepdims = True, axis = axis)
    