import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    maxm = np.max(x, axis = -1,  keepdims = True)
    xm = np.exp(x-maxm)
    return xm / np.sum(xm, axis = -1 ,  keepdims = True)