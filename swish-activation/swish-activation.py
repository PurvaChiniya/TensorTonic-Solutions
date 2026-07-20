import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x, dtype = np.float64)
    y = 1 /(1 + np.exp(-x))
    return x*y 
    
    