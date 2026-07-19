import numpy as np

def create_sequence(start, stop, param, kind):
    """
    Returns: 1D ndarray of float64 values
    """
    if kind == "arange":
        res = np.arange(start, stop, param, dtype = np.float64)
    else: 
        res = np.linspace(start, stop , param)
    return res 
