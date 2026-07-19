import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    np.random.seed(seed)
    if kind == "uniform":
        res = np.random.uniform(size=shape)
    elif kind == "normal":
        res = np.random.normal(size=shape)
    return res 
        
