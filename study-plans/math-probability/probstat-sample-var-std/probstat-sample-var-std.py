import numpy as np

def sample_var_std(x):
    """
    Returns: dict with 'variance' and 'std_dev' as floats.
    """
    x = np.array(x, dtype = np.float32)
    mean_x = np.mean(x, dtype = np.float32)
    
    
    variance = np.sum((x - mean_x )**2)

    n = len(x)
    var = variance/ (n-1)
    return {"variance": var, "std_dev": np.sqrt(var)}
    