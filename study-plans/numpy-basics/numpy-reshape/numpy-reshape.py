import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    arr = np.asarray(data, dtype=np.float64)
     
    if operation == "flatten":
        return arr.reshape(-1)

    elif operation == "transpose":
        return arr.T
    else:
        return np.expand_dims(arr, axis=0)
        
        
    return res.view(np.float64) 