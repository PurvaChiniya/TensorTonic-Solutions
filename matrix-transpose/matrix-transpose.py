import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A, dtype = np.float32)
    
    return np.transpose(A, (-1,-2))
