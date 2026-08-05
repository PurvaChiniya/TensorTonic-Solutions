import numpy as np

def matrix_determinant(A):
    """
    Returns: float, the determinant of square matrix A.
    """
    A = np.array(A, dtype = np.float32)
    return np.linalg.det(A)