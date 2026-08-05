import numpy as np

def matrix_trace(A):
    """
    Returns: float, the trace (sum of diagonal elements) of A.
    """
    n = len(A)
    res = 0 
    for i in range(n):
        res += A[i][i]
    return res 