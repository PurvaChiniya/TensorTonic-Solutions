import numpy as np

def pearson_correlation(X):
    """
    Returns: ndarray, the Pearson correlation matrix.
    """
    X = np.array(X, dtype = np.float32)
    corr = np.corrcoef(X.T)
    return corr 
    