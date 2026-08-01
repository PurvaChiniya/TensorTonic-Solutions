import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    n = len(y)
    y = np.asarray(y, dtype = np.float32)
    _, counts = np.unique(y, return_counts = True)
    p = counts/ counts.sum()
    res = -np.sum(p*np.log2(p, where=p>0))
    return res