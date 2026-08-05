import numpy as np

import numpy as np

def vector_norms(v):
    """
    Returns a float64 array of shape (3,) containing
    [L1, L2, L-infinity] norms.
    """
    v = np.asarray(v, dtype=np.float32)

    return np.array([
        np.linalg.norm(v, ord=1),       # L1 norm
        np.linalg.norm(v, ord=2),       # L2 norm
        np.linalg.norm(v, ord=np.inf)   # L-infinity norm
    ], dtype=np.float64)