import numpy as np
import math
import numpy as np
from scipy.special import ndtr

import numpy as np
from scipy.special import ndtr

def gelu(x):
    """
    Compute the exact Gaussian Error Linear Unit.

    Parameters
    ----------
    x : scalar, list, or np.ndarray

    Returns
    -------
    np.ndarray
        GELU applied element-wise.
    """
    x = np.asarray(x, dtype=np.float64)
    return x * ndtr(x)