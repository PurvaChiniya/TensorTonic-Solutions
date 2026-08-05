import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Returns: dict with 'mean', 'median', 'mode' as floats.
    """
    x = np.array(x, dtype = np.float32)
    counts = Counter(x.tolist())
    highest_count = max(counts.values())
    modes = [value for value, count in counts.items()
             if count == highest_count]
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(min(modes))
    }