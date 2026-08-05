import numpy as np

def euclidean_distance(x, y):
    """
    Return the Euclidean distance between x and y.

    Raises:
        ValueError: If x and y have different shapes.
    """
    x_array = np.asarray(x, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)

    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same shape.")

    return float(np.sqrt(np.sum((x_array - y_array) ** 2)))