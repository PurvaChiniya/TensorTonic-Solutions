import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Return:
        ndarray: softmax(Q @ K.T / sqrt(d_k)) @ V
    """
    q = np.asarray(Q, dtype=np.float32)
    k = np.asarray(K, dtype=np.float32)
    v = np.asarray(V, dtype=np.float32)

    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("Q, K, and V must be 2D arrays.")

    if q.shape[1] != k.shape[1]:
        raise ValueError("Q and K must have the same feature dimension.")

    if k.shape[0] != v.shape[0]:
        raise ValueError("K and V must contain the same number of tokens.")

    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)

    # Numerically stable softmax, applied across keys.
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attention_weights = np.exp(scores)
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

    return attention_weights @ v