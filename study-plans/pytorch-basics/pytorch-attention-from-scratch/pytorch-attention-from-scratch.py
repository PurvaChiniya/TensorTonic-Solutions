import torch

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: attention output tensor
    """
    dk = Q.shape[-1]
    scores = Q@K.transpose(-2,-1) 
    scores = scores / (dk**0.5)
    return torch.softmax(scores, dim = -1)@V
    