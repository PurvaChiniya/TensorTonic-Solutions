import torch

def causal_attention(Q, K, V):
    """
    Returns: masked attention output tensor
    """
    dk = K.shape[-1]
    scores = Q@K.transpose(-1,-2)
    scores = scores / (dk**0.5)
    upper_triangular = torch.triu(torch.ones(
            scores.shape[-2:],
            dtype=torch.bool,
            device=scores.device,
        ), diagonal = 1)
    scores = torch.where(
        upper_triangular,
        float("-inf"),
        scores
    )
        
    attention_scores = torch.softmax(scores, dim = -1)
    attention_scores = attention_scores@V
    return attention_scores
    