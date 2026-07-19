import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    # X is the tensor of shape N,D
    mean = torch.mean(X, dim = 0) # across batch 
    
    variance = torch.var(X, dim = 0, unbiased = False)
    print(mean , variance)
    X_bar = (X- mean) / torch.sqrt(variance  + eps )
    y = gamma* X_bar + beta 
    return y 
    
    
    
