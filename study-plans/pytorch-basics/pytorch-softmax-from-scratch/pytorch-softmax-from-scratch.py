import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    # n x c 
    batch_max = logits.max(dim = 1, keepdim=True ).values
    return torch.exp(logits - batch_max) / torch.sum( torch.exp(logits - batch_max), dim = 1, keepdim=True)
    
