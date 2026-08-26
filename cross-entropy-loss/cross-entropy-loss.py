import numpy as np
import torch
def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    # mean (log p) pi fro the true class 
    # yi corect class index 
    y_pred = torch.tensor(y_pred)
    y_true =  torch.tensor(y_true)
    log_probs = torch.log(y_pred)
    chosen = -log_probs[torch.arange(y_true.shape[0]) , y_true]
    return torch.mean(chosen , dim = -1).item()
    
    