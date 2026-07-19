import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype = torch.float32)
    if method == "relu":
        res = torch.where(x>0, x, 0  )
    
    elif method == "sigmoid":
        res = 1/ (1 + torch.exp(-x))
    elif method == "tanh":
        res = (torch.exp(x) - torch.exp(-x)) /  (torch.exp(x) +torch.exp(-x))
    else: 
        res = torch.where(x>0, x, 0.01*x)
    return res.tolist()
    