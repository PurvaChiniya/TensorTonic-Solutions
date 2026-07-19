import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    n = len(pred)
    pred = torch.tensor(pred,dtype = torch.float32)
     
    if method == "mse":
         
        target =  torch.tensor(target,dtype = torch.float32)
        loss = torch.sum((pred - target)**2)
    elif method == "cross_entropy":
        target =  torch.tensor(target,dtype = torch.long)
         
        shifted = pred - pred.max(dim=1, keepdim=True).values
        log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=1,keepdim=True))
        batch_indices = torch.arange(pred.shape[0])
        correct_log_probs = log_probs[batch_indices, target]
        loss = -torch.sum(correct_log_probs)
        
    else: 
        pred = torch.tensor(pred, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
    
        error = pred - target
        abs_error = torch.abs(error)
        loss = torch.where(abs_error<delta, 0.5*abs_error*abs_error, delta * (abs_error - 0.5 * delta) )
        loss = torch.sum(loss)
            
    return loss/n 
        
