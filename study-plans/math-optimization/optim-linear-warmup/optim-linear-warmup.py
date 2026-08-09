import numpy as np

def linear_warmup(X, y, base_lr, warmup_epochs, total_epochs):
    """Train with linear warmup learning rate schedule.

    Returns: see problem description for expected output format
    """

    X = np.array(X, dtype = np.float32)
    N, d = X.shape
    lrs = []
    losses = []
    w = np.zeros(d, dtype = np.float32)
    for i  in range(total_epochs) : 
        if i<warmup_epochs: 
            lr = base_lr * (i+1)/warmup_epochs
        else: 
            lr = base_lr
        error = X@w - y 
        
        loss = np.mean(error**2)
        g = (2/N)*(X.T)@error 
        w = w -lr*g 
        lrs.append(lr)
        losses.append(loss)
    return (lrs, losses)
    