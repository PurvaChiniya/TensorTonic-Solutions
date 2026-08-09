import numpy as np

def learning_rate_sweep(X, y, learning_rates, n_epochs):
    """Train a linear model with each learning rate and return loss curves.

    Returns: see problem description for expected output format
    """
    X = np.array(X, dtype = np.float32)
    N,d = X.shape
    losses = []
    for lr in learning_rates: 
        w = np.zeros(d, dtype = np.float32)
        loss = []
        for i in range(n_epochs):
            error = (X@w -y )
            loss_ = (1/N)*np.sum(error**2)
            loss.append(loss_)
            g = (2/N)*(X.T)@error
            w = w - lr*g 
        losses.append(loss)
    return losses 