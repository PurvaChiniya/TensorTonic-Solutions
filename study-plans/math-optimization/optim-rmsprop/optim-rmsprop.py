import numpy as np

def rmsprop(X, y, lr, decay, n_epochs):
    """
    Returns: tuple of (losses, effective_lrs) per epoch
    """
    X = np.array(X, dtype = np.float32)
    y = np.array(y, dtype = np.float32)
    w = np.zeros(X.shape[-1])
    effective_lr = []
    loss = []
    N, d = X.shape
    # runing avergae 
    E =0
    # loss is MSE 
    # gradient is 2/n x(Xw -y )
    for _ in range(n_epochs):
        error = (X@w - y)
        g = (2/N)*(X.T)@error
        loss_ = (1/N)*np.sum(error**2)
        loss.append(loss_)
        E = decay*E + (1-decay)*g**2
        # this is scalar 
        lr_ = lr*(1/np.sqrt(E+1e-8))
        effective_lr.append(lr_)
        w = w- lr_*g

    return (loss, effective_lr)
        

        

    