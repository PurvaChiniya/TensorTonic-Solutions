import numpy as np

def adam(X, y, lr, beta1, beta2, n_epochs):
    """
    Returns: tuple of (losses, final_weights)
    """
    loss = []
    final_weights = []
    # MSE loss for this 
    X = np.array(X, dtype = np.float32)
    y = np.array(y, dtype = np.float32)
    N, d = X.shape
    m = np.zeros(d, dtype=np.float32)
    v = np.zeros(d, dtype=np.float32)
    losses = []
    w = np.zeros(d, dtype = np.float32) # no bias term 
    for i in range(n_epochs):
        error = X@w - y 
        gradient = (2/N)*(X.T)@(error)
        loss = (1/N)*(np.sum(error**2))
        losses.append(loss)

        m = m*beta1 + (1-beta1)*(gradient)
        v = v*beta2 + (1-beta2)*(gradient**2)

        m_hat = m / (1-beta1**(i+1))
        v_hat =  v / (1-beta2**(i+1))

        w = w - lr*(m_hat)/(np.sqrt(v_hat)+1e-8)
    
    return (losses,w )