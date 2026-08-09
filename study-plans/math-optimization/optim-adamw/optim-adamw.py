import numpy as np

def adamw_compare(X, y, lr, beta1, beta2, weight_decay, n_epochs):
    """
    Returns: tuple of (adam_l2_losses, adamw_losses)
    """
    # linear regression MSE loss scalar 
    X = np.array(X, dtype = np.float32)
    y = np.array(y, dtype = np.float32)
    adam_l2_losses = []
    adamw_losses= []
    weight_adam = np.zeros(X.shape[1], dtype = np.float32)
    weight_adamw = np.zeros(X.shape[1], dtype = np.float32)
    g = 0 
    g_w= 0 
    N,d = np.shape(X)
    m = np.zeros(d, )
    v = np.zeros(d)
    m_w = np.zeros(d, )
    v_w = np.zeros(d)
    
    for i  in range(n_epochs):
        error = X@weight_adam -y
        error_w = X@weight_adamw -y 
        g = (2/N)*(X.T)@(error) + weight_decay*weight_adam
        g_w = (2/N)*(X.T)@(error_w)

        adam_l2_losses.append(np.mean(error**2))
        adamw_losses.append(np.mean(error_w**2))
        

        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g**2
        m_w = beta1*m_w + (1-beta1)*g_w
        v_w = beta2*v_w + (1-beta2)*g_w**2
        t = i+1
        m_hat = m/ (1-beta1**t)
        v_hat = v/ (1-beta2**t)
        m_hat_w = m_w/ (1-beta1**t)
        v_hat_w = v_w/ (1-beta2**t)

        weight_adam = weight_adam - lr*m_hat/(np.sqrt(v_hat)+1e-8)
        weight_adamw = weight_adamw - lr*m_hat_w/(np.sqrt(v_hat_w)+1e-8)- weight_decay*lr*weight_adamw

    return (adam_l2_losses, adamw_losses)
        
        
        
        
        
        
        
        
        