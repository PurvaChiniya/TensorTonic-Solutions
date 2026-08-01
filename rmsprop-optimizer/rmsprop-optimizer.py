import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    s = np.asarray(s, dtype = np.float32)
    g = np.asarray(g,dtype = np.float32)
    w = np.asarray(w, dtype = np.float32)
    
    
    s = beta*s + (1-beta)*g**2
    w  -= lr*(g)/(np.sqrt(s+eps))
    return (w,s)