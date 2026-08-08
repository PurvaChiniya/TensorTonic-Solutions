import numpy as np
import torch
def vanilla_gradient_descent(x0, y0, lr, n_iters):
    """
    Returns: dict with 'trajectory' (list of [x,y] pairs), 'final_point' ([x,y]), 'final_value' (float)
    """
    trajectory = [[x0,y0]]
    final = []
    y = y0 
    x = x0
    for _ in range(n_iters):
        y = y - lr*6*y
        x = x - lr*2*x
        trajectory.append([x,y])

    final = [x, y]
    final_pt = x**2+ 3*y*y
    return {"trajectory":trajectory, "final_point":final, "final_value":final_pt}
    
