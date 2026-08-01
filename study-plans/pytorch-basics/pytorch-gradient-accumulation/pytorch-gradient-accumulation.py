import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w = torch.asarray(w_init, dtype = torch.float32, requires_grad = True )
    last_avg = None
    for i, (x, t) in enumerate(micro_batches): 
        x = torch.tensor(x, dtype = torch.float32)
        t = torch.tensor(t, dtype = torch.float32)

        loss = (torch.dot(w,x) - t )**2
        loss.backward()
        if (i + 1) % accum_steps == 0:
            avg = w.grad / accum_steps
            last_avg = avg.tolist()
            with torch.no_grad():
                w -= lr*avg
            w.grad.zero_()
    return w.detach().tolist(), last_avg