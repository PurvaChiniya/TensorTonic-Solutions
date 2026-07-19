import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p 

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        if not self.training: 
            return x 
        if self.p == 1.0:
            return torch.zeros_like(x)
        self.mask = torch.rand_like(x) > self.p
        
        return (x *self.mask )/ (1-self.p)
