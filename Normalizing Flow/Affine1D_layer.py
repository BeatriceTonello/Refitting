import numpy as np

import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

class CondAffine1D(nn.Module):
    
    """ One conditional affine 1D flow layer: y = x * exp(s(parameters)) + t(parameters) """
    
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)      # outputs [s, t]
        )

    def forward(self, x, parameters):
        
        """ Forward (data -> latent): returns y and log|dy/dx| """
        """
        x: (B, 1)
        cond: (B, dim)
        """
        
        st = self.net(parameters)           # (B,2)
        s = st[:, :1]                 # (B,1)
        t = st[:, 1:]                 # (B,1)
        y = x * torch.exp(s) + t      # y = x * exp(s(parameters)) + t(parameters)
        logdet = s.squeeze(1)         # log |dy/dx| = s  (since dy/dx = exp(s))
        return y, logdet

    def inverse(self, y, parameters):
        
        """ Inverse (latent -> data): returns x and log|dx/dy| """
        
        st = self.net(parameters)
        s = st[:, :1]
        t = st[:, 1:]
        x = (y - t) * torch.exp(-s)
        logdet = -s.squeeze(1)
        return x, logdet