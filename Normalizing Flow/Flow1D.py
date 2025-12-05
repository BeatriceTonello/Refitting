import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from Affine1D_layer import CondAffine1D

torch.manual_seed(0)
np.random.seed(0)

# x is my log10(rho)

class CondFlow1D(nn.Module):
    
    def __init__(self, dim=6, n_layers=6, hidden=64): # dim = 6 is the number of parameters (A,alpha,beta,M0,rho,e0)
        
        super().__init__()
        self.layers = nn.ModuleList([CondAffine1D(dim, hidden) for _ in range(n_layers)])

    def forward(self, x, parameters):
        
        """ Map data x -> z and return z and total log(det) """
        """ 
        x: (B,1) 
        parameters: (B, dim)
        """
        
        logdet_sum = torch.zeros(x.size(0), device=x.device)
        z = x
        
        for layer in self.layers:
            z, logdet = layer(z, parameters)
            logdet_sum += logdet
            
        return z, logdet_sum

    def inverse(self, z, parameters):
        
        """ Map z -> x (sampling), returns x and total logdet (w.r.t z->x) """
        """
        z: (B,1)
        """
        
        logdet_sum = torch.zeros(z.size(0), device=z.device)
        x = z
        
        for layer in reversed(self.layers):
            x, logdet = layer.inverse(x, parameters)
            logdet_sum += logdet
            
        return x, logdet_sum

    def log_prob(self, x, parameters):
        """ Compute log p(x | parameters) """
        """
        x: (B,1)
        parameters: (B, dim)
        returns (B,) log probabilities
        """
        
        z, logdet = self.forward(x, parameters)  # z: (B,1), logdet: (B,)
        
        # base Gaussian log prob (standard normal)
        log_pz = -0.5 * (z**2) - 0.5 * math.log(2 * math.pi)  # (B,1)
        log_pz = log_pz.squeeze(1)  # (B,)
        return log_pz + logdet       # (B,)

    def sample(self, num_samples, parameters, device=None):
        """ Sample num_samples for each conditioning vector """
        """
        cond: (C, dim) or (dim,) or single tensor.
        Returns: samples (C*num_samples, 1) and cond_repeated (C*num_samples, dim)
        """
        if device is None:
            device = next(self.parameters()).device
            
        # normalize cond shape to (C, dim)
        parameters = torch.as_tensor(parameters, dtype=torch.float32, device=device)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)  # (1, dim)
        C = parameters.size(0)
        
        # sample z
        z = torch.randn(C * num_samples, 1, device=device)
        
        # repeat cond accordingly
        parameters_rep = parameters.unsqueeze(1).repeat(1, num_samples, 1).view(-1, parameters.size(1))  # (C*num_samples, dim)
        
        # inverse transform
        with torch.no_grad():
            x, _ = self.inverse(z, parameters_rep)
        return x.cpu().numpy().reshape(C, num_samples), parameters_rep.cpu().numpy().reshape(C, num_samples, -1)
