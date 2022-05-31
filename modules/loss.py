# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:51:17 2022

@author: Nicole
"""

import numpy as np
import sys
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F



"""
params: dict
{
'criterion': 'MAE'/'RMSE'
}
"""
class ReconLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
  
    def forward(self, origin, recon):
        assert origin.shape == recon.shape
        assert len(origin.shape) == 3       # (sample, sentence, vector)
        if self.params['criterion'] == 'MAE':
            return np.mean(origin - recon, axis=(0,1))
        elif self.params['criterion'] == 'RMSE':
            return np.sqrt(np.mean(np.square(origin - recon), axis=(0,1)))
        else:
            print('Wrong ReconLoss criterion!')
            sys.exit(0)



class DiscriminatorLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
    
    def forward(self, abstract, reality):
        x_fake = abstract
        x_real = reality
        discriminator = self.params['discriminator']
        d_real = discriminator(x_real)
        d_fake = discriminator(x_fake)
        d_loss = torch.log(d_real).mean() - torch.log(1-d_fake).mean()
        return d_loss



