# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:50:48 2022

@author: Nicole
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import torch.nn.Linear as Linear
import torch.nn.ReLU as ReLU
import torch.nn.LeakyReLU as LeakyReLU



"""
params: dict
{
'input_dim': input_dim,
'model': (layer_name, layer_params)
}
"""
class Discriminator(nn.Module):  # class name should be changed according to different variants
    def __init__(self, params):
        super().__init__()
        self.params = params
  
    def forward(self, embedded_inputs):
        x = embedded_inputs
        for i, l in enumerate(self.params['model']):
            lname = l[0]
            lparams = l[1]
            if i == 0:
                layer = eval(lname)(self.params['input_dim'], lparams)
            else:
                layer = eval(lname)(lparams)
            x = layer(x)
        return x





"""
params: dict
{
'input_dim': input_dim,
'linear1': layer_params of linear1 (output_dim),
'leaky_relu': layer_params of LeakyReLU (ratio)
'linear2': layer_params of linear2 (output_dim)
}
"""
class Discriminator1(nn.Module):  # class name should be changed according to different variants
    def __init__(self, params):
        super().__init__()
        self.params = params
  
    def forward(self, embedded_inputs):
        linear1 = Linear(self.params['input_dim'], self.params['linear1'])
        leaky_relu = LeakyReLU(self.params['leaky_relu'])
        linear2 = Linear(self.params['linear1'], self.params['linear2'])
        x = embedded_inputs
        x = linear1(x)
        x =  leaky_relu(x)
        x = linear2(x)
        return x





class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.size(0), *self.shape)





class Discriminator0(torch.nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            Reshape(64 * 7 * 7),
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(512, 1),
            Reshape(),
        )

    def forward(self, x):
        return self.net(x)