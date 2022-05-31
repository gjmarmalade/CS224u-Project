# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:46:17 2022

@author: Nicole
"""

#%% LIBS
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from encoders import Encoder_1, Encoder_2
from decoders import Decoder_1
from discriminator import Discriminator_1
from loss import ReconLoss, DiscriminatorLoss



#%% DATA & EMBEDDING
# load gigaword dataset
pass

# BERT tokenize
pass



#%% NETWORK
class EEDDModel(nn.Module):
    def __init__(self, Encoder1, Encoder2, Decoder, Discriminator, params):
        self.encoder1 = Encoder1(params['encoder1'])
        self.encoder2 = Encoder2(params['encoder2'])
        self.decoder = Decoder(params['decoder'])
        self.discriminator = Discriminator(params['discriminator'])
  
    def forward(self, embedded_inputs):
        abstract = self.encoder1(embedded_inputs)
        minor_info = self.encoder2(embedded_inputs)
        full_info = torch.cat(abstract, minor_info, dim = -1)
        recon1 = self.decoder(abstract)
        recon2 = self.decoder(minor_info)
        recon3 = self.decoder(full_info)
        return abstract, recon1, recon2, recon3



#%% LOSS
class EEDDLoss(nn.Module):
    def __init__(self, loss_ratios):
        self.ratios = loss_ratios

    def forward(self, origin, abstract, recon1, recon2, recon3, real_sentence):
        l1 = ReconLoss(origin, recon1)
        l2 = ReconLoss(origin, recon2)
        l3 = ReconLoss(origin, recon3)
        l4 = DiscriminatorLoss(abstract, real_sentence)
        loss = self.ratios[0]*l1 - self.ratios[1]*l2 + self.ratios[2]*l3 + self.ratios[3]*l4
        return loss



#%% TRAIN
def train(loss_ratios, model, Optimizer, opt_params, epochs, trainLoader, device):
    criterion = EEDDLoss(loss_ratios)
    optimizer = Optimizer(model.parameters(), lr=opt_params['lr'], momentum=opt_params['momentum'])   # eg: Optimizer = torch.optim.SGD
    model.to(device)
    for e in range(epochs):
        for i, embedded_inputs in enumerate(trainLoader):
            embedded_inputs = embedded_inputs.to(device)
            abstract, recon1, recon2, recon3 = model(embedded_inputs)
            loss = criterion(embedded_inputs, abstract, recon1, recon2, recon3, embedded_inputs)    # real_sentence is set as the original sentence for unsupervise
            optimizer.zero_grad()   # if don't call zero_grad, the grad of each batch will be accumulated
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print('epoch: {}, batch: {}, loss: {}'.format(e+1, i+1, loss.data))
    torch.save(model, 'EEDD.pth')




