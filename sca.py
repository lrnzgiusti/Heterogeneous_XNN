#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:20:59 2020

@author: ince
"""

import torch
from torch.optim import Optimizer
from itertools import tee

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

class SCA(Optimizer):
    r"""Implements successive convex approximation algorithm
    
    https://arxiv.org/pdf/1706.04769.pdf

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        rho (float, optional): linearity factor (default: 0)
        l1_pentaly (float, optional): weight decay (L1 penalty) (default: 0)

    Example:
        >>> optimizer = SCA(model.parameters(), lr=0.1, rho=0.9, l1_penalty=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=0.01, rho=0.0, l1_pentaly=0.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if rho < 0.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if l1_pentaly <= 0.0:
            raise ValueError("Invalid lambda value: {}".format(l1_pentaly))
        
        defaults = dict(lr=lr, rho=rho, l1_pentaly=l1_pentaly)
        params, layers = tee(params)
        super(SCA, self).__init__(params, defaults)
        self.d = [torch.zeros(*layer.shape).float().to(device) for layer in layers]#np.array([np.zeros(shape=(layer.shape)).astype(np.float32) for layer in params])



    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        for group in self.param_groups:
            rho = group['rho']
            lam = group['l1_pentaly']
            lr = group['lr']
            
            layer = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad # gradient of the parameters
                
                w_hat = (-1.0/lam)*(rho*d_p + (1-rho)*self.d[layer]) #compute the convex combination between the gradient and the d vector 
                p.add_(lr*(w_hat - p))
                self.d[layer] = rho*d_p + (1-rho)*self.d[layer] #update the d vector as convex combination between the gradient and the actual d vector 
                layer += 1
        return loss