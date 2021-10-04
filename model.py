#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 01:10:13 2020

@author: ince
"""

import torch
import torch.nn as nn
from torchvision import models


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        #vgg16_bn = models.vgg16_bn(pretrained=True)
        #resnet18 = models.resnet18(pretrained=True)
        inceptionv3 = models.inception_v3(pretrained=True)
        # Convolutional neural-net
        #self.conv = nn.Sequential(*(list(vgg16_bn.children())[:-2]))
        #self.conv = nn.Sequential(*(list(resnet18.children())[:-2]))
        self.conv = nn.Sequential(*(list(inceptionv3.children())[:-2]))
        # Feed Forward nerual network to process the raw data
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU()
        )
        
        
        # Feed Forward neural network in the joint vector
        self.fc_layer_cat = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(25600, 1024),
            nn.LeakyReLU(),
            
            nn.Dropout(p=0.15),
            nn.Linear(1024, num_classes),
            nn.LeakyReLU()
        )

    def forward(self, imgs, radiomics):

        c = self.conv(imgs)
        f = self.fc_layer(radiomics)
        combined = torch.cat((c.view(c.size(0), -1),
                          f.view(f.size(0), -1)), dim=1)
        
        out = self.fc_layer_cat(combined)

        return out
