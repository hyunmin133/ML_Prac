# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 12:18:17 2021

@author: Hyun Min Oh
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        conv1 = nn.Conv2d(1,32, kernel_size = 3, stride = 1, padding = 1)
        bn1 = nn.BatchNorm2d(32)
        pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        conv2 = nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1)
        bn2 = nn.BatchNorm2d(64)
        pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        conv3 = nn.Conv2d(64,128, kernel_size = 3, stride = 1, padding = 1)
        bn3 = nn.BatchNorm2d(128)
        pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)
        
        self.layer1 = nn.Sequential(conv1, bn1, nn.ReLU(), pool1)
        self.layer2 = nn.Sequential(conv2, bn2, nn.ReLU(), pool2)
        self.layer3 = nn.Sequential(conv3, bn3, nn.ReLU(), pool3)
        
        self.fc1 = nn.Linear(4 * 4 * 128, 625, bias = True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    return out