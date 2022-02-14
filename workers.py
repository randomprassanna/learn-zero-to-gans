import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


import torch.multiprocessing as mp










def fit_cycle(epochs, max_lr, model, train_loader, val_loader, opt_fn, weight_decay=0, grad_clip=None):
    history = []
    
    
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    
    optimizer = opt_fn(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
            print('batch_done')
            
        # Validation phase
        result = model.evaluate(val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history






def accuracy(outputs,labels):
    _,preds = torch.max(outputs,dim = 1)
    return torch.tensor((torch.sum(preds == labels).item())/len(preds))


class ClassificationBase(nn.Module):
    def training_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels) #it softmax done internally and labels converted internally
        return loss
    
    def validation_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels) #it softmax done internally and labels converted internally
        acc = accuracy(out,labels)
        return {'val_loss': loss.detach(), 'val_acc':acc}
    
    def evaluate(self,val_loader):
        val_outputs = [self.validation_step(batch) for batch in val_loader]
        batch_loss_list = [x['val_loss'] for x in val_outputs]
        batch_acc_list = [x['val_acc'] for x in val_outputs]
        epoch_loss = torch.stack(batch_loss_list).mean()
        epoch_acc =torch.stack(batch_acc_list).mean()
        result =  {'val_lossf':epoch_loss.item(), 'val_accf':epoch_acc.item()}
        return result
    
    def epoch_end(self,epoch, result):
            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_lossf'], result['val_accf']))

            
            
            
            
            
class Residual_net(ClassificationBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride=1, padding=1) #output is 128,8,28,28
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride=1, padding=1) #output is 128,8,28,28
        self.relu2 = nn.ReLU()
        
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(8*28*28, 10)) #output is 128,8*28*28
        
        
    def forward(self,x):
        xout  = self.conv1(x)
        out = self.relu1(xout)
        
        out = self.conv2(out)
        out = self.relu2(out) + xout
        
        out = self.classifier(out)
        return out
    
res_net = Residual_net()
    