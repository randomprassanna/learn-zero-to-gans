#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

#get_ipython().run_line_magic('matplotlib', 'inline')

import torch.multiprocessing as mp


# In[2]:


dataset_tensor = datasets.MNIST('data', train = True, transform=transforms.ToTensor())


# In[3]:


dataset = datasets.MNIST('data', train = True) #without converting to tensor or keepi PIL format


# In[4]:


dataset[0][0]


# In[5]:


#data augmentation and normalisation
stats = ((0.1307 ), (0.3081)) #these will be used as mean and std dev of mnist data

train_transformation = transforms.Compose([transforms.RandomCrop(28,padding=4,padding_mode='reflect'),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(*stats,inplace=True)])

val_transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize(*stats)])


# In[6]:


dataset[0]


# In[7]:


train_ds, val_ds = random_split(dataset, [50000,10000])


# In[8]:


train_ds[0]


# In[9]:


type(dataset_tensor[0][0]) #image is a tensor


# In[10]:


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

#     def __getitem__(self, index):
#         return self.map(self.dataset[index])

    def __getitem__(self, index):
        if self.map:     
            x = self.map(self.dataset[index][0]) 
        else:     
            x = self.dataset[index][0]  # image
        y = self.dataset[index][1]   # label      
        return x, y

    def __len__(self):
        return len(self.dataset)


# In[11]:


# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import torchvision.transforms.functional as TF


# class MyData(Dataset):
#     def __init__(self):
#         self.images = [TF.to_pil_image(x) for x in torch.ByteTensor(10, 3, 48, 48)]
#         self.set_stage(0) # initial stage

#     def __getitem__(self, index):
#         image = self.images[index]

#         # Just apply your transformations here
#         image = self.crop(image)
#         x = TF.to_tensor(image)
#         return x

#     def set_stage(self, stage):
#         if stage == 0:
#             print('Using (32, 32) crops')
#             self.crop = transforms.RandomCrop((32, 32))
#         elif stage == 1:
#             print('Using (28, 28) crops')
#             self.crop = transforms.RandomCrop((28, 28))

#     def __len__(self):
#         return len(self.images)


# dataset = MyData()
# loader = DataLoader(dataset,
#                     batch_size=2,
#                     num_workers=2,
#                     shuffle=True)

# for batch_idx, data in enumerate(loader):
#     print('Batch idx {}, data shape {}'.format(
#         batch_idx, data.shape))

# loader.dataset.set_stage(1)

# for batch_idx, data in enumerate(loader):
#     print('Batch idx {}, data shape {}'.format(
#         batch_idx, data.shape))


# In[12]:





# ### transform fucntion only works with PIL images

# In[ ]:

map = MapDataset
train_sds = MapDataset(train_ds, train_transformation)


# In[ ]:


val_sds = MapDataset(val_ds, val_transformation)


dl = DataLoader
train_loader = DataLoader(train_sds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_sds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)




# In[ ]:







class Residual_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self,x):
        out  = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out) + x
        return out


# In[ ]:


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


# In[ ]:


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





# In[ ]:


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


# In[ ]:


batch_size=128

max_lr=0.01
epochs= 1
grad_clip = 0.1
weight_decay = 1e-4
opt_fn= torch.optim.Adam
model=Residual_net()

fn = main.fit_cycle
model = main.Residual_net()
# # In[ ]:


# import workers
# fn = workers.fit_cycle
# model = workers.res_net

# if __name__ == '__main__':
#     num_processes = 2
#     model=model
#     # NOTE: this is required for the ``fork`` method to work
#     model.share_memory()
#     processes = []
#     for rank in range(num_processes):
#         p = mp.Process(target=fn, args=(epochs, max_lr, model, train_loader, 
#                                                val_loader, 
#                                                0.1,
#                                                1e-4,
#                                                torch.optim.Adam
#                                               ))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()


# # In[ ]:


# get_ipython().run_cell_magic('time', '', 'torch.set_num_threads(4)')


# # In[ ]:





# # In[ ]:


# get_ipython().run_cell_magic('time', '', 'fit_cycle(epochs, max_lr, model, train_loader, val_loader, grad_clip=grad_clip, weight_decay=weight_decay, opt_fn=opt_fn)')


# # In[ ]:




