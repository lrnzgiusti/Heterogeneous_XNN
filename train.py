#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 01:29:37 2020

@author: ince
"""
import torch
import torch.nn as nn
import numpy as np
import sca
import loader
from model import NN
#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Reproducibility configuration
#--------------------------------

torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
np.random.seed(42)

#%% Training hyperparameters

num_epochs = 100
batch_size = 2 
learning_rate = .5e-4
learning_rate_decay = 0.88 
reg = 0.05


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#%% Load the dataset


lo = loader.Loader(where_are_your_data='somewhere')


images_train, radiomics_train, y_train = lo.data['training_set']
images_test, radiomics_test, y_test = lo.data['training_set']

del lo ## free a bit of memory 

#%% Build torch dataset to feed the model


raw_dataset_train = torch.utils.data.TensorDataset(images_train, radiomics_train, y_train)
raw_dataset_test = torch.utils.data.TensorDataset(images_test, radiomics_test, y_test)


num_training = int(len(images_train)*0.8)
num_validation = len(images_train) - num_training

mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(raw_dataset_train, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(raw_dataset_train, mask)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)


test_loader = torch.utils.data.DataLoader(dataset=raw_dataset_test,
                                          batch_size=batch_size,
                                          shuffle=False)




#%% Build the model
input_shape = radiomics_train.shape[1]
num_classes = 2

model = NN(input_shape, num_classes).to(device)


#%% Training facilities
train_accuracy_list = []
validation_accuracy_list = []

train_loss_list = []
validation_loss_list = []

model.apply(weights_init)
model.train() #set dropout and batch normalization layers to training mode

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
optimizer = sca.SCA(model.parameters(), lr=learning_rate, rho=0.9, l1_pentaly=0.003)

# setup the learning rate
lr = learning_rate

total_train_step = len(train_loader)
total_valid_step = len(val_loader)

#%%## Start training + validation

for epoch in range(num_epochs):

    train_correct = 0
    train_total = 0
    train_loss = 0
    for i, (images, mics, labels) in enumerate(train_loader):
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + loss + optimize
        outputs = model(images, mics)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_loss += loss.item()
        if i % 225 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_train_step, loss.item()))

    print('Train accuracy is: {} %'.format(100 * train_correct / train_total))
    train_loss_list.append(train_loss / total_train_step)
    train_accuracy_list.append(100 * train_correct / train_total)
    # Code to update the lr
    if epoch % 5 == 0:
        lr *= learning_rate_decay
        update_lr(optimizer, lr)
    with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0
            for images, mics, labels in val_loader:
                outputs = model(images, mics)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += loss.item()

            print('Validataion accuracy is: {} %'.format(100 * correct / total))
            validation_loss_list.append(val_loss / total_valid_step)
            validation_accuracy_list.append(100 * correct / total)



#%% Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for  images, mics, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images, mics)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))



#%% Save The model
torch.save(model, 'model.ckpt')