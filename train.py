# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:04:17 2021

@author: alexi
"""


import models
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from engine import train,train_with_scheduler,validate
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler




matplotlib.style.use('ggplot')
# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                      
#intialize the model
model = models.model(pretrained=True, requires_grad=False).to(device)

# learning parameters
lr = 0.0001
epochs =10
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
sgdr_partial = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005) # train with learning step scheduler
criterion = nn.BCELoss()


# read the training csv file
train_csv = pd.read_csv('labels.csv')
# train dataset
train_data = ImageDataset(
    train_csv, train=True, test=False
)
# validation dataset
valid_data = ImageDataset(
    train_csv, train=False, test=False,
)
# train data loader
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size,
    shuffle=True
)
# validation data loader
valid_loader = DataLoader(
    valid_data, 
    batch_size=batch_size,
    shuffle=False
)



# start the training and validation
train_loss = []
valid_loss = []

train_f1_scores=[]
val_f1_scores=[]

train_acc_scores=[]
val_acc_scores=[]

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss,f1_train,training_accurracy = train_with_scheduler(
        model, train_loader, optimizer, criterion, sgdr_partial, train_data, device
    )
    
    # train_epoch_loss,training_accurracy = train(
    #     model, train_loader, optimizer, criterion, train_data, device
    # )
    
    
    valid_epoch_loss,f1_val,val_accuracy = validate(
        model, valid_loader, criterion, valid_data, device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
   
    train_f1_scores.append(f1_train)
    val_f1_scores.append(f1_val)

    train_acc_scores.append(training_accurracy)
    val_acc_scores.append(val_accuracy)

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Training Acc : {training_accurracy:.4f}")
    print(f"Training f1 : {f1_train:.4f}")

    print(f'Val Loss: {valid_epoch_loss:.4f}')
    print(f'Val Accuracy: {val_accuracy:.4f}')
    print(f"Val f1 : {f1_val:.4f}")

    
    
# save the trained model to disk
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, './output/model.pth')



# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./output/evaluation/loss.png')
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(train_f1_scores, color='orange', label='train f1 score')
plt.plot(val_f1_scores, color='red', label='valid f1 score')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./output/evaluation/f1_score.png')
plt.show()


plt.figure(figsize=(10, 7))
plt.plot(train_acc_scores, color='orange', label='train accuracy score')
plt.plot(val_acc_scores, color='red', label='valid accuracy score')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./output/evaluation/accuracy_score.png')
plt.show()




