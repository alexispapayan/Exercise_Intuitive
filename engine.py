# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:03:19 2021

@author: alexi
"""


import torch
from tqdm import tqdm
from sklearn.metrics import precision_score,f1_score,hamming_loss,accuracy_score


def train(model, dataloader, optimizer, criterion, train_data, device):
    ''' Training function calculating the Training loss and F1 score '''
    
    
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    accuracy_running = 0.0
    hamming_running=0.0

    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
       


        preds = torch.sigmoid(outputs).data > 0.5
        preds = preds.to(torch.float32)
       
       

       
        accuracy_running += precision_score(target.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() , average="samples")*data.size(0)
        hamming_running +=hamming_loss(target.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() , average="samples")*data.size(0)
        
        # backpropagation
        loss.backward()
        
        # update optimizer parameters
        optimizer.step()
    train_loss = train_running_loss / counter
    accuracy = accuracy_running / counter
    hamming= hamming_running / counter
    
    return train_loss,accuracy


def train_with_scheduler(model , dataloader , optimizer ,criterion ,scheduler,train_data, device):
    ''' Training function calculating the Training loss, validation loss and F1 score using cosine annealing learning rate scheduler'''
    
    
    print('Training')
    model.train()

    counter = 0
    train_running_loss = 0.0
    f1_running=0.0
    accuracy_running = 0.0


    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
       
        
        optimizer.zero_grad()
        outputs = model(data)    
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
       


        preds = outputs.data > 0.5
        preds = preds.to(torch.float32)

       

        f1_running += f1_score(target.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() , average="samples")*data.size(0)
        accuracy_running += accuracy_score(target.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() )*data.size(0)


        # backpropagation
        loss.backward()
        
        # update optimizer parameters
        optimizer.step()
        
        
    scheduler.step()
    train_loss = train_running_loss / counter
    accuracy=accuracy_running/len(train_data)
    f1= f1_running / len(train_data)

    return train_loss,f1,accuracy


# validation function
def validate(model, dataloader, criterion, val_data, device):
    ''' Validation function calculating the Validation loss, accuracy and F1 score '''

    
    
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    accuracy_running = 0.0
    f1_running=0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()
            
              
            preds = outputs.data > 0.5
            preds = preds.to(torch.float32)
            
            f1_running += f1_score(target.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() , average="samples")*data.size(0)
            accuracy_running += accuracy_score(target.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() )*data.size(0)
     

        
        val_loss = val_running_loss / counter
        accuracy = accuracy_running / len(val_data)
        f1 = f1_running / len(val_data)


        return val_loss,f1,accuracy