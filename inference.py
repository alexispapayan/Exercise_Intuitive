# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 21:52:27 2021

@author: alexi
"""

import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import DataLoader



def NN_result(output):
    ''' Printing the result with regards to the output'''
    if np.array(outputs[0][0])>0.5 and np.array(outputs[0][1])<0.5:
        string_predicted += "cat"
    elif np.array(outputs[0][0])<0.5 and np.array(outputs[0][1])>0.5:
         string_predicted += "bird"
    elif np.array(outputs[0][0])>0.5 and np.array(outputs[0][1])>0.5:
         string_predicted += "cat bird" 
    else:
         string_predicted += ""     



# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#intialize the model
model = models.model(pretrained=False, requires_grad=False).to(device)
# load the model checkpoint
checkpoint = torch.load('./output/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


train_csv = pd.read_csv('labels.csv')
genres = np.array(['cat','bird'])
# prepare the test dataset and dataloader
test_data = ImageDataset(
    train_csv, train=False, test=False
)
test_loader = DataLoader(
    test_data, 
    batch_size=1,
    shuffle=False
)



for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    print(test_data.image_names[counter])
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
   
    
   # get the predictions by passing the image through the model
    outputs = model(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    sorted_indices = np.argsort(outputs[0])
  
    best = sorted_indices[-1:]
    print(outputs,best)

    string_predicted = ''
    string_actual = ''
    # for i in range(len(best)):
    #     string_predicted += f"{genres[best[i]]}    "
    for i in range(len(target_indices)):
        string_actual += f"{genres[target_indices[i]]}    "
    if np.array(outputs[0][0])>0.5 and np.array(outputs[0][1])<0.5:
        string_predicted += "cat"
    elif np.array(outputs[0][0])<0.5 and np.array(outputs[0][1])>0.5:
         string_predicted += "bird"
    elif np.array(outputs[0][0])>0.5 and np.array(outputs[0][1])>0.5:
         string_predicted += "cat bird" 
    else:
         string_predicted += ""     
        
    image = image.squeeze(0)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    # plt.axis('off')
    plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
    plt.savefig(f"./output/inference_{counter}.jpg")
    plt.show()