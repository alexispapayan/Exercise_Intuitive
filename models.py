# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:00:25 2021

@author: alexi
"""
from torchvision import models as models
import torch.nn as nn
import torch.nn.functional as F




def model(pretrained, requires_grad):
    '''Using a pretrained resnet50 and attaching fully connected layers to the last layer.'''  

    
    model = models.resnet50(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
 
    def create_head(num_features , number_classes ,dropout_prob=0.5 ,activation_func =nn.ReLU):
      ''' Confuguration of attached layers to the end of resnet50 '''
     
      # features_lst = [num_features ]
      features_lst=[num_features , num_features//2 , num_features//4]
      # features_lst = [num_features , num_features//2 , num_features//4,num_features//8,num_features//16,num_features//32]

      layers = []
      for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
        layers.append(nn.Linear(in_f , out_f))
        layers.append(activation_func())
        layers.append(nn.BatchNorm1d(out_f))
        if dropout_prob !=0 : layers.append(nn.Dropout(dropout_prob))
      layers.append(nn.Linear(features_lst[-1] , number_classes))
      return nn.Sequential(*layers)
    
    top_head = create_head(2048 , 2) # because we have two layers
    model.fc = top_head # replace the fully connected layer
    

    return model

