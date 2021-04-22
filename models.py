# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:00:25 2021

@author: alexi
"""
from torchvision import models as models
import torch.nn as nn
import torch.nn.functional as F


# class CNN(nn.Module):

#     def __init__(self):
#         super(CNN, self).__init__()
#         self.layer1 =nn.Sequential(
#             nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
#           nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
#            nn.ReLU(),
#           nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc = nn.Linear(3 * 32 * 32, 2, bias=True)
        
#         nn.init.xavier_uniform_(self.fc.weight)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0), -1)   
#         out = self.fc(out)
#         return out



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 2)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        print(out)
        out = F.relu(out)
        return out
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x









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

def model2(pretrained, requires_grad):
    model = models.vgg16_bn(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 2 classes in total
    
    # model.fc = nn.Linear(2048, 2)    
    
    def create_head(num_features , number_classes ,dropout_prob=0.5 ,activation_func =nn.ReLU):
      features_lst = [num_features , num_features//2 , num_features//4,num_features//8,num_features//16,num_features//32]
      # features_lst=[num_features , num_features//2 , num_features//4]

      layers = []
      for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
        layers.append(nn.Linear(in_f , out_f))
        layers.append(activation_func())
        layers.append(nn.BatchNorm1d(out_f))
        if dropout_prob !=0 : layers.append(nn.Dropout(dropout_prob))
      layers.append(nn.Linear(features_lst[-1] , number_classes))
      return nn.Sequential(*layers)
    
    top_head = create_head(4096 , 2) # because ten classes
    model.fc = top_head # replace the fully connected layer
    

    return model


def mymodel():
    # my_model=VGG('VGG16')
    my_model=AlexNet(2)
    return my_model    