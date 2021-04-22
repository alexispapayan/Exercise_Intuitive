# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:04:55 2021

@author: alexi
"""
import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import PIL


def call_classifier(img):
    

    image = transforms.ToPILImage()(img)
    image=transforms.Resize((224,224))(image)    
    
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    image=transforms.ToTensor()(image).unsqueeze(0).to(device)
    
    
    #intialize the model
    model = models.model(pretrained=False, requires_grad=False).to(device)
    # load the model checkpoint
    checkpoint = torch.load('./output/model.pth')
    
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    
    outputs=model(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    
    if np.array(outputs[0][0])>0.5 and np.array(outputs[0][1])<0.5:
            print("DANGER cat")
    elif np.array(outputs[0][0])<0.5 and np.array(outputs[0][1])>0.5:
            print("bird")
    elif np.array(outputs[0][0])>0.5 and np.array(outputs[0][1])>0.5:
            print("DANGER cat and bird") 

##### Image ############
img=cv2.imread('dataset/training/birds/0.jpeg')


call_classifier(img)


'''
############## VIDEO ##########################

cap = cv2.VideoCapture(0)


while True:
    
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    call_classifier(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

'''

