# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:58:21 2021

@author: alexi
"""


import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    '''Class handling the datasets  '''
    
    def __init__(self, csv,train,test):
        self.csv=csv
        self.train=train
        self.test=test
        self.all_image_names=self.csv[:]['Id']
        self.all_labels = np.array(self.csv[['Cat','Bird']])
        self.train_ratio=len(self.csv[self.csv["Use"]=="Train"])
        self.valid_ratio = len(self.csv) - self.train_ratio

        if self.train==True:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = list(self.all_image_names[:self.train_ratio])
            self.labels =  list(self.all_labels[:self.train_ratio])


            # define the training transforms
            self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.RandomRotation(degrees=45),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
        elif self.train == False and self.test == False:
            print(f"Number of validation images: {self.valid_ratio}")
            self.image_names = list(self.all_image_names[-self.valid_ratio:-100])
            self.labels = list(self.all_labels[-self.valid_ratio:-100])

            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
         # set the test data images and labels, only last 100 images
        # this, we will use in a separate inference script
        elif self.test == True and self.train == False:
            self.image_names = list(self.all_image_names[-100:])
            self.labels = list(self.all_labels[-100:])

             # define the test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        use=self.image_names[index].split('_')[0]
        category=self.image_names[index].split('_')[1]
        
        if use=='training':
            if category=='background':
                image = cv2.imread(f"dataset/training/background/{self.image_names[index].split('_')[-1]}.jpg")
            if category=='cat':
                image = cv2.imread(f"dataset/training/cats/{self.image_names[index].split('_')[-1]}.jpeg")
            if category=='bird':
                image = cv2.imread(f"dataset/training/birds/{self.image_names[index].split('_')[-1]}.jpeg")
            if category=='catbird': 
                image = cv2.imread(f"dataset/training/cats_and_birds/{self.image_names[index].split('_')[-1]}.jpeg")
        if use=='test':
            if category=='background':
                image = cv2.imread(f"dataset/test/background/{self.image_names[index].split('_')[-1]}.jpg")
            if category=='catbird':
                image = cv2.imread(f"dataset/test/catbirds/{self.image_names[index].split('_')[-1]}.jpeg")
            if category=='cat':
                image = cv2.imread(f"dataset/test/cats/{self.image_names[index].split('_')[-1]}.jpeg")
            if category=='bird':
                image = cv2.imread(f"dataset/test/birds/{self.image_names[index].split('_')[-1]}.jpeg")
        
        # convert the image from BGR to RGB color format
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(use,category)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }