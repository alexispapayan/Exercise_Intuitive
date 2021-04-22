# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:09:55 2021

@author: alexi
"""


import os
import re
import csv
import pandas as pd
import cv2
import numpy as np
from sklearn.utils import shuffle

'''Added background pictures from: https://www.kaggle.com/balraj98/stanford-background-dataset?select=images '''


# Going through test and training files to create csv with labels

background_files=os.listdir('dataset/training/background/')

catfiles = os.listdir('dataset/training/cats/')
birdfiles = [filename for filename in os.listdir('dataset/training/birds/') if '.jpeg' in filename]
catbirdfiles=os.listdir('dataset/training/cats_and_birds/')
test_cat=os.listdir('dataset/test/cats/')
test_catbird=os.listdir('dataset/test/catbirds/')
test_background=os.listdir('dataset/test/background/')

cat_exclusions=[]
cat_bird_exclusions=['266','1682','1707','1849','2459','6976','7442','9560','13357','19890','26741','27226','32195','32584']
test_bird=os.listdir('dataset/test/birds')
bird_exclusions=['4419','13940','15217','31049']
bird_cat_exclusions=['6841','23035','27200','27738','31456','31735']


with open('labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","Use", "Cat", "Bird"])
    for filename in background_files:
        writer.writerow(["training_background_"+filename.split('.')[0],"Train", 0, 0])
    for filename in catfiles:
        writer.writerow(["training_cat_"+filename.split('.')[0],"Train", 1, 0])
    for filename in birdfiles:
        writer.writerow(["training_bird_"+filename.split('.')[0],"Train", 0, 1])
    for filename in catbirdfiles:
        writer.writerow(["training_catbird_"+filename.split('.')[0],"Train", 1, 1])
        
        
        
    for filename in test_background:
            writer.writerow(["test_background_"+filename.split('.')[0], "Test",0, 0])
    for filename in test_catbird:
            writer.writerow(["test_catbird_"+filename.split('.')[0], "Test",1, 1])
    for filename in test_cat:
        if  filename.split('.')[0] in cat_bird_exclusions:
            writer.writerow(["test_cat_"+filename.split('.')[0],"Test", 1, 1])
        elif filename.split('.')[0] in cat_exclusions:
            writer.writerow(["test_cat_"+filename.split('.')[0],"Test", 0, 1])
        else:
            writer.writerow(["test_cat_"+filename.split('.')[0], "Test",1, 0])
    for filename in test_bird:
        if  filename.split('.')[0] in bird_exclusions:
            writer.writerow(["test_bird_"+filename.split('.')[0],"Test", 1, 0])
        if  filename.split('.')[0] in bird_cat_exclusions:
            writer.writerow(["test_bird_"+filename.split('.')[0],"Test", 1, 1])
        else:
            writer.writerow(["test_bird_"+filename.split('.')[0], "Test",0, 1])
     
df=pd.read_csv('labels.csv')           
train=df[(df.Use=='Train')]           
test=df[(df.Use=='Test')]
test=shuffle(test)
result = pd.concat([train,test])
result.to_csv('labels.csv')