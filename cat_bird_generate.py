# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:44:55 2021

@author: alexi
"""

import cv2
import os

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


catfiles = os.listdir('dataset/training/cats/')
birdfiles = [filename for filename in os.listdir('dataset/training/birds/') if '.jpeg' in filename]

length=min(len(birdfiles),len(catfiles))

# for i in range(length):
#     cat=cv2.imread('./dataset/training/cats/'+catfiles[i])
#     bird=cv2.imread('./dataset/training/birds/'+birdfiles[i])
#     cat_bird=hconcat_resize_min([cat,bird])
#     cv2.imwrite('./dataset/training/cats_and_birds/'+str(i)+'.jpeg',cat_bird)

for i in range(length):
    cat=cv2.imread('./dataset/training/cats/'+catfiles[i])
    bird=cv2.imread('./dataset/training/birds/'+birdfiles[i])
    cat_resized=cv2.resize(cat,(400,400),interpolation=cv2.INTER_CUBIC)
    bird_resized=cv2.resize(bird,(400,400),interpolation=cv2.INTER_CUBIC)
    new_img=cv2.addWeighted(cat_resized,0.5,bird_resized,0.5,0)
    cv2.imwrite('./dataset/training/cats_and_birds/'+str(i)+'.jpeg',new_img)