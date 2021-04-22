# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:07:02 2021

@author: alexi
"""

# import the necessary packages
import os
from skimage import exposure
from skimage.exposure import cumulative_distribution
import matplotlib.pyplot as plt
import argparse
import cv2

def cdf(im):
 '''
 computes the CDF of an image im as 2D numpy ndarray
 '''
 c, b = cumulative_distribution(im) 
 # pad the beginning and ending pixels and their CDF values
 c = np.insert(c, 0, [0]*b[0])
 c = np.append(c, [1]*(255-b[-1]))
 return c

def hist_matching(c, c_t, im):
 '''
 c: CDF of input image computed with the function cdf()
 c_t: CDF of template image computed with the function cdf()
 im: input image as 2D numpy ndarray
 returns the modified pixel values
 ''' 
 pixels = np.arange(256)
 # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of   
 # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
 new_pixels = np.interp(c, c_t, pixels) 
 im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
 return im

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

catfiles = os.listdir('dataset/training/cats/')
birdfiles = [filename for filename in os.listdir('dataset/training/birds/') if '.jpeg' in filename]
length=min(len(catfiles),len(birdfiles))
for i in range(length): 
    cat=cv2.imread('dataset/training/cats/'+catfiles[i])
    bird=cv2.imread('dataset/training/birds/'+birdfiles[i])

    
    matched = exposure.match_histograms(bird, cat, multichannel=True)
    # show the output images
    # cv2.imshow("Source", bird)
    # cv2.imshow("Reference", cat)
    # cv2.imshow("Matched", matched)
    h_img = hconcat_resize_min([cat,matched])
    cv2.imwrite('dataset/training/cats_and_birds/'+str(i)+'.jpeg',h_img)

cv2.imshow("Matched1", h_img)
