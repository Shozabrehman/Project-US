# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:55:59 2020

@author: shozab
"""


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy import asarray


def image_crop_extent(vid_file):
  
  image_list = []
  scan_video = cv2.VideoCapture(vid_file)
  while(True):   
      ret,frame = scan_video.read()     
      if ret: 
          frame_numpy = asarray(frame)
          image_list.append(frame_numpy)          
      else:
       break 
  image = asarray(image_list)
  mask1 = image[:,:,:,0] < image[:,:,:,1]*0.8
  mask2 = image[:,:,:,1] > 120
  mask3 = image[:,:,:,2] < image[:,:,:,1]*0.8
  curve_mask = mask1*mask2*mask3 
  frames = curve_mask.shape[0]
  h = curve_mask.shape[1]
  w = curve_mask.shape[2]
  zmap3d = np.tile(np.tile(np.arange(h), [w,1]).T, [frames,1,1])
  zvals = zmap3d[curve_mask]
  new_zvals = []
  for zval in zvals:
    if zval >= 100:
      new_zvals.append(zval)
  
  min_z = np.min(new_zvals)
  
  return min_z



def extract_images(vid_file):
    images = []
    cap = cv2.VideoCapture(vid_file)
    currentframe = 0
    original_name = (os.path.splitext(os.path.basename(vid_file))[0])  
    if not os.path.exists(original_name):
      os.makedirs(original_name)
    while(True):   
        ret,frame = cap.read()       
        if ret:     

            mask = (frame[:,:,0] < frame[:,:,1]*0.8)*(frame[:,:,1] > 120)*(frame[:,:,2] < frame[:,:,1]) 
            
            
            frame[mask] = [0, 0, 0]

            frame_cropped_singlechannel = frame[ 50:350, 160:500, 0]
           
            frame_numpy = asarray(frame_cropped_singlechannel)
            currentframe += 1
            images.append(frame_numpy)
        else:
         break
    return images


def display_diaphragm_lines(image, s, kernel_size): 

    a = [0]
    b = np.linspace(-2, 2, kernel_size)
    aa, bb = np.meshgrid(a, b)
    
    Mexicanhat_filter = np.exp(-( bb ** 2 ) / (2 * s ** 2))
    Mexicanhat_filter = (1/np.pi * s**4)*(1-2*( bb**2)/(2*s**2))*Mexicanhat_filter
    
    image_filtered = signal.convolve2d(image, Mexicanhat_filter, mode = 'same')
    image_filtered *= 255.0 / image_filtered.max() 
       
    def mask(image):
      max_val = np.max(image)
      limit = 3
      h = image.shape[0]
      w= image.shape[1]
      t = 0.2
        
      diaphragm_mask = np.zeros(image.shape)
          
      for x in range(0, w):
          lw = 0
          lc = 0
          allow_count = True
          for   z in range(h-1,-1,-1):
             
              if image[z,x]   > max_val*t:
                 if allow_count: 
                   lw += 1
                   if lw == limit:
                     diaphragm_mask[z, x]= 1
                     allow_count = False
                     lc += 1
                     if lc == 2:
                       break
              else:
                allow_count = True
                lw = 0
      #plt.imshow(diaphragm_mask, cmap = 'gray')
      #plt.show()
      return diaphragm_mask
    diaphragm_line = mask(image_filtered)
    
    #plt.figure()
    #plt.imshow(diaphragm_line, cmap = 'gray')
    #plt.show()
    return diaphragm_line
def display_mask(vidcap, sv,kv):  
  
  images_list = extract_images(vidcap)
  for image in images_list[0:280:10]:
    mask = display_diaphragm_lines(image, sv, kv)
    plt.figure(figsize = [15,8])
    plt.subplot(1,2,1)
    plt.imshow(image, cmap = 'gray')
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap = 'gray')
    plt.show
    
mask = display_mask("EB1_2.wmv", 0.3, 39)