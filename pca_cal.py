#!/home/hackpython/anaconda3/bin/python

# Author: Abhishek Sharma

import os
import cv2
import numpy as np

train_images_path = "Eigenfaces/Train/"
train_path_list = []
for files in os.listdir(train_images_path):
    if os.path.isfile(os.path.join(train_images_path,files)):
        train_path_list.append(files)



print(len(train_path_list))


    #image = cv2.imread(files,cv2.IMREAD_GRAYSCALE)
    
