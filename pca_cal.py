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

# np.empty:Return a new array of given shape and type, without initializing entries. 
images = np.empty(len(train_path_list),dtype=object)

for n in range(0, len(train_path_list)):
    images[n] = cv2.imread(os.path.join(train_images_path,train_path_list[n]),cv2.IMREAD_GRAYSCALE)
   

dim = (100,100)
resized = cv2.resize(images[0], dim, interpolation= cv2.INTER_AREA)
cv2.imshow("resize",resized)
cv2.imshow("original",images[0])
cv2.waitKey(0)
#print(images[0].reshape(-1))


#print(len(images))
#print(len(train_path_list))


    #image = cv2.imread(files,cv2.IMREAD_GRAYSCALE)
    
