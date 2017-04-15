#!/home/hackpython/anaconda3/bin/python

# Author: Abhishek Sharma

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def display_grid(images):
    grid = gridspec.GridSpec(5, 5, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
    count = 0
    dim = (100,100)
    for g in grid:
        ax = plt.subplot(g)
        resized = cv2.resize(images[count], dim, interpolation= cv2.INTER_AREA)
        count = count + 1
        ax.imshow(resized)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
                                    
train_images_path = "Eigenfaces/Train/"
train_path_list = []
for files in os.listdir(train_images_path):
    if os.path.isfile(os.path.join(train_images_path,files)):
        train_path_list.append(files)

# np.empty:Return a new array of given shape and type, without initializing entries. 
images = np.empty(len(train_path_list),dtype=object)

for n in range(0, len(train_path_list)):
    images[n] = cv2.imread(os.path.join(train_images_path,train_path_list[n]))

#print(np.ravel(images[0]))
#display_grid(images)


