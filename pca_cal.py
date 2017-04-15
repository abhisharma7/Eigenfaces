#!/home/hackpython/anaconda3/bin/python

# Author: Abhishek Sharma
# Program: Eigenfaces 

import os
import numpy as np
from scipy.misc import * 
from scipy import linalg
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

class Eigenfaces:

    def __init__(self):
        self.traindir = 'Eigenfaces/Train/'
        self.testdir  =  'Eigenfaces/Test/'
        self.images_path = []
        self.displaygrid = True
        self.main_function()
    
    def main_function(self):
            
        images_array = self.load_images()
    
        if self.displaygrid:
            self.display_images(100,100)

    def load_images(self):
        
        self.images_path = glob.glob(self.traindir + '/*.jpg')
        images_array = np.array([imread(i,True).flatten() for i in self.images_path])        
        return images_array

    def display_images(self,width,height):

        grid = gridspec.GridSpec(5, 5, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
        count = 0
        dim = (width,height)
        for g in grid:
            ax = plt.subplot(g)
            resized = cv2.resize(cv2.imread(self.images_path[count]), dim, interpolation= cv2.INTER_AREA)
            count = count + 1
            ax.imshow(resized)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()


if __name__ == '__main__':
    Eigenfaces()
