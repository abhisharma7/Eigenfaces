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
        self.displaygrid = False
        self.display = False
        self.displayeigen = True
        self.main_function()
    
    def main_function(self):
            
        images_array = self.load_images()
        if self.displaygrid:
            self.display_images(100,100,img=None,griddisplay=True)

        eigen_face,weight, mean_face = self.pca_calculation(images_array)
        mean_face_display = mean_face.reshape(425,425)
       
        if self.display:
            self.display_images(100,100,img=mean_face_display,griddisplay=False)
       
        if self.displayeigen:
            self.display_eigenface(eigen_face)

    
    def pca_calculation(self,images_array):
        
        mean_face = np.mean(images_array,0)
        mean_data = images_array - mean_face
        eigen_face, sigma, v = np.linalg.svd(mean_data.transpose(),full_matrices=False)
        weight = np.dot(mean_data,eigen_face)
        return eigen_face,weight,mean_face


    def load_images(self):
        
        self.images_path = glob.glob(self.traindir + '/*.jpg')
        images_array = np.array([imread(i,True).flatten() for i in self.images_path])        
        return images_array


    def display_images(self,width,height,img,griddisplay):
        dim = (width,height)
        if griddisplay:
            grid = gridspec.GridSpec(5, 5, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
            count = 0
            for g in grid:
                ax = plt.subplot(g)
                resized = cv2.resize(cv2.imread(self.images_path[count]), dim, interpolation= cv2.INTER_AREA)
                count = count + 1
                ax.imshow(resized)
                ax.set_xticks([])
                ax.set_yticks([])
            plt.show()

        else:
            resized = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
            imgplot = plt.imshow(resized,cmap='gray')
            plt.show()
    
    def display_eigenface(self,eigen_face):
    
        dim = (100,100)
        grid = gridspec.GridSpec(5, 5, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
        count = 0
        for g in grid:
            ax = plt.subplot(g)
            resized = cv2.resize(eigen_face[:,count].reshape(425,425),dim,interpolation=cv2.INTER_AREA)
            count = count + 1
            ax.imshow(resized,cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

if __name__ == '__main__':
    Eigenfaces()
