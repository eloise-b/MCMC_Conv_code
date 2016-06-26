'''Function to add a planet to the disk'''

from __future__ import print_function, division
import scipy.ndimage as nd
import numpy as np
import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import radmc3dPy as r3
import astropy.io.fits as pyfits
import pdb
import os
import shutil
from keck_tools import *
import pickle
#import time
import multiprocessing

def planet_in_disk(size=256., fwhm=0.3, height=1e-9, x0=103., y0=128.):
    
    '''Putting a Gaussian in the disk as a planet
    
    --------------
    Parameters:
    size: same as the image output array from radmc3d
    fwhm: radius of planet
    height: brightness of the planet
    x0: float, position in x
    y0: float, position in y

    
    '''
    imag_obj=r3.image.readImage('image.out') 
    im=imag_obj.image[:,:,0]
    
    #size = 256. #size of the square the gaussian will be in, use size of image
    #size= npix_mod
    #fwhm = 0.3 #fwhm of gaussian eg effective radius
    #Height might need to be changes into contrast or contrast ratio...
    #height = 1e-9 # height of Gaussian

    x_range = np.arange(0, size, 1, float)
    y_range = x_range[:,np.newaxis]
    
    #Postions need to be able to take sub pixels, not sure how to do this, but since there 
    #are 1000 points in each of x and y, maybe this is doable?
    #x0 = 103. # Location of gaussian centre in x
    #y0 = 128. # Location of gaussian centre in y

    gauss_im = height*np.exp(-4*np.log(2) * ((x_range-x0)**2 + (y_range-y0)**2) / fwhm**2)

    plt.imshow(gauss_im, cmap=cm.cubehelix)
    #plt.imshow(np.arcsinh((im/np.max(im))/0.0001), cmap=cm.cubehelix)
    plt.colorbar()
    plt.savefig('gauss_im.eps')
    plt.clf()

    #put it in the disk
    #imag_obj = r3.image.readImage('image.out')
    #im = imag_obj.image[:,:,0]
    im = im+gauss_im
    plt.imshow(np.arcsinh(im/np.max(im)), cmap=cm.cubehelix)
    plt.colorbar()
    plt.savefig('disk+planet.eps')
    plt.clf()
    
    return im