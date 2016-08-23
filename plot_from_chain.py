""" 
Copyright Eloise Birchall, ANU
eloise.birchall@anu.edu.au

Script to read in the chainfile.pkl files from the MCMC and plot some results"""

from __future__ import print_function, division
import scipy.ndimage as nd
import numpy as np
#import emcee
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import radmc3dPy as r3
#import astropy.io.fits as pyfits
import pdb
import os
#from keck_tools import *
import pickle

file = open('chainfile.pkl','r')
prob, chain = pickle.load(file)
file.close()

flatprob = np.reshape(prob, 26000)
flatchain = np.reshape(chain, (26000,12))

print("minimum ln prob =", np.max(flatprob))
print("this is a chi-squared of ", np.max(flatprob)/128./128./10*2*10000) 
#10000 is the temperature, change this to the temperature that was actually used

print("the chain at this point is (with the first 4 ln) ", flatchain[np.argmax(flatprob)])
print("this means the first 4 parameters are actually ", np.exp(flatchain[np.argmax(flatprob)][0:4]))
print("and temperature of the star is ", np.exp(flatchain[np.argmax(flatprob)][8]))

#Plot the MC threads
plt.plot(prob.T)
plt.title("MC Threads")
plt.xlabel( 'Iterations')
plt.ylabel('ln probability')
plt.savefig('MCMC_thread.eps')
plt.clf()

#name of parameters and units
names=['d_to_g','gap_dep','r_in','r_wall','inclination','pa','star_x','star_y','star_temp',\
       'planet_x','planet_y','planet_r']
param_number = np.arange(0,len(flatchain[1]))
flatchain = flatchain

#Plot each of the parameters against each other
for i,k in zip(param_number,names):
    for j,l in zip(param_number,names):
        if i!=j:
            if (i in [4,5,6,7,9,10,11]) and (j in [4,5,6,7,9,10,11]):
                plt.plot((flatchain[:,i]), (flatchain[:,j]),'.')
            elif i in [4,5,6,7,9,10,11]:
                plt.plot((flatchain[:,i]), np.exp(flatchain[:,j]),'.')
            elif j in [4,5,6,7,9,10,11]:
                plt.plot(np.exp(flatchain[:,i]), (flatchain[:,j]),'.')
            else:
                plt.plot(np.exp(flatchain[:,i]), np.exp(flatchain[:,j]),'.')

            #plt.plot(np.exp(flatchain[:,i]), np.exp(flatchain[:,j]),'.')
            title = k + ' vs ' + l
            im_name = k+"_vs_"+l+".eps"
            plt.title(title)
            plt.xlabel(k)
            plt.ylabel(l)
            plt.savefig(im_name)
            plt.clf()
            
            