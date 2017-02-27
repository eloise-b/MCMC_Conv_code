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

complete = True
#if the chain did not complete, use join_chain.py to make the chain_file.txt
plot_results = False
im_num = 15.
temp=10000.
#the names of the parameters
names=['d_to_g','gap_dep_1','gap_dep_2','r_dust','r_in','r_wall','inclination','pa','star_x','star_y',\
       'planet_x','planet_y','planet_r']
#value=[d_to_g,gap_dep_1,gap_dep_2,r_dust,r_in,r_wall,inclination,pa,star_x,star_y,\
#       planet_x,planet_y,planet_r]

if complete:
    file = open('chainfile.pkl','r')
    prob, chain = pickle.load(file)
    file.close()

    flatprob = np.reshape(prob, 26000)
    flatchain = np.reshape(chain, (26000,13))

    print("minimum ln prob =", np.max(flatprob))
    print("this is a chi-squared of ", np.max(flatprob)/128./128./im_num*2*temp) 
    #10000 is the temperature, change this to the temperature that was actually used

    print("the chain at this point is (with the first 6 ln) ", flatchain[np.argmax(flatprob)])
    print("this means the first 6 parameters are actually ", np.exp(flatchain[np.argmax(flatprob)][0:6]))
    
    param_number = np.arange(0,len(flatchain[1]))
    flatchain = flatchain
    
    #find the errors for each of the parameters and the appropriate value to quote
    with open('uncert_table.txt', 'w') as f:
        for i in param_number:
            samples = flatchain
            if i < 6:
                samples[:,i] = np.exp(samples[:,i])
            value = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
            #print(names[i], ' has value and errors: ', value[i])
            #print(' ')
            #print('in latex format for a table these parameters are:')
            val = value[i]
            #print('in latex format for a table these parameters are:')
            if val[0] < 1.:
            #if i in [0,1,2,8,9]:
                print(names[i], ': ', '{0:6.3e}'.format(val[0]),'$^{+','{0:6.3e}'.format(val[1]),'}_{-','{0:6.3e}'.format(val[2]),'}$')
                f.write(str(names[i])+': '+'{0:6.3e}'.format(val[0])+'$^{+'+'{0:6.3e}'.format(val[1])+'}_{-'+'{0:6.3e}'.format(val[2])+'}$\n')
            else:
                print(names[i], ': ', '{0:6.3f}'.format(val[0]),'$^{+','{0:6.3f}'.format(val[1]),'}_{-','{0:6.3f}'.format(val[2]),'}$')
                f.write(str(names[i])+': '+'{0:6.3f}'.format(val[0])+'$^{+'+'{0:6.3f}'.format(val[1])+'}_{-'+'{0:6.3f}'.format(val[2])+'}$\n')
    
    if plot_results:
        os.makedirs("param_vs_param")
        #Plot the MC threads
        plt.plot(prob.T)
        plt.title("MC Threads")
        plt.xlabel( 'Iterations')
        plt.ylabel('ln probability')
        plt.savefig('MCMC_thread.eps')
        plt.clf()    

        #Plot each of the parameters against each other
        for i,k in zip(param_number,names):
            for j,l in zip(param_number,names):
                if i!=j:
                    if (i in range(6,13)) and (j in range(6,13)):
                        plt.plot((flatchain[:,i]), (flatchain[:,j]),'.')
                    elif i in range(6,13):
                        plt.plot((flatchain[:,i]), np.exp(flatchain[:,j]),'.')
                    elif j in range(6,13):
                        plt.plot(np.exp(flatchain[:,i]), (flatchain[:,j]),'.')
                    else:
                        plt.plot(np.exp(flatchain[:,i]), np.exp(flatchain[:,j]),'.')

                    #plt.plot(np.exp(flatchain[:,i]), np.exp(flatchain[:,j]),'.')
                    title = k + ' vs ' + l
                    im_name = "param_vs_param/"+k+"_vs_"+l+".eps"
                    plt.title(title)
                    plt.xlabel(k)
                    plt.ylabel(l)
                    plt.savefig(im_name)
                    plt.clf()
                
    
else:
    f = np.loadtxt('chain_file.txt')

    flatprob = f[:,0]
    flatchain = f[:,1:]

    param_number = np.arange(0,len(flatchain[1]))
    flatchain = flatchain
    
    print("minimum ln prob =", np.max(flatprob))
    print("this is a chi-squared of ", np.max(flatprob)/128./128./im_num*2*temp) 
    #10000 is the temperature, change this to the temperature that was actually used

    print("the chain at this point is ", flatchain[np.argmax(flatprob)])
    
    #find the errors for each of the parameters and the appropriate value to quote
    for i in param_number:
        samples = flatchain
        if i < 6:
            samples[:,i] = np.exp(samples[:,i])
        value = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
        print(names[i], ' has value and errors: ', value[i])
    
    if plot_results:
        os.makedirs("param_vs_param")
    
        #Plot each of the parameters against each other
        for i,k in zip(param_number,names):
            for j,l in zip(param_number,names):
                if i!=j:
                    if (i in range(6,13)) and (j in range(6,13)):
                        plt.plot((flatchain[:,i]), (flatchain[:,j]),'.')
                    elif i in range(6,13):
                        plt.plot((flatchain[:,i]), np.exp(flatchain[:,j]),'.')
                    elif j in range(6,13):
                        plt.plot(np.exp(flatchain[:,i]), (flatchain[:,j]),'.')
                    else:
                        plt.plot(np.exp(flatchain[:,i]), np.exp(flatchain[:,j]),'.')

                    #plt.plot(np.exp(flatchain[:,i]), np.exp(flatchain[:,j]),'.')
                    title = k + ' vs ' + l
                    im_name = "param_vs_param/"+k+"_vs_"+l+".eps"
                    plt.title(title)
                    plt.xlabel(k)
                    plt.ylabel(l)
                    plt.savefig(im_name)
                    plt.clf()
            
            