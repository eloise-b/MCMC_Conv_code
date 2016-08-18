"""
Copyright Eloise Birchall, Australian National University
eloise.birchall@anu.edu.au

Script to do convolve rotate model, over the grid of models that has been created.

Example line:
lnprob_conv_disk_radmc3d([np.log(6.894e-3),np.log(3.012e-3),np.log(11.22),np.log(22.13),48.85,129.5,1.0,1.0,np.log(8000.0),5.0,5.0,np.log(3000),np.log(1.0)],remove_directory=False)

Once MCMC is run, you need to save the chain. e.g. 

import pickle
chainfile = open('chainfile.pkl','w')
pickle.dump(chainfile,(sampler.lnprobability,sampler.chain))

"""

#Import the things that will be useful

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
from add_planet import *


def lnprob_conv_disk_radmc3d(x, temperature=10000.0, filename='good_ims.fits',nphot="long(4e4)",\
    nphot_scat="long(2e4)", remove_directory=True, star_r=2.0, star_m=2.0, planet_mass=0.001,\
    planet_temp=1500.0):
    #nphot_scat="long(2e4)", remove_directory=True, asymmetry=False, planet=False, planet_mass=0.001):
    """Return the logarithm of the probability that a disk model fits the data, given model
    parameters x.
    
    Parameters
    ----------
    x: numpy array
        [log(dtog),log(gap_depletion),log(r_in),log(r_wall),inc,pa,star_x,star_y,log(star_temp),\
         planet_x,planet_y,log(planet_temp),log(planet_r)]
    temperature: float
        "Temperature" when viewing the monte carlo markov chain as an simulated 
        annealing process. Equivalent to scaling uncertainties in the image by
        np.sqrt(temperature).
    filename: string
        filename containing a fits file with packaged target and calibrator images.
    nphot and nphot_scat : radmc3d inputs
        these must be in that format
    remove_directory : Boolean
        true to remove directory, false to keep it
    star_r = float
        radius of star in solar radii, will be converted to Radmc input later
    star_m = float
        mass of star in solar masses, will be converted to Radmc input later
    planet_m = float
        mass of planet in solar masses, will be converted to Radmc input later
        
    """
    
    #Parameters that go into the code
    params = {'dtog':np.exp(x[0]),'gap_depletion':np.exp(x[1]),'r_in':np.exp(x[2]),\
            'r_wall':np.exp(x[3]),'inc':x[4],'pa':x[5],'star_x':x[6],'star_y':x[7],\
            'star_temp':np.exp(x[8]),'planet_x':x[9], 'planet_y':x[10], 'planet_r':np.exp(x[11])}
                
    #Target images.
    tgt_ims = pyfits.getdata(filename,0)
    ntgt = tgt_ims.shape[0] #Number of target images.

    #PSF Library                    
    cal_ims = pyfits.getdata(filename,1)
    cal_ims = cal_ims[1:] #!!! Mike Hack !!! The fits file *itself* should be changed instead.
    ncal = cal_ims.shape[0] #Number of calibrator images.
               
    #Image size     
    sz = cal_ims.shape[1]

    #This should come from a library! but to see details, lets do it manually.
    #do the fast fourier transform of the psfs
    cal_ims_ft = np.zeros( (ncal,sz*2,sz+1),dtype=np.complex )
    for j in range(ncal):
        cal_im_ft_noresamp = np.fft.rfft2(cal_ims[j,:,:])
        cal_ims_ft[j,0:sz/2,0:sz/2+1] = cal_im_ft_noresamp[0:sz/2,0:sz/2+1]
        cal_ims_ft[j,-sz/2:,0:sz/2+1] = cal_im_ft_noresamp[-sz/2:,0:sz/2+1]

    #----------------------------------------------------------------------------------------

    #Create our working directory
    pid_str = str(os.getpid())
    print("AAAAA " + pid_str)
    shutil.rmtree(pid_str, ignore_errors=True)
    #time.sleep(10)
    os.makedirs(pid_str)
    #time.sleep(10)
    os.chdir(pid_str)
    
    #Copy dust_kappa_carbon.inp into directory
    os.system('cp ../dustkappa_carbon.inp .')
    
    r3.analyze.writeDefaultParfile('ppdisk')
    
    #Convert parameters to RadMC3D strings
    r_in = '{0:7.3f}*au'.format(params['r_in'])
    gapin  = '[{0:7.3f}*au, {1:7.3f}*au]'.format(params['r_in'],params['r_wall'])
    gapout = '[{0:7.3f}*au, 60*au]'.format(params['r_wall'])
    gap_depletion = '[{0:10.3e}, 1e-1]'.format(params['gap_depletion'])
    x_bound = '[{0:7.3f}*au, ({0:7.3f}+0.1)*au, {1:7.3f}*au, {1:7.3f}*1.1*au, 100*au]'.format(params['r_in'],params['r_wall'])
    n_x = [20., 30., 20., 40.]
    n_z = 60
    
    if params['planet_temp'] and params['planet_r'] != 0.0:
        star_pos = '[[{0:7.3f}*au,{1:7.3f}*au,0.0],[{2:7.3f}*au,{3:7.3f}*au,0.0]]'.format(params['star_x'],params['star_y'],params['planet_x'],params['planet_y'])
        star_temp = '[{0:7.3f}, {1:7.3f}]'.format(params['star_temp'],params['planet_temp'])
        mass = '[{0:7.3f}*ms, {1:7.3f}*ms]'.format(star_m,planet_mass)
        radii = '[{0:7.3f}*rs, {1:7.3f}*rs]'.format(star_r,params['planet_r']) 
        
    else:
        star_pos = '[{0:7.3f}*au,{1:7.3f}*au,0.0]'.format(params['star_x'],params['star_y'])
        star_temp = '[{0:7.3f}]'.format(params['star_temp'])
        mass = '[{0:7.3f}*ms]'.format(star_m)
        radii = '[{0:7.3f}*rs]'.format(star_r)
        
    #edit the problem parameter file
    r3.setup.problemSetupDust('ppdisk', binary=False, mstar=mass, tstar=star_temp, rstar=radii,\
                                pstar=star_pos, dustkappa_ext="['carbon']", gap_rin=gapin,\
                                gap_rout=gapout, gap_drfact=gap_depletion, dusttogas=params['dtog'],\
                                rin=r_in,nphot=nphot,nphot_scat=nphot_scat, nx=n_x, xbound=x_bound,\
                                nz=n_z, srim_rout=1.0)
                            
    # run the thermal monte carlo
    os.system('radmc3d mctherm > mctherm.out') 
    #Create the image
    npix_mod = 256
    r3.image.makeImage(npix=npix_mod, sizeau=0.6*npix_mod, wav=3.776, incl=params['inc'], posang=0.)

    imag_obj=r3.image.readImage('image.out') 
    imag=imag_obj.image[:,:,0]
    
    #!!! Warning the central source flux changes a little. Maybe it is best to start 
    #with a (slightly) convolved image. Play with this!
    
    #Gaussian kernel
    kernel = np.array([[.25,.5,.25],[.5,1,.5],[.25,.5,.25]])
    kernel /= np.sum(kernel)
    im = nd.filters.convolve(imag,kernel)                    

    #Inclination angle, detailed disk properties can only come from RADMC-3D
    #Pa to add to the model image PA. Note that this is instrument (not sky) PA.
    
    # Define model type for if making model chi txt
    model_type = str(params['dtog']) + ',' + str(params['gap_depletion']) + ',' + str(params['r_in']) + ','\
                 + str(params['r_wall']) + ',' + str(params['inc']) + ',' + str(params['pa']) + ',' \
                 + str(params['star_x']) + ',' + str(params['star_y']) + ',' + str(params['star_temp']) \
                 + ',' + str(params['planet_x']) + ',' + str(params['planet_y']) + ',' \
                + str(params['planet_r'])
    
    model_chi_txt=''
    
    #This line call Keck tools
    chi_tot = rotate_and_fit(im, params['pa'],cal_ims_ft,tgt_ims, model_type, model_chi_txt,plot_ims=False)
    
    #This is "cd .."
    os.chdir(os.pardir)
                    
    #Clean up 
    #shutil.rmtree(pid_str, ignore_errors=True)
    
    if remove_directory:
        shutil.rmtree(pid_str)
    
    #Return log likelihood
    lnlike = -0.5*chi_tot/temperature
    print("*** Computed likelihood {0:7.1f} for thread {1:s} ***".format(lnlike,pid_str))

    c = open('chain'+pid_str+'.txt','a')
    c.write(str(lnlike) + ',' + model_type + '\n')
    c.close()
        
    return lnlike
    
#Here is some code that will run with %run but not import.
if __name__ == "__main__":
    #nwalkers is set to be twice the number of parameters - should make this automatic
    nwalkers = 26
    print('nwalkers=',nwalkers)
    threads = multiprocessing.cpu_count()
    #set parameters to 0 if you don't want them investigated not log(0) actually 0.0
    ipar = np.array([np.log(6.894e-3),np.log(3.012e-3),np.log(11.22),np.log(22.13),48.85,129.5,1.0,1.0,np.log(8000.0),5.0,5.0,np.log(1.0)])
    #set parameter in cloud to zero to not investigate it
    ipar_sig = np.array([.01,.01,.01,.01,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001])
    ndim = len(ipar)
    #Could use parameters of random.normal instead of below. But Mike likes this way.
    p0 = [ipar + np.random.normal(size=ndim)*ipar_sig for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,threads=threads)
    sampler.run_mcmc(p0,5)
    
    
    chainfile = open('chainfile.pkl','w')
    pickle.dump((sampler.lnprobability,sampler.chain),chainfile)
    chainfile.close()

#Useful things
#np.max(sampler.flatlnprobability)
#np.argmax(sampler.flatlnprobability)
#sampler.flatchain[np.argmax(sampler.flatlnprobability)]
#np.exp(sampler.flatchain[np.argmax(sampler.flatlnprobability)][0:5])
 
  
