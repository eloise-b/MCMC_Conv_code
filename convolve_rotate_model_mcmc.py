"""Script to do convolve rotate model, over the grid of models that has been created.

Example line:
lnprob_conv_disk_radmc3d([log(0.006894),log(1.553e-8),log(3.012e-3),log(11.22),log(22.13),48.85,129.5],remove_directory=False)

Once MCMC is run, you need to save the chain. e.g. 

import pickle
chainfile = open('chainfile.pkl','w')
pickle.dump(chainfile,(sampler.lnprobability,sampler.chain))

With MPI:

mpirun -np 2 python convolve_rotate_model_mcmc.py
"""

#Need to loop over same parameters as the grid, so that I can change into the correct files

#Possibly should make the grid a function and convolving code a function so that I can call
#them and output the directory names to a list and then cycle the convolution code over
#that list

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
    nphot_scat="long(2e4)", r_dust='0.3*au', remove_directory=True, planet=True):
    """Return the logarithm of the probability that a disk model fits the data, given model
    parameters x.
    
    Parameters
    ----------
    x: numpy array
        [log(dtog),log(gap_depletion_1),log(gap_depletion_2),log(r_in),log(r_wall),inc,pa]
    temperature: float
        "Temperature" when viewing the monte carlo markov chain as an simulated 
        annealing process. Equivalent to scaling uncertainties in the image by
        np.sqrt(temperature).
    filename: string
        filename containing a fits file with packaged target and calibrator images.
        
    """
    
#    params = {'dtog':np.exp(x[0]),'gap_depletion_1':np.exp(x[1]),'gap_depletion_2':np.exp(x[2]),\
#                'r_in':np.exp(x[3]),'r_wall':np.exp(x[4]),'inc':x[5],'pa':x[6]}
    params = {'dtog':np.exp(x[0]),'gap_depletion_1':np.exp(x[1]),'gap_depletion_2':np.exp(x[2]),\
                'r_in':np.exp(x[3]),'r_wall':np.exp(x[4]),'inc':x[5],'pa':x[6],'x0':x[7],'y0':x[8],\
                'fwhm':np.exp(x[9]),'height':np.exp(x[10])}
                
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
    gapin  = '[0*au, {0:7.3f}*au, {1:7.3f}*au]'.format(params['r_in'],params['r_wall'])
    gapout = '[{0:7.3f}*au, {1:7.3f}*au, 60*au]'.format(params['r_in'],params['r_wall'])
    gap_depletion = '[{0:10.3e}, {1:10.3e}, 1e-1]'.format(params['gap_depletion_1'],params['gap_depletion_2'])
    r_d = 0.3 #Same as r_dust, but without the *au so that the format is correct for xbound
    x_bound = '[{0:7.3f}*au, ({0:7.3f}+0.1)*au, {1:7.3f}*au, {1:7.3f}*1.1*au, {2:7.3f}*au, {2:7.3f}*1.1*au, 100*au]'.format(r_d,params['r_in'],params['r_wall'])
    n_x = [20., 30., 10., 20., 10., 30.]
    
    #edit the problem parameter file
    r3.setup.problemSetupDust('ppdisk', binary=False, mstar='[2.0*ms]', tstar='[9000.0]',\
                                 dustkappa_ext=['carbon'], gap_rin=gapin, gap_rout=gapout,\
                                 gap_drfact=gap_depletion, dusttogas=params['dtog'], \
                                 rin=r_dust,nphot=nphot,nphot_scat=nphot_scat, \
                                 nx=n_x, xbound=x_bound)
                            
    # run the thermal monte carlo
    os.system('radmc3d mctherm > mctherm.out') 
    #Create the image
    npix_mod = 256
    r3.image.makeImage(npix=npix_mod, sizeau=0.6*npix_mod, wav=3.776, incl=params['inc'], posang=0.)

    #--- Parameters that are the same every time in the PA loop ---

    
    if planet: 
        image = planet_in_disk(size=npix_mod,x0=params['x0'],y0=params['y0'],fwhm=params['fwhm'],height=params['height'])
    else:
        imag_obj = r3.image.readImage('image.out')
        image = imag_obj.image[:,:,0]
    
    #!!! Warning the central source flux changes a little. Maybe it is best to start 
    #with a (slightly) convolved image. Play with this!
    
    #Gaussian kernel
    kernel = np.array([[.25,.5,.25],[.5,1,.5],[.25,.5,.25]])
    kernel /= np.sum(kernel)
    im = nd.filters.convolve(image,kernel)                    

    #Inclination angle, detailed disk properties can only come from RADMC-3D
    #Pa to add to the model image PA. Note that this is instrument (not sky) PA.
    
    # Define model type for if making model chi txt
    model_type = str(params['dtog']) + ',' + str(params['gap_depletion_1']) + ',' + str(params['gap_depletion_2']) + ',' + str(params['r_in']) + ',' + str(params['r_wall']) + ',' + str(params['inc']) + ',' + str(params['pa'])
    model_chi_txt = ''
    
    #This line call Keck tools
    chi_tot = rotate_and_fit(im, params['pa'],cal_ims_ft,tgt_ims, model_type, model_chi_txt)
    
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
    #pool = MPIPool()
    #if not pool.is_master():
    #    pool.wait()
    #    sys.exit(0)
    #comm = MPI.COMM_WORLD
    #nwalkers = comm.Get_size()
    nwalkers = 22
    print('nwalkers=',nwalkers)
    threads = multiprocessing.cpu_count()
    ipar = np.array([np.log(6.894e-3),np.log(1.553e-8),np.log(3.012e-3),np.log(11.22),np.log(22.13),48.85,129.5,103.,128.,np.log(0.3),np.log(1e-9)])
    # load in the results from the previous chain and use it as a starting point
    #c = open('chainfile.pkl','r')
    #old_prob,old_chain = pickle.load(c)
    #c.close()
    #define starting point as the last model of the last thread from the previous mcmc
    #ipar = old_chain[-1,-1]
    #make the starting cloud smaller
    ipar_sig = np.array([.01,.03,.01,.01,.01,0.1,0.1,0.1,0.1,0.1,0.1])
    ndim = len(ipar)
    #Could use parameters of random.normal instead of below. But Mike likes this way.
    p0 = [ipar + np.random.normal(size=ndim)*ipar_sig for i in range(nwalkers)]
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,pool=pool)
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,threads=threads, args=[old_chain, old_prob])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,threads=threads)
    #sampler.lnprobability = old_prob
    #sampler.chain = old_chain
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d, args=[temperature=170.0])
    sampler.run_mcmc(p0,1500)
    #pool.close
    
    chainfile = open('chainfile_cont.pkl','w')
    pickle.dump((sampler.lnprobability,sampler.chain),chainfile)
    chainfile.close()

#Useful things
#np.max(sampler.flatlnprobability)
#np.argmax(sampler.flatlnprobability)
#sampler.flatchain[np.argmax(sampler.flatlnprobability)]
#np.exp(sampler.flatchain[np.argmax(sampler.flatlnprobability)][0:5])
 
  
