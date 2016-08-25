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
from keck_tools import rotate_and_fit
import pickle
#import time
import multiprocessing

def ft_and_resample(cal_ims):
    """Create the Fourier transform of a set of images, resampled onto half the
    pixel scale """
    #Number of images
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
    return cal_ims_ft


def lnprob_conv_disk_radmc3d(x, temperature=10000.0, filename='good_ims.fits',nphot="long(4e4)",\
    nphot_scat="long(2e4)", remove_directory=True, star_r=2.0, star_m=2.0, planet_mass=0.001,\
    planet_temp=1500.0, dist=120.0, pxsize=0.01, wav_in_um=3.776, mdisk=0.0001, plot_ims=False):
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
    planet_temp = float
        temperature of planet in K
    dist: float
        Distance in pc
    pxsize : float
        Pixel size in arcsec. Note that the MCMC image is sampled 2 times higher than this.
    mdisk: float
        total disk mass in M_Sun. The RadMC default is 1e-4 M_sun.
    """
   
    print("Debugging... planet_temp is: {0:5.1f}".format(planet_temp)) 
    #Parameters that go into the code
    params = {'dtog':np.exp(x[0]),'gap_depletion':np.exp(x[1]),'r_in':np.exp(x[2]),\
            'r_wall':np.exp(x[3]),'inc':x[4],'pa':x[5],'star_x':x[6],'star_y':x[7],\
            'star_temp':np.exp(x[8]),'planet_x':x[9], 'planet_y':x[10], 'planet_r':x[11]}
                
    #Target images.
    tgt_ims = pyfits.getdata(filename,0)

    #PSF Library                    
    cal_ims = pyfits.getdata(filename,1)
    
    #Resample onto half pixel size and Fourier transform.
    cal_ims_ft = ft_and_resample(cal_ims)
    
    #----------------------------------------------------------------------------------------

    #Create our working directory
    pid_str = str(os.getpid())
    print("AAAAA " + pid_str)
    shutil.rmtree(pid_str, ignore_errors=True)
    #time.sleep(10)
    os.makedirs(pid_str)
    #time.sleep(10)
    os.chdir(pid_str)
    
    #!!! This **really** shouldn't go here but should be its own routine.
    if plot_ims:
        stretch=0.01
        pxscale=0.01
        sz=128
        vmax = np.arcsinh(1/stretch)
        vmin = np.arcsinh(-2)
        extent = [-pxscale*sz/2, pxscale*sz/2, -pxscale*sz/2, pxscale*sz/2]
        
        plt.clf()
        cal_sum = np.zeros((128,128))
        for i in range(len(cal_ims)):
            shifts = np.unravel_index(np.argmax(cal_ims[i]), (128,128))
            cal_sum += np.roll(np.roll(cal_ims[i],64-shifts[0],axis=0),64-shifts[1],axis=1)
        plt.imshow(np.arcsinh(cal_sum/np.max(cal_sum)/stretch), interpolation='nearest',cmap=cm.cubehelix, extent=extent, vmin=vmin, vmax=vmax)
        plt.xlabel('Offset (")')
        plt.ylabel('Offset (")')
        ticks = np.linspace(vmin,vmax,6)
        cbar = plt.colorbar(ticks=ticks)
        #Note that the following line doesn't work in interactive mode.
        cbar.ax.set_yticklabels(["{0:5.2f}".format(y) for y in stretch*np.sinh(ticks)])
        im_name = 'cal_sum.png'
        plt.savefig(im_name)
    
    #Copy dust_kappa_carbon.inp into directory
    os.system('cp ../dustkappa_carbon.inp .')
    
    r3.analyze.writeDefaultParfile('ppdisk')
    
    #Convert parameters to RadMC3D strings
    r_in = '{0:7.3f}*au'.format(params['r_in'])
    gapin  = '[{0:7.3f}*au, {1:7.3f}*au]'.format(params['r_in'],params['r_wall'])
    gapout = '[{0:7.3f}*au, 60*au]'.format(params['r_wall'])
    gap_depletion = '[{0:10.3e}, 1e-1]'.format(params['gap_depletion'])
    dusttogas_str = "{0:8.6f}".format(params['dtog'])
    mdisk_str = '[{0:9.7f}*ms]'.format(mdisk)
    x_bound = '[{0:7.3f}*au, ({0:7.3f}+0.1)*au, {1:7.3f}*au, {1:7.3f}*1.1*au, 100*au]'.format(params['r_in'],params['r_wall'])
    n_x = [20., 30., 20., 40.]
    n_z = 60
    if params['planet_r'] != 0.0:
        star_pos = '[[{0:7.3f}*au,{1:7.3f}*au,0.0],[{2:7.3f}*au,{3:7.3f}*au,0.0]]'.format(params['star_x'],params['star_y'],params['planet_x'],params['planet_y'])
        star_temp = '[{0:7.3f}, {1:7.3f}]'.format(params['star_temp'],planet_temp)
        mass = '[{0:7.3f}*ms, {1:7.3f}*ms]'.format(star_m,planet_mass)
        radii = '[{0:7.3f}*rs, {1:7.3f}*rs]'.format(star_r,params['planet_r']) 
        staremis_type = '["blackbody","blackbody"]' 
    else:
        star_pos = '[{0:7.3f}*au,{1:7.3f}*au,0.0]'.format(params['star_x'],params['star_y'])
        star_temp = '[{0:7.3f}]'.format(params['star_temp'])
        mass = '[{0:7.3f}*ms]'.format(star_m)
        radii = '[{0:7.3f}*rs]'.format(star_r)
        staremis_type = '["blackbody"]'
       
    #edit the problem parameter file
    r3.setup.problemSetupDust('ppdisk', binary=False, mstar=mass, tstar=star_temp, rstar=radii,\
                                pstar=star_pos, dustkappa_ext="['carbon']", gap_rin=gapin,\
                                gap_rout=gapout, gap_drfact=gap_depletion, dusttogas=dusttogas_str,\
                                rin=r_in,nphot=nphot,nphot_scat=nphot_scat, nx=n_x, xbound=x_bound,\
                                nz=n_z, srim_rout=1.0, staremis_type=staremis_type,mdisk=mdisk_str)
    # run the thermal monte carlo
    os.system('radmc3d mctherm > mctherm.out') 
    #Create the image
    npix_mod = 256
    #The size per pixel in the image is pxsize * dist / 2. 
    #This is because a distance is a conversion of AU per arcsec, and we are subsampling
    #by a factor of 2.
    r3.image.makeImage(npix=npix_mod, sizeau=pxsize*dist/2*npix_mod, wav=wav_in_um, incl=params['inc'], posang=0.)

    imag_obj=r3.image.readImage('image.out') 
    im=imag_obj.image[:,:,0]
    
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
    chi_tot = rotate_and_fit(im, params['pa'],cal_ims_ft,tgt_ims, model_type, model_chi_txt,plot_ims=plot_ims)
    
    #This is "cd .."
    os.chdir(os.pardir)
                    
    #Clean up 
    #shutil.rmtree(pid_str, ignore_errors=True)
    
    if remove_directory:
        shutil.rmtree(pid_str)
    else:
        print("*** Figures saved in " + pid_str + " ***")
    
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
    #set parameters to 0.0 if you don't want them investigated
    ipar = np.array([np.log(6.894e-3),np.log(3.012e-3),np.log(11.22),np.log(22.13),48.85,129.5,0.0,0.0,np.log(8000.0),0.0,0.0,0.0])
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
 
  
