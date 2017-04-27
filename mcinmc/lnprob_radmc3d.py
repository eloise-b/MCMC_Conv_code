"""
Copyright Eloise Birchall, Mike Ireland,  Australian National University
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
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import radmc3dPy as r3
import astropy.io.fits as pyfits
import pdb
import os
import shutil
from imtools import *
import pickle
#import time
import multiprocessing

def lnprob_conv_disk_radmc3d(x, temperature=10000.0, filename='good_ims.fits',nphot="long(4e4)",\
    nphot_scat="long(2e4)", remove_directory=True, star_r=2.0, star_m=2.0, planet_mass=0.001,\
    planet_temp=1500.0, dist=120.0, pxsize=0.01, wav_in_um=3.776, mdisk=0.0001,\
    star_temp=9000.0, kappa = "['carbon']", Kurucz= True, plot_ims=False, save_im_data=False, \
    make_sed=False, data_sed_ratio = 8.672500426996962, sed_ratio_uncert=0.01, out_wall = 60., \
    out_dep = 1e-1, n_x = [5., 20., 30., 20., 40.], n_z = 60, n_y = [10,30,30,10], \
    paper_ims=False, label='', north_ims=False, rotate_present = False,
    kurucz_dir='/Users/mireland/theory/', background=None, empirical_background=True):
#def lnprob_conv_disk_radmc3d(x, temperature=10000.0, filename='good_ims.fits',nphot="long(4e4)",\
#    nphot_scat="long(2e4)", remove_directory=True, star_r=2.0, star_m=2.0, planet_mass=0.001,\
#    planet_temp=1500.0, dist=120.0, pxsize=0.01, wav_in_um=3.776, mdisk=0.0001, r_dust=0.3,\
#    star_temp=9000.0, kappa = "['carbon']", Kurucz= True, plot_ims=False, save_im_data=False, \
#    make_sed=False, rel_flux = 8.672500426996962):
    """
    Return the logarithm of the probability that a disk model fits the data, given model
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
        temperature of planet in K - must be 3500K or greater if model type is Kurucz
    dist: float
        Distance in pc
    pxsize : float
        Pixel size in arcsec. Note that the MCMC image is sampled 2 times higher than this.
    mdisk: float
        total disk mass in M_Sun. The RadMC default is 1e-4 M_sun.
    r_dust = float
        the inner radius of the inner disk in au
    star_temp = float
        Temperature of the star in Kelvin
    kappa = string
        this is the name of the dustkappa file that you want to use
    Kurucz: Boolean
        this is the type of emission you want the star to have, blackbody or interpolated 
        from a Kurucz model - Kurucz will happen if true
    rel_flux: float
        the value of the relative flux of the star compared to the disc, (disc/star), that 
        you are aiming for with the SED parameter
    out_wall: float
        outer wall radius in au
    out_dep : float
        depletion of the outer region
    label : string
        what you want written in the image
    background : float
        background level in target images
    empirical_background : boolean (default True)
        Do we find an empirical background from the chip corners?
    """
   
    print("Debugging... planet_temp is: {0:5.1f}".format(planet_temp)) 
    #Parameters that go into the code
    #params = {'dtog':np.exp(x[0]),'gap_depletion':np.exp(x[1]),'r_in':np.exp(x[2]),\
    #        'r_wall':np.exp(x[3]),'inc':x[4],'pa':x[5],'star_x':x[6],'star_y':x[7],\
    #        'star_temp':np.exp(x[8]),'planet_x':x[9], 'planet_y':x[10], 'planet_r':x[11]}
    params = {'dtog':np.exp(x[0]),'gap_depletion1':np.exp(x[1]),'gap_depletion2':np.exp(x[2]),\
            'r_dust':np.exp(x[3]),'r_in':np.exp(x[4]),'r_wall':np.exp(x[5]),'inc':x[6],\
            'pa_sky':x[7],'star_x':x[8],'star_y':x[9],'planet_x':x[10], 'planet_y':x[11], \
            'planet_r':x[12]}
                
    #Target images.
    target_ims = pyfits.getdata(filename,0)
    
    #PSF Library                    
    calib_ims = pyfits.getdata(filename,1)
    
    #Get the pa information for the object from the fits file
    bintab = pyfits.getdata(filename,2)
    pa_vert = bintab['pa'] 
    if not background:
        background = bintab['background'] 
    
    #Flip the target ims so 0,0 is in the bottom left, not the top left
    #Rotate the data so that you undo what the telescope rotation does, so that North is up and East is left
    tgt_ims = []
    for i in range(target_ims.shape[0]):
        f = np.flipud(target_ims[i])
        #r = nd.interpolation.rotate(f, -pa_vert[i], reshape=False, order=1)
        tgt_ims.append(f)
    tgt_ims = np.asarray(tgt_ims)
   
    #Flip the cal ims so 0,0 is in the bottom left, not the top left
    #Rotate the data so that you undo what the telescope rotation does, so that North is up and East is left
    cal_ims = []
    for i in range(calib_ims.shape[0]):
        f = np.flipud(calib_ims[i])
        #r = nd.interpolation.rotate(f, -pa_vert[i], reshape=False, order=1)
        cal_ims.append(f)
    cal_ims = np.asarray(cal_ims)
    
    #Resample onto half pixel size and Fourier transform.
    #FIXME: This really *shouldn't* be done at every iteration of the Monte-Carlo loop!
    cal_ims_ft = ft_and_resample(cal_ims, empirical_background=empirical_background)
    
    #read in the star_only image for comparison
    imag_obj_star=r3.image.readImage('image_star.out') 
    im_star=imag_obj_star.image[:,:,0]
    
    #Calculate the sum of the star only image for comparison later
    star_sum = np.sum(im_star)
    
    #----------------------------------------------------------------------------------------

    #Create our working directory
    pid_str = str(os.getpid())
    print("AAAAA " + pid_str)
    shutil.rmtree(pid_str, ignore_errors=True)
    #time.sleep(10)
    os.makedirs(pid_str)
    #time.sleep(10)
    os.chdir(pid_str)
    
    #FIXME: This **really** shouldn't go here but should be its own routine.
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
    #if kappa == "['carbon']":
    #    os.system('cp ../dustkappa_carbon.inp .')
    #if kappa == "['56e-3_pah']":
    #    os.system('cp ../dustkappa_56e-3_pah.inp .')
    
    k =  kappa.split("'")[1]
    kappa_str = 'cp ../dustkappa_'+k+'.inp .'
    os.system(kappa_str)
    
    r3.analyze.writeDefaultParfile('ppdisk')
    
    #dodgy fix - need priors instead
    #Make corrections if the r_in and r_wall are around the wrong way
    #if params['r_in'] > params['r_wall']:
    #    r_in = params['r_in']
    #    r_wall = params['r_wall']
    #    params['r_wall'] = r_in
    #    params['r_in'] = r_wall
    
    #Convert parameters to RadMC3D strings
    r_in = '{0:7.9f}*au'.format(params['r_dust'])
    gapin  = '[0.0*au, {0:7.9f}*au, {1:7.9f}*au]'.format(params['r_in'],params['r_wall'])
    gapout = '[{0:7.9f}*au, {1:7.9f}*au, {2:7.9f}*au]'.format(params['r_in'],params['r_wall'],out_wall)
    gap_depletion = '[{0:10.9e}, {1:10.9e}, {2:10.9e}]'.format(params['gap_depletion1'],params['gap_depletion2'],out_dep)
    dusttogas_str = "{0:8.6f}".format(params['dtog'])
    mdisk_str = '[{0:9.7f}*ms]'.format(mdisk)
    #dodgy fix - need priors instead
    #testing to see if this is where the bug is
    #if params['r_wall'] > 60.:
    #    params['r_wall']=60.
    x_bound = '[{0:7.3f}*au,{1:7.3f}*au, ({1:7.3f}+0.1)*au, {2:7.3f}*au, {2:7.3f}*1.1*au, 100*au]'.format(params['r_dust'],params['r_in'],params['r_wall'])
    #n_x = [5., 20., 30., 20., 40.]
    #n_z = 60
    if params['planet_r'] != 0.0:
        star_pos = '[[{0:7.9e}*au,{1:7.9e}*au,0.0],[{2:7.9e}*au,{3:7.9e}*au,0.0]]'.format(params['star_x'],params['star_y'],params['planet_x'],params['planet_y'])
        star_temp = '[{0:7.3f}, {1:7.3f}]'.format(star_temp,planet_temp)
        mass = '[{0:7.3f}*ms, {1:7.3f}*ms]'.format(star_m,planet_mass)
        radii = '[{0:7.3f}*rs, {1:7.3f}*rs]'.format(star_r,params['planet_r']) 
        if Kurucz:
            staremis_type = '["kurucz","kurucz"]'
        else: 
            staremis_type = '["blackbody","blackbody"]' 
    else:
        star_pos = '[{0:7.9e}*au,{1:7.9e}*au,0.0]'.format(params['star_x'],params['star_y'])
        star_temp = '[{0:7.3f}]'.format(star_temp)
        mass = '[{0:7.3f}*ms]'.format(star_m)
        radii = '[{0:7.3f}*rs]'.format(star_r)
        if Kurucz:
            staremis_type = '["kurucz"]'
        else:
            staremis_type = '["blackbody"]'
       
    #edit the problem parameter file
    r3.setup.problemSetupDust('ppdisk', binary=False, mstar=mass, tstar=star_temp, rstar=radii,\
                                pstar=star_pos, dustkappa_ext=kappa, gap_rin=gapin,\
                                gap_rout=gapout, gap_drfact=gap_depletion, dusttogas=dusttogas_str,\
                                rin=r_in,nphot=nphot,nphot_scat=nphot_scat, nx=n_x, xbound=x_bound,\
                                nz=n_z,ny=n_y, srim_rout=1.0, staremis_type=staremis_type,mdisk=mdisk_str,\
                                kurucz_dir=kurucz_dir)
    # run the thermal monte carlo
    grep_output=0
    ntries_mctherm=0
    while grep_output == 0:
        ntries_mctherm+=1
        if ntries_mctherm > 10:
            raise UserWarning("mctherm isn't working on pid: " + pid_str)
        os.system('radmc3d mctherm > mctherm.out') 
        grep_output=os.system('grep ERROR mctherm.out')
        
    #Create the image
    npix_mod = 256
    #The size per pixel in the image is pxsize * dist / 2. 
    #This is because a distance is a conversion of AU per arcsec, and we are subsampling
    #by a factor of 2.
    r3.image.makeImage(npix=npix_mod, sizeau=pxsize*dist/2*npix_mod, wav=wav_in_um, incl=params['inc'], posang=0.)

    imag_obj=r3.image.readImage('image.out') 
    im=imag_obj.image[:,:,0]
    
    #calculate the sum of the model image to use as comparison for SED
    intensity = np.sum(im)
    
    #compare the star and the star+disc -> This is the SED Parameter
    model_sed_ratio = intensity/star_sum
    
    #Inclination angle, detailed disk properties can only come from RADMC-3D
    #Pa to add to the model image PA. Note that this is instrument (not sky) PA.
    #this PA stuff has been fixed now
    
    # Define model type for if making model chi txt
    model_type = str(params['dtog']) + ',' + str(params['gap_depletion1']) + ','  + \
                 str(params['gap_depletion2']) + ',' + str(params['r_dust']) + ','+ str(params['r_in']) + ',' \
                 + str(params['r_wall']) + ',' + str(params['inc']) + ',' + str(params['pa_sky'])\
                 + ',' + str(params['star_x']) + ',' + str(params['star_y']) + ',' + \
                 str(params['planet_x']) + ',' + str(params['planet_y']) + ',' + str(params['planet_r'])\
                 + ',' + str(model_sed_ratio)
    
    model_chi_txt=''
    
    #This line call Keck tools
    chi_tot = rotate_and_fit(im, pa_vert, params['pa_sky'],cal_ims_ft,tgt_ims, model_type, model_chi_txt,\
               plot_ims=plot_ims,save_im_data=save_im_data, make_sed=make_sed,paper_ims=paper_ims,\
               label=label,north_ims=north_ims, rotate_present=rotate_present, bgnd=background)
    
    #This is "cd .."
    os.chdir(os.pardir)
                    
    #Clean up 
    #shutil.rmtree(pid_str, ignore_errors=True)
    
    if remove_directory:
        shutil.rmtree(pid_str)
    else:
        print("*** Figures saved in " + pid_str + " ***")
    
    #This line needs to be edited to reflect the result when not using the full image to 
    #calculate the chi^2
    print("*** Computed chi-squared {0:7.1f} for thread {1:s} ***".format(chi_tot/np.prod(target_ims.shape),pid_str))
    
    #Return log likelihood
    lnlike = -1*(0.5*chi_tot/temperature +((np.log10(data_sed_ratio)-np.log10(model_sed_ratio))**2/(2*sed_ratio_uncert**2)))
    print("*** Computed log likelihood {0:7.1f} for thread {1:s} ***".format(lnlike,pid_str))

    c = open('chain'+pid_str+'.txt','a')
    c.write(str(lnlike) + ',' + model_type + '\n')
    c.close()
        
    return lnlike
'''
def lnprior(theta):
    #params = {'dtog':np.exp(x[0]),'gap_depletion1':np.exp(x[1]),'gap_depletion2':np.exp(x[2]),\
    #        'r_in':np.exp(x[3]),'r_wall':np.exp(x[4]),'inc':x[5],'pa_sky':x[6],'star_x':x[7],\
    #        'star_y':x[8],'planet_x':x[9], 'planet_y':x[10], 'planet_r':x[11]}
    params['r_in'], params['r_wall'] = theta
    if params['r_in'] < params['r_wall']:
        return 0.0
    return -np.inf
    
def lnprob(theta, x, temperature=10000.0, filename='good_ims.fits',nphot="long(4e4)",\
    nphot_scat="long(2e4)", remove_directory=True, star_r=2.0, star_m=2.0, planet_mass=0.001,\
    planet_temp=1500.0, dist=120.0, pxsize=0.01, wav_in_um=3.776, mdisk=0.0001, r_dust=0.3,\
    star_temp=9000.0, kappa = "['carbon']", Kurucz= True, plot_ims=False, save_im_data=False, make_sed=False):
    lp = lnprior(theta)
    if not np.isinfinite(lp):
        return -np.inf
    return lp + lnprob_conv_disk_radmc3d()

def lnprob(theta, x, kwargs=kwargs):
    lp = lnprior(theta)
    if not np.isinfinite(lp):
        return -np.inf
    return lp + lnprob_conv_disk_radmc3d()
    
def lnprob(args):
    return lnprob_conv_disk_radmc3d(args)
    
'''

def lnprior(x, out_wall):
    params = {'dtog':np.exp(x[0]),'gap_depletion1':np.exp(x[1]),'gap_depletion2':np.exp(x[2]),\
            'r_dust':np.exp(x[3]),'r_in':np.exp(x[4]),'r_wall':np.exp(x[5]),'inc':x[6],\
            'pa_sky':x[7],'star_x':x[8],'star_y':x[9],'planet_x':x[10], 'planet_y':x[11],\
            'planet_r':x[12]}
    amr_safe_frac = 1.2
    if params['r_in']   > amr_safe_frac*params['r_dust'] and \
       params['r_wall'] > amr_safe_frac*params['r_in'] and \
       out_wall         > amr_safe_frac*params['r_wall'] and \
       params['r_dust'] > 0.1 and \
       params['dtog']   < 5000. and \
       0. <= params['inc'] <= 360. and \
       0. <= params['pa_sky'] <= 360. and \
       np.sqrt(params['star_x']**2+params['star_y']**2) <= 1. and \
       (params['planet_r'] == 0. or params['planet_r'] > 0.02):
#       params['r_in'] < params['r_wall'] and params['r_wall'] < out_wall and 
#       params['r_wall'] > params['r_in']+0.1 and params['r_dust'] < params['r_in'] and 
        #np.sqrt(params['star_x']**2+params['star_y']**2) <= 1. and 
        #np.round(params['r_dust'],3) != np.round(params['r_in'],3):
        return 0.0
    return -np.inf
   
def lnprob(x, temperature=10000.0, filename='IRS48_ims.fits',nphot="long(4e4)",\
    nphot_scat="long(2e4)", remove_directory=True, star_r=2.0, star_m=2.0, planet_mass=0.001,\
    planet_temp=1500.0, dist=120.0, pxsize=0.01, wav_in_um=3.776, mdisk=0.0001,\
    star_temp=9000.0, kappa = "['carbon']", Kurucz= True, plot_ims=False, save_im_data=False, \
    make_sed=False, data_sed_ratio = 8.672500426996962, sed_ratio_uncert=0.01, out_wall = 60.,\
    out_dep = 1e-1, paper_ims=False, label='',north_ims=False, rotate_present = False):
    #planet_temp=1500.0, dist=120.0, pxsize=0.01, wav_in_um=3.776, mdisk=0.0001, r_dust=0.3,\
    lp = lnprior(x, out_wall)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnprob_conv_disk_radmc3d(x, temperature=temperature, filename=filename,\
    nphot=nphot, nphot_scat=nphot_scat, remove_directory=remove_directory, star_r=star_r,\
    star_m=star_m, planet_mass=planet_mass, planet_temp=planet_temp, dist=dist, \
    pxsize=pxsize, wav_in_um=wav_in_um, mdisk=mdisk, star_temp=star_temp, \
    kappa = kappa, Kurucz= Kurucz, plot_ims=plot_ims, save_im_data=save_im_data, make_sed=make_sed,\
    data_sed_ratio=data_sed_ratio, sed_ratio_uncert=sed_ratio_uncert, out_wall=out_wall, \
    out_dep=out_dep, paper_ims=paper_ims, label=label, north_ims=north_ims, rotate_present=rotate_present)
    #pxsize=pxsize, wav_in_um=wav_in_um, mdisk=mdisk, r_dust=r_dust, star_temp=star_temp, \
    #return lp + lnprob_conv_disk_radmc3d(x, temperature=10000.0, filename='IRS48_ims.fits',nphot="long(4e4)",\
    #nphot_scat="long(2e4)", remove_directory=True, star_r=2.0, star_m=2.0, planet_mass=0.001,\
    #planet_temp=1500.0, dist=120.0, pxsize=0.01, wav_in_um=3.776, mdisk=0.0001, r_dust=0.3,\
    #star_temp=9000.0, kappa = "['carbon']", Kurucz= True, plot_ims=False, save_im_data=False, make_sed=False)

#Here is some code that will run with %run but not import.
if __name__ == "__main__":
    #nwalkers is set to be twice the number of parameters - should make this automatic
    nwalkers = 26
    print('nwalkers=',nwalkers)
    threads = multiprocessing.cpu_count()
    #set parameters to 0.0 if you don't want them investigated
    ipar = np.array([np.log(6.894e-3),np.log(6e-6),np.log(3.012e-3),np.log(11.22),np.log(22.13),48.85,129.5,0.0,0.0,0.0,0.0,0.0])
    #set parameter in cloud to zero to not investigate it
    ipar_sig = np.array([.01,.01,.01,.01,.01,0.1,0.1,0.1,0.1,0.1,0.1,0.001])
    ndim = len(ipar)
    #Could use parameters of random.normal instead of below. But Mike likes this way.
    p0 = [ipar + np.random.normal(size=ndim)*ipar_sig for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,threads=threads)
    sampler.run_mcmc(p0,5)
    '''
    c = open('chainfile.pkl','r')
    old_prob,old_chain = pickle.load(c)
    c.close()
    #define starting point as the last model of the last thread from the previous mcmc
    ipar = old_chain[-1,-1]
    #make the starting cloud smaller
    ipar_sig = np.array([.01,.03,.01,.001,.001,0.1,0.1])
    ndim = len(ipar)
    #Could use parameters of random.normal instead of below. But Mike likes this way.
    p0 = [ipar + np.random.normal(size=ndim)*ipar_sig for i in range(nwalkers)]
    p0 = old_chain
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,pool=pool)
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,threads=threads, args=[old_chain, old_prob])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,threads=threads)
    #sampler.lnprobability = old_prob
    #sampler.chain = old_chain
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d, args=[temperature=170.0])
    sampler.run_mcmc(p0,20)
    '''
    chainfile = open('chainfile.pkl','w')
    pickle.dump((sampler.lnprobability,sampler.chain),chainfile)
    chainfile.close()

#Useful things
#np.max(sampler.flatlnprobability)
#np.argmax(sampler.flatlnprobability)
#sampler.flatchain[np.argmax(sampler.flatlnprobability)]
#np.exp(sampler.flatchain[np.argmax(sampler.flatlnprobability)][0:5])
 
  
