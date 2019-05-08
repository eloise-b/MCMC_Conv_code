from __future__ import print_function, division
import scipy.ndimage as nd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import radmc3dPy as r3
import astropy.io.fits as pyfits
#import opticstools as ot
#import pdb
import os
from os.path import exists
import pickle
#--- SUBPIXEL ---
import psf_marginalise as pm

def rotate_and_fit(im, pa_vert, pa_sky, cal_ims_ft, tgt_ims, plot_ims=True,
    preconvolve=True, pxscale=0.01, save_im_data=True, paper_ims=False, label='',
    bgnd=[360000.0], gain=4.0, rnoise=10.0, extn='.pdf', chi2_calc_hw=40, bgnd_cal=[360000.0], 
    empirical_var=True, empirical_background=True, filename='', wave_min=3.5e-6, diam=10.0):
    
    """Rotate a model image, and find the best fit. Output (for now!) 
    goes to file in the current directory.
    
    Parameters
    ----------
    im: numpy array
        The image from radMC3D, which has double the sampling of the data. 
        This is what you'd want to change if you were not making a radmc model
    pa_vert : array
        vertical pa from each of the data images
    pa_sky: float
        Position angle on sky that we will combine with vertical to rotate the RADMC image by
    cal_ims_ft: numpy complex array
        Fourier transform of our PSF image libaray
    tgt_ims: numpy array
        Target images to fit to.
    model_chi_txt: string
        File name prefix for a saved model chi-squared.
    model_type: string
        A string to pre-pend to each line of the saved model chi-squared file.
    plot_ims: Bool
        do you want images plotted?
    save_im_data: Bool
        do you want pickles of the images
    make_sed: bool
        do you want an sed?
    paper_ims: Bool
        if true, better figures are made, so that they can be used in a paper
    label: string
        what label do you want on your image - this is passed from lnprob_radmc3d
    north_ims: Bool
        if true, plots some images with north arrow on them *** still needs work
    rotate_present: Bool
        if true makes images that are rotated to have north up and east left, and that could be
        used in a paper.
    bgnd: float or numpy float array
        The background level in each target image.
    bgnd_cal: float or numpy float array
        The background level in each calibrator image (not used!).
    gain: float (optional)
        Gain in electrons per ADU. Default 4.0.
    rnoise: float (optional)
        Readout noise in ADU. Default 10.0
    chi2_calc_hw: int
        Half-width of the region in the target image for which we are calculating chi^2.
    empirical_var: bool (optonal) Default True
        Do we use an empirical calculation of target variance from edge pixels?
    empirical_background: bool (optonal) Default True
        Do we use an empirical calculation of the background level from edge pixels?
    filename : str 
        for when making synth data set, need to know where to get cals, header etc
        
    Returns
    -------
    chi2:
        Chi-squared. Not returned if model_chi_txt has non-zero length.
    """
    mod_sz = im.shape[0]
    sz = tgt_ims.shape[1]
    ntgt = tgt_ims.shape[0]
    ncal = cal_ims_ft.shape[0]
    extent = [-pxscale*sz/2, pxscale*sz/2, -pxscale*sz/2, pxscale*sz/2]
    extent_radec = [pxscale*sz/2, -pxscale*sz/2, -pxscale*sz/2, pxscale*sz/2]
    crop_scale = sz-chi2_calc_hw
    extent_crop = [pxscale*crop_scale/2, -pxscale*crop_scale/2, -pxscale*crop_scale/2, pxscale*crop_scale/2]
    stretch=0.01
    mcmc_stretch=1e-4
    
    if len(bgnd) != ntgt:
        bgnd = bgnd*np.ones(ntgt)

    if len(bgnd_cal) != ncal:
        bgnd_cal = bgnd_cal*np.ones(ncal)
    
    #'''
    #since the data is rotated by itself, this section is no longer needed
    #the chip pa to be used
    pa =[]
    for p in range(len(pa_vert)):
        pa_chip = pa_sky - pa_vert[p]
        if pa_chip <= 0.:
            pa_c = pa_chip + 360.
        elif pa_chip >= 360.:
            pa_c = pa_chip - 360. 
        else:
             pa_c = pa_chip 
        #pa_c = pa_vert[p] + pa_sky -270.  
        pa.append(pa_c)
    #'''
    #-------------------------------------------------
    #Convolve the model image with a kernel to maintain flux conservation on rotation.
    if (preconvolve):
        #Gaussian kernel - lets slightly convolve the image now to avoid issues when we
        #rotate and Fourier transform.
        kernel = np.array([[.25,.5,.25],[.5,1,.5],[.25,.5,.25]])
        kernel /= np.sum(kernel)
        im = nd.filters.convolve(im,kernel)                    
    
    #-------------------------------------------------
    #'''
    #Since the model image only needs to be rotated once, this loop is no longer relevant
    # Do the rotation Corresponding to the position angle input.  
    rotated_ims = []
    rotated_image_ft = np.zeros((ntgt,sz,sz//2+1), dtype=complex)
    for i in range(ntgt):
        rotated_image = nd.interpolation.rotate(im, pa[i], reshape=False, order=1)
#        if plot_ims:
#            arcsinh_plot(rotated_image, mcmc_stretch, im_name='mcmc_im_'+str(i)+extn, extent=extent)
        rotated_image = rotated_image[mod_sz//2 - sz:mod_sz//2 + sz,mod_sz//2 - sz:mod_sz//2 + sz]
        rotated_ims.append(rotated_image)        
        #---  SUBPIXEL ---
        # Find the Fourier transform of this image, then return the array size to the 
        # same as the target images. Note that this will only work, if mod_sz is
        # greater than sz.
        ft_subpixel = np.fft.rfft2(np.fft.fftshift(rotated_image))
        rotated_image_ft[i,0:sz//2,0:sz//2+1] = ft_subpixel[0:sz//2,0:sz//2+1]
        rotated_image_ft[i,-sz//2:,0:sz//2+1] = ft_subpixel[-sz//2:,0:sz//2+1]

    rotated_image = np.array(rotated_ims)
     
    #Calculating the best model and chi squared    
    #Go through the target images, find the best fitting calibrator image and record the chi-sqaured.
    chi_squared = np.empty( (ntgt,ncal) )
    best_model_ims = np.empty( (ntgt,sz,sz) )
    residual_ims = np.empty( (ntgt,sz,sz) )
    ratio_ims = np.empty( (ntgt,sz,sz) )
    best_chi2s = np.empty( ntgt )
    best_convs = np.empty( ntgt, dtype=np.int)
    tgt_sum = np.zeros( (sz,sz) )
    model_sum = np.zeros( (sz,sz) )
    tgt_rot_sum = np.zeros( (sz,sz) )
    cal_rot_sum = np.zeros( (sz,sz) )
    tgt_match_rot_sum = np.zeros( (sz,sz) )
    rot_best_model_ims = np.empty( (ntgt,sz,sz) )
    rot_residuals = np.empty( (ntgt,sz,sz) )
    rot_ratios = np.empty( (ntgt,sz,sz) )
    rot_conv_sum = np.zeros( (sz,sz) )
    rot_resid_sum = np.zeros( (sz,sz) )
    rot_ratio_sum = np.zeros( (sz,sz) )
    
    #--- SUBPIXEL ---
    #Pre-compute target Fourier transforms and our UV grid.
    sampled_uv, uv = pm.make_uv_grid(sz, diam, wave_min, pxscale)
    tgt_ims_ft = ft_and_resample(tgt_ims, empirical_background=empirical_background, resample=False)
    mod_im_ft = np.zeros_like(tgt_ims_ft[0])
    
    for n in range(ntgt):
        ims_shifted = np.empty( (ncal,sz,sz) )
        #Find the peak for the target
        xypeak_tgt = np.argmax(tgt_ims[n])
        xypeak_tgt = np.unravel_index(xypeak_tgt, tgt_ims[n].shape)
        
        #Giant warning if the peak is too close to the edge.
        if (xypeak_tgt[0]<chi2_calc_hw) or (xypeak_tgt[1]<chi2_calc_hw) or \
            (xypeak_tgt[0]>tgt_ims[n].shape[0]-chi2_calc_hw) or \
            (xypeak_tgt[1]>tgt_ims[n].shape[1]-chi2_calc_hw):
            raise UserWarning("Image too close to the edge! Reduce chi2_calc_hw...")
 
        #FIXME: This expensive calculation (for variance and background)
        #doesn't need to be done every time in e.g. a monte-carlo loop.
        xy = np.meshgrid(np.arange(tgt_ims[n].shape[0]) - xypeak_tgt[0], \
                         np.arange(tgt_ims[n].shape[1]) - xypeak_tgt[1], indexing='ij')
        rr = np.sqrt(xy[0]**2 + xy[1]**2)
        outer_pix = np.where(rr > tgt_ims[n].shape[0]//2)
 
        if empirical_background:
            this_im = tgt_ims[n] - np.median(tgt_ims[n][outer_pix])
        else:
            this_im = tgt_ims[n]
 
        #pixel_var has to either be a single number or a 2D array.
        if empirical_var:
            pixel_var = np.maximum(this_im,0)/gain + np.var(this_im[outer_pix])
        else:
            #Compute the pixel variance.
            pixel_var = (np.maximum(this_im,0) + np.maximum(bgnd[n],0) + rnoise**2)/gain 
        
        for j in range(ncal):
            #--- SUBPIXEL ---
            ftim = cal_ims_ft[j][sampled_uv]*rotated_image_ft[n][sampled_uv]
            #Find the required tilt in pixels.
            tilt = pm.optimize_tilt(ftim, tgt_ims_ft[n][sampled_uv], uv, scale_flux=True)
            
            #Compute the tilted convolved model at the sampled_uv points
            sampled_mod_ft = pm.optimize_tilt_function(tilt, ftim, None, uv, return_model=True)
            mod_im_ft[sampled_uv] = sampled_mod_ft
            ims_shifted[j] = np.fft.irfft2(mod_im_ft)
    
            #Normalise the image
            ims_shifted[j] *= np.sum(this_im)/np.sum(ims_shifted[j])
                                       
            #Now compute the chi-squared!
            chi_squared[n,j] = np.sum( \
                ((ims_shifted[j] - this_im)**2/pixel_var)\
                [xypeak_tgt[0]-chi2_calc_hw:xypeak_tgt[0]+chi2_calc_hw,xypeak_tgt[1]-chi2_calc_hw:xypeak_tgt[1]+chi2_calc_hw] )
        
        #Find the best shifted calibrator image (convolved with model) and save this as the best model image for this target image.
        best_conv = np.argmin(chi_squared[n])
        best_chi2s[n] = chi_squared[n,best_conv]
        best_model_ims[n] = ims_shifted[best_conv]
        best_convs[n] = best_conv
        print("Tgt: {0:d} Cal: {1:d}".format(n,best_conv))
        residual_ims[n] = this_im-best_model_ims[n]
        ratio_ims[n] = best_model_ims[n]/this_im
        
        #Create a shifted residual image.
        tgt_sum += np.roll(np.roll(this_im, sz//2 - xypeak_tgt[0], axis=0), 
                                               sz//2 - xypeak_tgt[1], axis=1)
        model_sum += np.roll(np.roll(best_model_ims[n], sz//2 - xypeak_tgt[0], axis=0), 
                                                        sz//2 - xypeak_tgt[1], axis=1)
        tgt_shift = np.roll(np.roll(this_im, sz//2 - xypeak_tgt[0], axis=0), sz//2 - xypeak_tgt[1], axis=1)
        tgt_rot_sum += nd.interpolation.rotate(tgt_shift, pa_vert[n], reshape=False, order=1)
        #Create a calibrator rotated sum in the same way. This is calibrator "best_conv", and
        #we have to center it first. For simplicity, lets do it the same way as the target 
        #(strictly only correct in the high contrast regime.
        cal_im_raw = np.fft.irfft2(cal_ims_ft[best_conv])
        xypeak_cal = np.unravel_index(np.argmax(cal_im_raw), cal_im_raw.shape)
        cal_im_raw = np.roll(np.roll(cal_im_raw, sz//2 - xypeak_cal[0], axis=0), sz//2 - xypeak_cal[1], axis=1)
        cal_rot_sum += nd.interpolation.rotate(cal_im_raw, pa_vert[n], reshape=False, order=1)
        #!!! the following line isn't used anymore. It was to look at everything in on-ship units of the first image for testing
        tgt_match_rot_sum += nd.interpolation.rotate(tgt_shift, pa_vert[n]-pa_vert[0], reshape=False, order=1)
        #Make shifted and rotated images
        model_shift = np.roll(np.roll(best_model_ims[n], sz//2 - xypeak_tgt[0], axis=0), 
                                           sz//2 - xypeak_tgt[1], axis=1)
        rot_best_model_ims[n] = nd.interpolation.rotate(model_shift, pa_vert[n], reshape=False, order=1)
        residual_shift = np.roll(np.roll(residual_ims[n], sz//2 - xypeak_tgt[0], axis=0), 
                                           sz//2 - xypeak_tgt[1], axis=1)
        rot_residuals[n] = nd.interpolation.rotate(residual_shift, pa_vert[n], reshape=False, order=1)
        ratio_shift = np.roll(np.roll(ratio_ims[n], sz//2 - xypeak_tgt[0], axis=0), 
                                           sz//2 - xypeak_tgt[1], axis=1)
        rot_ratios[n] = nd.interpolation.rotate(ratio_shift, pa_vert[n], reshape=False, order=1)
        #make sums of the shifted and rotated images so that we can make figures of them later
        rot_conv_sum += rot_best_model_ims[n]
        rot_resid_sum += rot_residuals[n]
        rot_ratio_sum += rot_ratios[n]
        
    
    #if paper_ims:
    #    arcsinh_plot(tgt_sum, stretch, asinh_vmin=0, im_label='Data', im_name='target_sum_paper_labelled'+extn, extent=extent)
    #    arcsinh_plot(tgt_sum, stretch, asinh_vmin=0, im_name='target_sum_paper'+extn, extent=extent)
    #    arcsinh_plot(tgt_match_rot_sum, stretch, asinh_vmin=0, im_name='target_match_rot_sum_paper'+extn, extent=extent)
    #    arcsinh_plot(tgt_rot_sum, stretch, asinh_vmin=0, im_label='Target', im_name='target_rot_sum_paper'+extn, extent=extent)
    #    arcsinh_plot(model_sum, stretch, asinh_vmin=0, im_label=label+'Conv Model', im_name='model_sum_paper'+extn, extent=extent)
    #    arcsinh_plot(tgt_sum-model_sum, stretch, im_label=label+'Residual, D - M', res=True, im_name = 'resid_sum_paper'+extn, extent=extent, scale_val=np.max(tgt_sum))
    #    #these 2 residuals have north up, but rotated first before making the residuals
    #    arcsinh_plot(tgt_rot_sum-rot_conv_sum, stretch, im_label=label+'Residual, D - M', res=True, im_name = 'resid_sum_paper_rot_first'+extn, extent=extent_radec, scale_val=np.max(tgt_sum), x_ax_label='RA Offset (")', y_ax_label='Dec Offset (")', radec=True  )
    #    arcsinh_plot(tgt_rot_sum[sz//2-chi2_calc_hw:sz//2+chi2_calc_hw,sz//2-chi2_calc_hw:sz//2+chi2_calc_hw]-rot_conv_sum[sz//2-chi2_calc_hw:sz//2+chi2_calc_hw,sz//2-chi2_calc_hw:sz//2+chi2_calc_hw], stretch, im_label=label+'Residual, D - M', res=True, im_name = 'resid_sum_paper_rot_first_crop'+extn, extent=extent_crop, scale_val=np.max(tgt_sum), x_ax_label='RA Offset (")', y_ax_label='Dec Offset (")', chi_crop=True)
    #    #plot a model image only rotated by the pa
    #    
    #    plt.clf()
    #    plt.imshow(model_sum/tgt_sum, interpolation='nearest', extent=extent, cmap=cm.PiYG, vmin=0., vmax=2.)
    #    plt.xticks(fontsize=18)
    #    plt.yticks(fontsize=18)        
    #    plt.xlabel('Offset (")',fontsize=23)
    #    plt.ylabel('Offset (")',fontsize=23)
    #    cbar = plt.colorbar(pad=0.0)
    #    cbar.set_label('Model/Data',size=23)
    #    cbar.ax.tick_params(labelsize=18)
    #    plt.text(-0.6,0.6,label+'Ratio',color='black',ha='left',va='top',fontsize=23)
    #    plt.savefig('ratio_paper_2'+extn, bbox_inches='tight')
    #    plt.clf()
    #    #these 2 ratios have north up, but rotated first before making the ratios
    #    plt.imshow(rot_conv_sum/tgt_rot_sum, interpolation='nearest', extent=extent_radec, cmap=cm.PiYG, vmin=0., vmax=2.)
    #    plt.xticks(fontsize=18)
    #    plt.yticks(fontsize=18)      
    #    plt.xlabel('RA Offset (")',fontsize=23)
    #    plt.ylabel('Dec Offset (")',fontsize=23)
    #    cbar = plt.colorbar(pad=0.0)
    #    cbar.set_label('Model/Data',size=23)
    #    cbar.ax.tick_params(labelsize=18)
    #    plt.text(0.6,0.6,label+'Ratio',color='black',ha='left',va='top',fontsize=23)
    #    plt.savefig('ratio_paper_rot_first'+extn, bbox_inches='tight')
    #    plt.clf()
    #    plt.imshow(rot_conv_sum[sz//2-chi2_calc_hw:sz//2+chi2_calc_hw,sz//2-chi2_calc_hw:sz//2+chi2_calc_hw]/tgt_rot_sum[sz//2-chi2_calc_hw:sz//2+chi2_calc_hw,sz//2-chi2_calc_hw:sz//2+chi2_calc_hw], interpolation='nearest', extent=extent_crop, cmap=cm.PiYG, vmin=0., vmax=2.)
    #    plt.xticks(fontsize=18)
    #    plt.yticks(fontsize=18)      
    #    plt.xlabel('RA Offset (")',fontsize=23)
    #    plt.ylabel('Dec Offset (")',fontsize=23)
    #    cbar = plt.colorbar(pad=0.0)
    #    cbar.set_label('Model/Data',size=23)
    #    cbar.ax.tick_params(labelsize=18)
    #    plt.text(0.4,0.4,label+'Ratio',color='black',ha='left',va='top',fontsize=23)
    #    plt.savefig('ratio_paper_rot_first_crop'+extn, bbox_inches='tight')
    #    plt.clf()
        
    
    #add all these chi-squared values together
    total_chi = np.sum(best_chi2s)
    chi_tot = str(total_chi)
    #Calculate the normalised chi squared
    norm_chi = total_chi/(128.*128.*ntgt)
    norm_chi = str(norm_chi)

    return total_chi



                         
def lnprob_conv_disk_radmc3d(temperature=10000.0, filename='good_ims.fits',nphot="long(4e4)",\
    nphot_scat="long(2e4)", remove_directory=True, pxsize=0.01, wav_in_um=3.776, plot_ims=False, save_im_data=False, \
    paper_ims=True, label='', background=None, empirical_background=True, pa_sky=99.0):
#def lnprob_conv_disk_radmc3d(x, temperature=10000.0, filename='good_ims.fits',nphot="long(4e4)",\
#    nphot_scat="long(2e4)", remove_directory=True, star_r=2.0, star_m=2.0, planet_mass=0.001,\
#    planet_temp=1500.0, dist=120.0, pxsize=0.01, wav_in_um=3.776, mdisk=0.0001,\
#    star_temp=9000.0, kappa = "['carbon']", Kurucz= True, plot_ims=False, save_im_data=False, \
#    make_sed=False, data_sed_ratio = 8.672500426996962, sed_ratio_uncert=0.01, out_wall = 60., \
#    out_dep = 1e-1, n_x = [5, 20, 30, 20, 40], n_z = 60, n_y = [10,30,30,10], \
#    paper_ims=False, label='', north_ims=False, rotate_present = False, synth=False, make_synth=False,\
#    kurucz_dir='/Users/mireland/theory/', background=None, empirical_background=True):

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
    pxsize : float
        Pixel size in arcsec. Note that the MCMC image is sampled 2 times higher than this.
    label : string
        what you want written in the image
    synth : boolean
        are you using a syntheti data set?
    background : float
        background level in target images
    empirical_background : boolean (default True)
        Do we find an empirical background from the chip corners?
    pa_sky = rotation of the object on the sky - set this to be what your pa in your simulations is                
    """             
    
    """***This is the stuff that reads in the target images, the calibrators and the pas***"""
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
    cal_ims_ft = ft_and_resample(cal_ims, empirical_background=empirical_background, resample=False)
    
    #read in the model image you want to compare this currently reads in a radmc model as an array
    imag_obj=r3.image.readImage('image.out')
    im=imag_obj.image[:,:,0]
    
    #This line calls im tools
    chi_tot = rotate_and_fit(im, pa_vert, pa_sky, cal_ims_ft,tgt_ims,\
               plot_ims=plot_ims,save_im_data=save_im_data,paper_ims=paper_ims,\
               label=label, bgnd=background, filename=filename)
    
    #Return log likelihood
    lnlike = -1*(0.5*chi_tot/temperature) 
    print("*** Computed log likelihood {0:7.1f}  ***".format(lnlike))
    print("*** Computed chi-squared {0:7.1f}  ***".format(-1*lnlike/80./80./(tgt_ims.shape[0])*2*temperature))
        
    return lnlike

def ft_and_resample(cal_ims, empirical_background=True, resample=True):
    """Create the Fourier transform of a set of images, resampled onto half the
    pixel scale if resample is set to True. """
    #Number of images
    ncal = cal_ims.shape[0] #Number of calibrator images.
 
    #Image size     
    sz = cal_ims.shape[1]

    #Corner pixels
    xy = np.meshgrid(np.arange(sz) - sz//2, np.arange(sz) - sz//2, indexing='ij')
    rr = np.sqrt(xy[0]**2 + xy[1]**2)
    outer_pix = np.where(rr > sz//2)

    #This should come from a library! but to see details, lets do it manually.
    #do the fast fourier transform of the psfs
    if resample:
        cal_ims_ft = np.zeros( (ncal,sz*2,sz+1),dtype=np.complex )
    else:
        cal_ims_ft = np.zeros( (ncal,sz,sz//2+1),dtype=np.complex )
    
    for j in range(ncal):
        if empirical_background:
            this_im = cal_ims[j] - np.median(cal_ims[j][outer_pix])
        else:
            this_im = cal_ims[j]
        cal_im_ft_noresamp = np.fft.rfft2(this_im)
        if resample:
            cal_ims_ft[j,0:sz//2,0:sz//2+1] = cal_im_ft_noresamp[0:sz//2,0:sz//2+1]
            cal_ims_ft[j,-sz//2:,0:sz//2+1] = cal_im_ft_noresamp[-sz//2:,0:sz//2+1]
        else:
            cal_ims_ft[j] = cal_im_ft_noresamp
    return cal_ims_ft
    
def arcsinh_plot(im, stretch, asinh_vmax=None, asinh_vmin=None, extent=None, im_name='arcsinh_im.png', \
    scale_val=None, im_label=None, res=False, north=False, angle=0., x_ax_label = 'Offset (")',\
    y_ax_label = 'Offset (")', radec=False, chi_crop=False, circle=False, circle_x=None, \
    circle_y=None, circle_r=None):
    """A helper routine to make an arcsinh stretched image.
    
    Parameters
    ----------
    
    im: numpy array
        The input image, with bias level of 0 and arbitrary maximum.
    stretch: float
        After division by the maximum (or scale_val), the level to divide the data by. This
        is the boundary between the linear and logarithmic regime.
    asinh_vmax: float
        The maximum value of asinh of the data. Defaults to asinh of the maximum value.
    asinh_vmin: float
        The minimum value of asinh of the data. Defaults to asinh of the minimum value.
    extent: list
        The extent parameter to be passed to imshow.
    im_name: string
        The name of the image file to be saved.
    scale_val: float
        The value to divide the data by. Defaults to max(im)
    im_title: string
        if you want to have the image have a title, but something in here
    res: Boolean
        is the image a residual image?
    north: Bool
        do you want the north arrows plotted on the images
    angle: float
        at what angle should the north arrow be at   
    x_ax_label: string
        what you want the label on the x axis to be
    y_ax_label: string
        what you want the label on the y axis to be  
    chi_crop: Boolean
        is the image one that is cropped for the region chi squared is calc'd in?
        chi crop and radec can't be at the same time
    circle: Bool
        true if you want to plot a circle on your image
    circle_x,y,r: float
        parameters for position and size of circle if circle is true
    """
    if not scale_val:
        scale_val = np.max(im)
    stretched_im = np.arcsinh(im/scale_val/stretch)
 
    if asinh_vmax:
        vmax = asinh_vmax
    else:
        vmax = np.max(stretched_im)
    
    if asinh_vmin:
        vmin = asinh_vmin
    else:
        vmin = np.min(stretched_im)
    
    if north and im_label:
        #angle = pa_vert[8]*(np.pi/180)
        arrow_x1 = -0.4*np.sin(angle)
        arrow_y2 = 0.55*np.cos(angle)
        arrow_y1 = 0.4*np.cos(angle)
        arrow_x2 = -0.55*np.sin(angle)
        #e_arrow_x1 = -0.4*np.sin(angle+(np.pi/2.))
        e_arrow_y2 = -0.15*np.sin(angle)+0.4*np.cos(angle)
        #e_arrow_y1 = 0.4*np.cos(angle+(np.pi/2.))
        e_arrow_x2 = -0.15*np.cos(angle)-0.4*np.sin(angle)
        #e_arrow_y2 = arrow_x2*np.sin((np.pi/2.))+arrow_y2*np.cos((np.pi/2.))
        #e_arrow_x2 = arrow_x2*np.cos((np.pi/2.))-arrow_y2*np.sin((np.pi/2.))
        plt.clf()
        plt.imshow(stretched_im, interpolation='nearest',cmap=cm.cubehelix, extent=extent, vmin=vmin, vmax=vmax)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)        
        plt.xlabel(x_ax_label,fontsize=23)
        plt.ylabel(y_ax_label,fontsize=23)
        #plt.title(im_title)
        ticks = np.linspace(vmin,vmax,6)
        cbar = plt.colorbar(ticks=ticks, pad=0.0)
        if res:
            #cbar.set_label('I/I(data)'+r'$_{max}$',size=23)
            cbar.set_label('I/max(I(data))',size=23)
        else:
            #cbar.set_label('I/I'+r'$_{max}$',size=23)
            cbar.set_label('I/max(I)',size=23)
        #Note that the following line doesn't work in interactive mode.
        if stretch <= 0.001:
            fmt_string = "{0:5.3f}"
        else:
            fmt_string = "{0:5.2f}"
        cbar.ax.set_yticklabels([fmt_string.format(y) for y in stretch*np.sinh(ticks)])
        cbar.ax.tick_params(labelsize=18)
        if chi_crop:
            plt.text(0.4,0.4,im_label,color='white',ha='left',va='top',fontsize=23)
        elif radec:
            plt.text(0.6,0.6,im_label,color='white',ha='left',va='top',fontsize=23)
        else:
            plt.text(-0.6,0.6,im_label,color='white',ha='left',va='top',fontsize=23)
        plt.arrow(arrow_x1, arrow_y1, arrow_x2-arrow_x1, arrow_y2-arrow_y1, fc="red", ec="red")
        plt.arrow(arrow_x1, arrow_y1, e_arrow_x2-arrow_x1, e_arrow_y2-arrow_y1, fc="red", ec="red")
        plt.text(arrow_x2-0.04, arrow_y2+0.01,'N',color='red',ha='right',va='bottom',fontsize=23)
        plt.text(e_arrow_x2-0.04, e_arrow_y2,'E',color='red',ha='right',va='top',fontsize=23)
        plt.savefig(im_name, bbox_inches='tight')
        plt.clf()   
    elif im_label:
        plt.clf()
        if circle:
            circle1 = plt.Circle((circle_x, circle_y), circle_r, color='yellow', fill=False)
            plt.gcf().gca().add_artist(circle1)
        plt.imshow(stretched_im, interpolation='nearest',cmap=cm.cubehelix, extent=extent, vmin=vmin, vmax=vmax)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)        
        plt.xlabel(x_ax_label,fontsize=23)
        plt.ylabel(y_ax_label,fontsize=23)
        #plt.title(im_title)
        ticks = np.linspace(vmin,vmax,6)
        cbar = plt.colorbar(ticks=ticks, pad=0.0)
        if res:
            #cbar.set_label('I/I(data)'+r'$_{max}$',size=23)
            cbar.set_label('I/max(I(data))',size=23)
        else:
            #cbar.set_label('I/I'+r'$_{max}$',size=23)
            cbar.set_label('I/max(I)',size=23)
        #Note that the following line doesn't work in interactive mode.
        if stretch <= 0.001:
            fmt_string = "{0:5.3f}"
        else:
            fmt_string = "{0:5.2f}"
        cbar.ax.set_yticklabels([fmt_string.format(y) for y in stretch*np.sinh(ticks)])
        cbar.ax.tick_params(labelsize=18)
        if chi_crop:
            plt.text(0.4,0.4,im_label,color='white',ha='left',va='top',fontsize=23)
        elif radec:
            plt.text(0.6,0.6,im_label,color='white',ha='left',va='top',fontsize=23)
        else:
            plt.text(-0.6,0.6,im_label,color='white',ha='left',va='top',fontsize=23)
        plt.savefig(im_name, bbox_inches='tight')
        plt.clf()
    elif north:
        #angle = pa_vert[8]*(np.pi/180)
        arrow_x1 = -0.4*np.sin(angle)
        arrow_y2 = 0.55*np.cos(angle)
        arrow_y1 = 0.4*np.cos(angle)
        arrow_x2 = -0.55*np.sin(angle)
        #e_arrow_x1 = -0.4*np.sin(angle+(np.pi/2.))
        e_arrow_y2 = -0.15*np.sin(angle)+0.4*np.cos(angle)
        #e_arrow_y1 = 0.4*np.cos(angle+(np.pi/2.))
        e_arrow_x2 = -0.15*np.cos(angle)-0.4*np.sin(angle)
        plt.clf()
        plt.imshow(stretched_im, interpolation='nearest',cmap=cm.cubehelix, extent=extent, vmin=vmin, vmax=vmax)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)        
        plt.xlabel(x_ax_label,fontsize=23)
        plt.ylabel(y_ax_label,fontsize=23)
        #plt.title(im_title)
        ticks = np.linspace(vmin,vmax,6)
        cbar = plt.colorbar(ticks=ticks, pad=0.0)
        if res:
            #cbar.set_label('I/I(data)'+r'$_{max}$',size=23)
            cbar.set_label('I/max(I(data))',size=23)
        else:
            #cbar.set_label('I/I'+r'$_{max}$',size=23)
            cbar.set_label('I/max(I)',size=23)
        #Note that the following line doesn't work in interactive mode.
        if stretch <= 0.001:
            fmt_string = "{0:5.3f}"
        else:
            fmt_string = "{0:5.2f}"
        cbar.ax.set_yticklabels([fmt_string.format(y) for y in stretch*np.sinh(ticks)])
        cbar.ax.tick_params(labelsize=18)
        plt.arrow(arrow_x1, arrow_y1, arrow_x2-arrow_x1, arrow_y2-arrow_y1, fc="red", ec="red")
        plt.arrow(arrow_x1, arrow_y1, e_arrow_x2-arrow_x1, e_arrow_y2-arrow_y1, fc="red", ec="red")
        plt.text(arrow_x2-0.04, arrow_y2+0.01,'N',color='red',ha='right',va='bottom',fontsize=23)
        plt.text(e_arrow_x2-0.04, e_arrow_y2,'E',color='red',ha='right',va='top',fontsize=23)
        plt.savefig(im_name, bbox_inches='tight')
        plt.clf()
    else:    
        plt.clf()
        plt.imshow(stretched_im, interpolation='nearest',cmap=cm.cubehelix, extent=extent, vmin=vmin, vmax=vmax)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(x_ax_label,fontsize=23)
        plt.ylabel(y_ax_label,fontsize=23)
        ticks = np.linspace(vmin,vmax,6)
        cbar = plt.colorbar(ticks=ticks, pad=0.0)
        if res:
            #cbar.set_label('I/I(data)'+r'$_{max}$',size=23)
            cbar.set_label('I/max(I(data))',size=23)
        else:
            #cbar.set_label('I/I'+r'$_{max}$',size=23)
            cbar.set_label('I/max(I)',size=23)
        #Note that the following line doesn't work in interactive mode.
        if stretch <= 0.001:
            fmt_string = "{0:5.3f}"
        else:
            fmt_string = "{0:5.2f}"
        cbar.ax.set_yticklabels([fmt_string.format(y) for y in stretch*np.sinh(ticks)])
        cbar.ax.tick_params(labelsize=18)
        plt.savefig(im_name, bbox_inches='tight')
        plt.clf()