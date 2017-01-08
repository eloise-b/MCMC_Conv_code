'''

Copyright Eloise Birchall, Mike Ireland, Australian National University
eloise.birchall@anu.edu.au

Code that rotates the model imgae and compares it with the data, also has option to make
images of the models and the data and the residuals.

'''

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
import opticstools as ot
import pdb
import os
from os.path import exists
import pickle
#from image_script import *
#Have plt.ion commented out so that the code doesn't generate the image windows and just 
#writes the images directly to file, this should make the code run for a shorter time
#plt.ion()

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

def arcsinh_plot(im, stretch, asinh_vmax=None, asinh_vmin=None, extent=None, im_name='arcsinh_im.png', \
    scale_val=None, im_label=None):
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
    
    
    if im_label:
        plt.clf()
        plt.imshow(stretched_im, interpolation='nearest',cmap=cm.cubehelix, extent=extent, vmin=vmin, vmax=vmax)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)        
        plt.xlabel('Offset (")',fontsize=20)
        plt.ylabel('Offset (")',fontsize=20)
        #plt.title(im_title)
        ticks = np.linspace(vmin,vmax,6)
        cbar = plt.colorbar(ticks=ticks, pad=0.0)
        cbar.set_label('I/I'+r'$_{max}$',size=20)
        #Note that the following line doesn't work in interactive mode.
        if stretch <= 0.001:
            fmt_string = "{0:5.3f}"
        else:
            fmt_string = "{0:5.2f}"
        cbar.ax.set_yticklabels([fmt_string.format(y) for y in stretch*np.sinh(ticks)])
        cbar.ax.tick_params(labelsize=15)
        plt.text(-0.6,0.6,im_label,color='white',ha='left',va='center',fontsize=20)
        plt.savefig(im_name, bbox_inches='tight')
        plt.clf()
    
    else:    
        plt.clf()
        plt.imshow(stretched_im, interpolation='nearest',cmap=cm.cubehelix, extent=extent, vmin=vmin, vmax=vmax)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Offset (")', fontsize=20)
        plt.ylabel('Offset (")', fontsize=20)
        ticks = np.linspace(vmin,vmax,6)
        cbar = plt.colorbar(ticks=ticks, pad=0.0)
        cbar.set_label('I/I'+r'$_{max}$',size=20)
        #Note that the following line doesn't work in interactive mode.
        if stretch <= 0.001:
            fmt_string = "{0:5.3f}"
        else:
            fmt_string = "{0:5.2f}"
        cbar.ax.set_yticklabels([fmt_string.format(y) for y in stretch*np.sinh(ticks)])
        cbar.ax.tick_params(labelsize=15)
        plt.savefig(im_name, bbox_inches='tight')
        plt.clf()


#-------------------------------------------------------------------------------------
def rotate_and_fit(im, pa_vert, pa_sky ,cal_ims_ft,tgt_ims,model_type, model_chi_txt='',plot_ims=True,
    preconvolve=True, pxscale=0.01, save_im_data=True, make_sed=True, paper_ims=True, label='',
    model_chi_dir = '/Users/eloisebirchall/Documents/Uni/Masters/radmc-3d/IRS_48_grid/MCMC_stuff/'):
    """Rotate a model image, and find the best fit. Output (for now!) 
    goes to file in the current directory.
    
    Parameters
    ----------
    im: numpy array
        The image from radMC3D, which has double the sampling of the data.
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
    mode_type: string
        A string to pre-pend to each line of the saved model chi-squared file.
    Returns
    -------
    chi2:
        Chi-squared. Not returned if model_chi_txt has non-zero length.
    """
    #Set constants    
    pixel_std = 300.0 #Rough pixel standard deviation

    mod_sz = im.shape[0]
    sz = tgt_ims.shape[1]
    ntgt = tgt_ims.shape[0]
    ncal = cal_ims_ft.shape[0]
    extent = [-pxscale*sz/2, pxscale*sz/2, -pxscale*sz/2, pxscale*sz/2]
    stretch=0.01
    mcmc_stretch=1e-4
    
    #the chip pa to be used
    pa =[]
    for p in range(len(pa_vert)):
        pa_c = pa_sky - pa_vert[p] + 360.
        pa.append(pa_c)
    #-------------------------------------------------
    #Convolve the model image with a kernel to maintain flux conservation on rotation.
    if (preconvolve):
        #Gaussian kernel - lets slightly convolve the image now to avoid issues when we
        #rotate and Fourier transform.
        kernel = np.array([[.25,.5,.25],[.5,1,.5],[.25,.5,.25]])
        kernel /= np.sum(kernel)
        im = nd.filters.convolve(im,kernel)                    
    
    #-------------------------------------------------
    # Do the rotation Corresponding to the position angle input.  
    rotated_ims = []
    rotated_ims_ft = []
    for i in range(ntgt):
        rotated_image = nd.interpolation.rotate(im, pa[i], reshape=False, order=1)
        if plot_ims:
            arcsinh_plot(rotated_image, mcmc_stretch, im_name='mcmc_im_'+str(i)+'.png', extent=extent)
        rotated_image = rotated_image[mod_sz/2 - sz:mod_sz/2 + sz,mod_sz/2 - sz:mod_sz/2 + sz]
        rotated_image_ft = np.fft.rfft2(np.fft.fftshift(rotated_image))
        rotated_ims.append(rotated_image)
        rotated_ims_ft.append(rotated_image_ft)
    rotated_image = np.array(rotated_ims)
    rotated_image_ft = np.array(rotated_ims_ft)
    if plot_ims:
        arcsinh_plot(np.average(rotated_image, axis=0), mcmc_stretch, im_name='rot_im.eps', extent=extent)
    if paper_ims:
        arcsinh_plot(np.average(rotated_image, axis=0), mcmc_stretch, im_label=label+'Model', im_name='rot_im_paper.eps', extent=extent)
    
    #Output the model rotated image if needed.
    #if plot_ims:
    #    arcsinh_plot(rotated_image, mcmc_stretch, im_name='mcmc_im.png', extent=extent)
    
    #Chop out the central part. Note that this assumes that mod_sz is larger than 2*sz.
    #rotated_image = rotated_image[mod_sz/2 - sz:mod_sz/2 + sz,mod_sz/2 - sz:mod_sz/2 + sz]
    #rotated_image = ot.rebin(rotated_image, (mod_sz/2,mod_sz/2))[mod_sz/4 - sz/2:mod_sz/4 + sz/2,mod_sz/4 - sz/2:mod_sz/4 + sz/2]
    #rotated_image_ft = np.fft.rfft2(np.fft.fftshift(rotated_image))
    #conv_ims = np.empty( (ncal*ntgt,2*sz,2*sz) )

    #Convolve all the point-spread-function images with the model.
    #for j in range(ncal):
    #    conv_ims[j,:,:] = np.fft.irfft2(cal_ims_ft[j]*rotated_image_ft)
     
    #Calculating the best model and chi squared    
    #Go through the target images, find the best fitting calibrator image and record the chi-sqaured.
    chi_squared = np.empty( (ntgt,ncal) )
    best_model_ims = np.empty( (ntgt,sz,sz) )
    best_chi2s = np.empty( ntgt )
    best_convs = np.empty( ntgt, dtype=np.int)
    tgt_sum = np.zeros( (sz,sz) )
    model_sum = np.zeros( (sz,sz) )
    for n in range(ntgt):
        ims_shifted = np.empty( (ncal,sz,sz) )
        #Find the peak for the target
        xypeak_tgt = np.argmax(tgt_ims[n])
        xypeak_tgt = np.unravel_index(xypeak_tgt, tgt_ims[n].shape)
        conv_ims = np.empty( (ncal,2*sz,2*sz) )
        for j in range(ncal):
            conv_ims[j,:,:] = np.fft.irfft2(cal_ims_ft[j]*rotated_image_ft[n])
            #Do a dodgy shift to the peak.
            xypeak_conv = np.argmax(conv_ims[j])
            xypeak_conv = np.unravel_index(xypeak_conv, conv_ims[j].shape)
            
            #Shift this oversampled image to the middle.
            bigim_shifted = np.roll(np.roll(conv_ims[j],2*xypeak_tgt[0]-xypeak_conv[0],axis=0),2*xypeak_tgt[1]-xypeak_conv[1],axis=1)
            
            #Rebin the image.
            ims_shifted[j] = ot.rebin(bigim_shifted,(sz,sz))
            
            #Normalise the image
            ims_shifted[j] *= np.sum(tgt_ims[n])/np.sum(ims_shifted[j])
            
            #Now compute the chi-squared!
            chi_squared[n,j] = np.sum( (ims_shifted[j] - tgt_ims[n])**2/pixel_std**2 )
        
        #Find the best shifted calibrator image (convolved with model) and save this as the best model image for this target image.
        best_conv = np.argmin(chi_squared[n])
        best_chi2s[n] = chi_squared[n,best_conv]
        best_model_ims[n] = ims_shifted[best_conv]
        best_convs[n] = best_conv
        print("Tgt: {0:d} Cal: {1:d}".format(n,best_conv))
        
        #Create a shifted residual image.
        tgt_sum += np.roll(np.roll(tgt_ims[n], sz//2 - xypeak_tgt[0], axis=0), 
                                               sz//2 - xypeak_tgt[1], axis=1)
        model_sum += np.roll(np.roll(best_model_ims[n], sz//2 - xypeak_tgt[0], axis=0), 
                                                        sz//2 - xypeak_tgt[1], axis=1)
        
        if plot_ims:
            #plot the stretched version of the best model image
            arcsinh_plot(best_model_ims[n], stretch, im_name = 'model_stretch_' + str(n) + '.png', extent=extent)
            #plot the best model images, linear scaling.
            plt.clf()
            plt.imshow(best_model_ims[n], interpolation='nearest', extent=extent)
            im_name = 'model_im_' + str(n) + '.png'
            plt.savefig(im_name, bbox_inches='tight')
            plt.clf()
            plt.imshow(tgt_ims[n]-best_model_ims[n], interpolation='nearest',cmap=cm.cubehelix, extent=extent)
            plt.colorbar(pad=0.0)
            stretch_name = 'target-model_' + str(n) + '.png'
            plt.savefig(stretch_name, bbox_inches='tight')
            plt.clf()
            #generate_images(best_model_ims,n)

    if plot_ims:
        arcsinh_plot(tgt_sum, stretch, asinh_vmin=-2, im_name='target_sum.eps', extent=extent)
        arcsinh_plot(model_sum, stretch, asinh_vmin=-2, im_name='model_sum.eps', extent=extent)
        arcsinh_plot(tgt_sum-model_sum, stretch, im_name = 'resid_sum.eps', extent=extent, scale_val=np.max(tgt_sum))
        plt.imshow(tgt_sum-model_sum, interpolation='nearest', extent=extent, cmap=cm.cubehelix)
        plt.colorbar(pad=0.0)
        plt.savefig('residual.eps',bbox_inches='tight')
        plt.clf()
    
    if paper_ims:
        arcsinh_plot(tgt_sum, stretch, asinh_vmin=0, im_label='Data', im_name='target_sum_paper.eps', extent=extent)
        arcsinh_plot(model_sum, stretch, asinh_vmin=0, im_label=label+'Convolved Model', im_name='model_sum_paper.eps', extent=extent)
        arcsinh_plot(tgt_sum-model_sum, stretch, im_label=label+'Residual, Target - Model', im_name = 'resid_sum_paper.eps', extent=extent, scale_val=np.max(tgt_sum))
        #arcsinh_plot(tgt_sum/model_sum, stretch, im_label=label+'Ratio, Target/Model', im_name = 'ratio_paper.eps', extent=extent)#, scale_val=np.max(tgt_sum))        
        plt.imshow(tgt_sum/model_sum, interpolation='nearest', extent=extent, cmap=cm.cubehelix)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)        
        plt.xlabel('Offset (")',fontsize=20)
        plt.ylabel('Offset (")',fontsize=20)
        cbar = plt.colorbar(pad=0.0)
        cbar.ax.tick_params('Data/Model',labelsize=15)
        plt.text(-0.6,0.6,'Ratio',color='orange',ha='left',va='center',fontsize=20)
        plt.savefig('ratio_paper.eps', bbox_inches='tight')
        plt.clf()
        plt.imshow(tgt_sum/model_sum, interpolation='nearest', extent=extent, cmap=cm.PiYG)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)        
        plt.xlabel('Offset (")',fontsize=20)
        plt.ylabel('Offset (")',fontsize=20)
        cbar = plt.colorbar(pad=0.0)
        cbar.ax.tick_params('Data/Model',labelsize=15)
        plt.text(-0.6,0.6,'Ratio',color='white',ha='left',va='center',fontsize=20)
        plt.savefig('ratio_paper_2.eps', bbox_inches='tight')
        plt.clf()
        
    #Save the final image data as a pickle, so that it can be read by another code to make
    #images for a paper later
    if save_im_data:
        tgt_file = open('tgt_sum.pkl','w')
        pickle.dump(tgt_sum,tgt_file)
        tgt_file.close()
        model_file = open('model_sum.pkl','w')
        pickle.dump(model_sum,model_file)
        model_file.close()
        res_sum = tgt_sum-model_sum
        res_file = open('res_sum.pkl','w')
        pickle.dump(res_sum,res_file)
        res_file.close()
        rot_mod_file = open('rot_mod.pkl','w')
        pickle.dump(rotated_image,rot_mod_file)
        rot_mod_file.close()
    
    if make_sed:
        #SED stuff
        os.system('radmc3d sed')
        #os.system('radmc3d sed incl 55.57 phi 130.1')  
        #os.system('radmc3d spectrum loadlambda incl 55.57 phi 130.1')           
        
        spec = np.loadtxt('spectrum.out', skiprows=3)
        
        #Plot the SED
        #define c in microns
        c = 2.99792458*(10**10)
        nu = c / spec[:,0]
        plt.loglog(spec[:,0],nu*spec[:,1], label='model')
        plt.axis((1e-1,1e4,1e-16,1e-6))
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('nu*F_nu')
        plt.legend(loc='best')
        title = 'SED'
        plt.title(title)
        name = 'SED.eps'
        plt.savefig(name)
        plt.clf()
        
    
    #TESTING: save best_chi2s and PSFs.
    #np.savetxt('best_chi2s.txt', best_chi2s)
    #np.savetxt('best_psfs.txt', best_convs, fmt="%d")
    
    #add all these chi-squared values together
    total_chi = np.sum(best_chi2s)
    chi_tot = str(total_chi)
    #Calculate the normalised chi squared
    norm_chi = total_chi/(128.*128.*ntgt)
    norm_chi = str(norm_chi)
    #----------------------------------------------------------------
    #Write the model info and chi-squareds to file
    #record sum of all the chi-squareds for this model and record it with the model type
    #write to a file with model name
    if len(model_chi_txt) > 0:
        top_layer_model_chi = model_chi_dir + model_chi_txt
        model_chi_file = open(top_layer_model_chi, 'a+')
        model_chi_file.write(model_type + str(params['pa']) +',' + chi_tot + ',' + norm_chi + '\n')
        model_chi_file.close()
    else:
        return total_chi
                         
