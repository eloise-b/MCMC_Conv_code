from __future__ import division
 
"""
Copyright Eloise Birchall ANU
eloise.birchall@anu.edu.au

Script to run Radmc3dPy and make a single image of a protoplanetary disk. 
Should be run in ipython --pylab to work best.
"""

# import numpy and other things
from __future__ import print_function, division
import scipy.ndimage as nd
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import radmc3dPy as r3
import astropy.io.fits as pyfits
import opticstools as ot
import pdb
import os
from keck_tools import *
#from image_script import *
from os.path import exists

#Set up parameter file for protoplanetary disk
r3.analyze.writeDefaultParfile('ppdisk')

#DEFINE THE PARAMETERS HERE
dtog = '6.13e-4'
#gapin  = '[0*au, 11.13*au, 23.09*au]'
#gapout = '[11.13*au, 23.09*au, 60*au]'
#gap_depletion = '[2.671e-8, 2.902e-3, 1e-1]'
#r_dust = '0.3*au'
inc = 57.6
pa = 130.6
#r_d = 0.3 #Same as r_dust, but without the *au so that the format is correct for xbound
r_in = 11.33
r_wall = 25.87
#gd1 = 2.671e-8
gd2 = 1.458e-2
r_inau = '{0:7.3f}*au'.format(r_in)
#gapin  = '[0*au, {0:7.3f}*au, {1:7.3f}*au]'.format(r_in,r_wall)
#gapout = '[{0:7.3f}*au, {1:7.3f}*au, 60*au]'.format(r_in,r_wall)
#gap_depletion = '[{0:10.3e}, {1:10.3e}, 1e-1]'.format(gd1,gd2)
gapin  = '[{0:7.3f}*au, {1:7.3f}*au]'.format(r_in,r_wall)
gapout = '[{0:7.3f}*au, 60*au]'.format(r_wall)
gap_depletion = '[{0:10.3e}, 1e-1]'.format(gd2)
x_bound = '[{0:7.3f}*au, {0:7.3f}*1.1*au, {1:7.3f}*au, {1:7.3f}*1.1*au, 100*au]'.format(r_in,r_wall)
#x_bound = '[1.0*au, {0:7.3f}*au, {0:7.3f}*1.1*au, {1:7.3f}*au, {1:7.3f}*1.1*au, 100*au]'.format(r_in,r_wall)
#n_x = [20., 20., 30., 20., 40.]
n_x = [20., 30., 20., 40.]
n_z = 60
#star x and y are zero if no asymmetry
star_x = -0.139
star_y = 0.0386
#vary star temperature instead of having inner disk
star_temp = 10237.0
#mass of star, should remain set
star_m = 2.0
#radius of star, should remain set
star_r = 2.0
#planet x and y, zero if no planet
planet_x = -8.1375
planet_y = 0.0423
#planet temperature, zero if no planet
planet_temp = 1500.0
#planet radius, zero if no planet
planet_r = 0.9575
#planet mass, not important, but set to zero if there is no planet
planet_m = 0.001
mdisk=1e-4

#CHOOSE WHICH PARAMETERS ARE APPROPRIATE, WITH/WITHOUT PLANET
#Defining the parameters so that there can be a planet and/or asymmetry
star_pos = '[[{0:7.3f}*au,{1:7.3f}*au,0.0],[{2:7.3f}*au,{3:7.3f}*au,0.0]]'.format(star_x,star_y,planet_x,planet_y)
star_t = '[{0:7.3f}, {1:7.3f}]'.format(star_temp,planet_temp)
mass = '[{0:7.3f}*ms, {1:7.3f}*ms]'.format(star_m, planet_m)
radii = '[{0:7.3f}*rs, {1:7.3f}*rs]'.format(star_r, planet_r)
staremis_type = '["blackbody","blackbody"]'  
#WITHOUT PLANET
#star_pos = '[{0:7.3f}*au,{1:7.3f}*au,0.0]'.format(star_x,star_y)
#star_t = '[{0:7.3f}]'.format(star_temp)
#mass = '[{0:7.3f}*ms]'.format(star_m)
#radii = '[{0:7.3f}*rs]'.format(star_r)    
#staremis_type = '["blackbody"]'


#edit the problem parameter file
r3.setup.problemSetupDust('ppdisk', binary=False, mstar=mass, tstar=star_temp, rstar=radii,\
                            pstar=star_pos, dustkappa_ext="['carbon']", gap_rin=gapin,\
                            gap_rout=gapout, gap_drfact=gap_depletion, dusttogas=dtog,\
                            rin=r_in, nx=n_x, xbound=x_bound,\
                            nz=n_z, srim_rout=1.0, staremis_type=staremis_type,mdisk=mdisk)
                         
#Run the radmc3d mctherm code
os.system('radmc3d mctherm')	

#Create the image
#make the image set posang to 100 as for IRS 48
npix_mod = 256
r3.image.makeImage(npix=npix_mod, sizeau=0.6*npix_mod, wav=3.776, incl=inc, posang=0.)
#read image
imag = r3.image.readImage()
#plot image, to change colour set cmap=cm.colour_fn_name - default is cmap=cm.gist_gray
r3.image.plotImage(imag, arcsec=True, dpc=120., log=True, maxlog=5, cmap=cm.cubehelix)
#save image
img_name = "model.eps"
#plt.savefig(img_name)
print("image.png saved")
plt.clf()

filename = '../good_ims.fits'
#Images we match to
tgt_ims = pyfits.getdata(filename,0)
ntgt = tgt_ims.shape[0] #Number of target images.

#PSF Library                    
cal_ims = pyfits.getdata(filename,1)
cal_ims = cal_ims[1:] #!!! Mike Hack
ncal = cal_ims.shape[0] #Number of calibrator images.
                    
sz = cal_ims.shape[1]


#This should come from a library! but to see details, lets do it manually.
#do the fast fourier transform of the psfs
cal_ims_ft = np.zeros( (ncal,sz*2,sz+1),dtype=np.complex )
for j in range(ncal):
    cal_im_ft_noresamp = np.fft.rfft2(cal_ims[j,:,:])
    cal_ims_ft[j,0:sz/2,0:sz/2+1] = cal_im_ft_noresamp[0:sz/2,0:sz/2+1]
    cal_ims_ft[j,-sz/2:,0:sz/2+1] = cal_im_ft_noresamp[-sz/2:,0:sz/2+1]

imag_obj = r3.image.readImage('image.out')
im = imag_obj.image[:,:,0]

xypeak_model = np.argmax(im)
xypeak_model = np.unravel_index(xypeak_model, im.shape)

pixel_std = 300.0 #Rough pixel standard deviation

mod_sz = im.shape[0]
sz = tgt_ims.shape[1]
ntgt = tgt_ims.shape[0]
ncal = cal_ims_ft.shape[0]

#!!! Warning the central source flux changes a little. Maybe it is best to start 
#with a (slightly) convolved image. Play with this!

#Gaussian kernel
kernel = np.array([[.25,.5,.25],[.5,1,.5],[.25,.5,.25]])
kernel /= np.sum(kernel)
im = nd.filters.convolve(im,kernel)                    


#Inclination angle, detailed disk properties can only come from RADMC-3D
#Pa to add to the model image PA. Note that this is instrument (not sky) PA.
p= pa

rotated_image = nd.interpolation.rotate(im, p, reshape=False, order=1)
#Chop out the central part. Note that this assumes that mod_sz is larger than 2*sz.
rotated_image = rotated_image[mod_sz/2 - sz:mod_sz/2 + sz,mod_sz/2 - sz:mod_sz/2 + sz]
#rotated_image = ot.rebin(rotated_image, (mod_sz/2,mod_sz/2))[mod_sz/4 - sz/2:mod_sz/4 + sz/2,mod_sz/4 - sz/2:mod_sz/4 + sz/2]
rotated_image_ft = np.fft.rfft2(np.fft.fftshift(rotated_image))
conv_ims = np.empty( (ncal,2*sz,2*sz) )

#Convolve all the point-spread-function images with the model.
for j in range(ncal):
    conv_ims[j,:,:] = np.fft.irfft2(cal_ims_ft[j]*rotated_image_ft)
 
#Calculating the best model and chi squared    
#Go through the target images, find the best fitting calibrator image and record the chi-sqaured.
chi_squared = np.empty( (ntgt,ncal) )
best_model_ims = np.empty( (ntgt,sz,sz) )
res_ims = np.empty( (ntgt,sz,sz) )
best_chi2s = np.empty( ntgt )
best_convs = np.empty( ntgt, dtype=np.int)
peak_coords = np.empty( (ntgt,2) )
for n in range(ntgt):
    ims_shifted = np.empty( (ncal,sz,sz) )
    #Find the peak for the target
    xypeak_tgt = np.argmax(tgt_ims[n])
    xypeak_tgt = np.unravel_index(xypeak_tgt, tgt_ims[n].shape)
    peak_coords[n,:] = np.array(xypeak_tgt)
    
    
    for j in range(ncal):
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
    res_ims[n] = tgt_ims[n]-best_model_ims[n]
    
    #plot the best model images
    plt.clf()
    #'''
    '''
    #plt.imshow(np.log10(best_model_ims[n]), interpolation='nearest',cmap=cm.cubehelix)
    
    #Uncomment/set these for RA and Dec offest as axes (and use extent=[xneg,xpos,yneg,ypos])
    xneg =  -(xypeak_tgt[1])*0.005
    xpos = (128.-xypeak_tgt[1])*0.005
    yneg = -(128.-xypeak_tgt[0])*0.005
    ypos = (xypeak_tgt[0])*0.005
    '''
    #This chunk makes the convolved model image
    plt.imshow(((best_model_ims[n]/np.max(best_model_ims[n]))), interpolation='nearest',cmap=cm.cubehelix)
    #plt.imshow(((best_model_ims[n]/np.max(best_model_ims[n]))), interpolation='nearest',cmap=cm.cubehelix,  extent=[xneg,xpos,yneg,ypos])
    plt.colorbar(label=r'$I/I_{max}$')
    plt.xlabel('RA Offset (")')
    plt.ylabel('Dec Offset (")')
    im_name = 'model_im_' + str(n) + '.eps'
    plt.savefig(im_name)
    plt.clf()
    #plt.imshow(np.arcsinh(best_model_ims[n]), interpolation='nearest',cmap=cm.cubehelix, clim = (3.0,15.0))
    #plt.colorbar()
    #stretch_name = 'model_stretch_' + str(n) + '.png'
    #plt.savefig(stretch_name)
    #plt.clf()
    '''
    #plt.imshow(np.log10(((tgt_ims[n]-best_model_ims[n])/(np.min(tgt_ims[n]-best_model_ims[n])))+1.), interpolation='nearest',cmap=cm.cubehelix)
    #plt.imshow(((tgt_ims[n]-best_model_ims[n])/np.max(tgt_ims[n]-best_model_ims[n])), interpolation='nearest',cmap=cm.cubehelix)
    #Find limits for extent
    plt.imshow(((tgt_ims[n])/np.max(tgt_ims[n])), interpolation='nearest',cmap=cm.cubehelix, extent=[xneg,xpos,yneg,ypos])
    plt.colorbar(label=r'$I/I_{max}$')
    #plt.colorbar.setlabel(r'I/I_max')
    #plt.plot(xypeak_tgt[1], xypeak_tgt[0], 'y*', markersize=10) # this line puts a star on the centre of an image
    t_name = 'target' + str(n) + '.eps'
    plt.xlabel('RA Offset (")')
    plt.ylabel('Dec Offset (")')
    plt.savefig(t_name)
    plt.clf()
    '''
    #This is the code that makes the residual images
    plt.imshow(((tgt_ims[n]-best_model_ims[n])/np.max(tgt_ims[n])), interpolation='nearest',cmap=cm.cubehelix)
    #plt.imshow(((tgt_ims[n]-best_model_ims[n])/np.max(tgt_ims[n])), interpolation='nearest',cmap=cm.cubehelix, extent=[xneg,xpos,yneg,ypos])
    #plt.imshow(((tgt_ims[n]-best_model_ims[n])/np.max(tgt_ims[n]-best_model_ims[n])), interpolation='nearest',cmap=cm.cubehelix)
    #plt.imshow(((tgt_ims[n]-best_model_ims[n])/np.max(tgt_ims[n]-best_model_ims[n])), interpolation='nearest',cmap=cm.cubehelix, extent=[xneg,xpos,yneg,ypos])
    plt.colorbar(label=r'$I/I_{max observed}$')
    #plt.colorbar.setlabel(r'I/I_max')
    plt.plot(xypeak_tgt[1], xypeak_tgt[0], 'y*', markersize=10) # this line puts a star on the centre of an image
    #plt.plot(xypeak_model[1]/2., xypeak_model[0]/2., 'r*', markersize=10)
    t_m_name = 'target-model_star_' + str(n) + '.eps'
    #plt.xlabel('RA Offset (")')
    #plt.ylabel('Dec Offset (")')
    plt.savefig(t_m_name)
    plt.clf()
'''
    
# makes the rotated model image, radmc3d one isn't rotated
plt.imshow((np.arcsinh(rotated_image/np.max(rotated_image)))**0.15, interpolation='nearest',cmap=cm.cubehelix)
#plt.imshow((np.arcsinh(rotated_image/np.max(rotated_image)))**0.15, interpolation='nearest',cmap=cm.cubehelix,extent=[xneg,xpos,yneg,ypos])
plt.colorbar(label=r'arcsinh($I/I_{max}$)')
#plt.xlabel('RA Offset (")')
#plt.ylabel('Dec Offset (")')
im_name = 'rotated.eps'
plt.savefig(im_name)
plt.clf()
'''
'''    
#To make the averaged residual images
xneg =  -(peak_coords[0,1])*0.005
xpos = (128.-peak_coords[0,1])*0.005
yneg = -(128.-peak_coords[0,0])*0.005
ypos = (peak_coords[0,0])*0.005
av = np.average(res_ims[:5], axis=0)
plt.imshow(av/np.max(av),interpolation='nearest',cmap=cm.cubehelix,extent=[xneg,xpos,yneg,ypos])
plt.colorbar(label=r'$I/I_{max}$')
plt.xlabel('RA Offset (")')
plt.ylabel('Dec Offset (")')
plt.savefig('average_5uncentred.eps')
plt.clf()

#making the rolled residuals
peak_coords = peak_coords.astype("int")
shift_res_ims = np.empty ((ntgt,sz,sz))
xneg =  -(peak_coords[5,0])*0.005
xpos = (128.-peak_coords[5,0])*0.005
yneg = -(128.-peak_coords[5,1])*0.005
ypos = (peak_coords[5,1])*0.005
ploty = [[0,64],[128,64]]
plotx = [[64,0],[64,128]]
for j in range(ntgt):
    shift_res_ims[j] = np.roll(np.roll(res_ims[j],peak_coords[j,0]-64,axis=1),peak_coords[j,1]-64,axis=0)
    #shift_res_ims[j] = np.roll(np.roll(res_ims[j],peak_coords[j,0]-peak_coords[5,0],axis=1),peak_coords[j,1]-peak_coords[5,1],axis=0)
    plt.imshow(((shift_res_ims[j])/np.max(shift_res_ims[j])), interpolation='nearest',cmap=cm.cubehelix)
    #plt.plot(peak_coords[j,1], peak_coords[j,0], 'y*', markersize=10)
    plt.colorbar(label=r'$I/I_{max}$')
    #plt.xlabel('RA Offset (")')
    #plt.ylabel('Dec Offset (")')
    name = 'shifted_' + str(j) +'.eps'
    plt.savefig(name)
    plt.clf()
av_res = np.average(shift_res_ims[:], axis=0)
plt.imshow(av_res/np.max(av_res),interpolation='nearest',cmap=cm.cubehelix,extent=[-0.32,0.32,-0.32,0.32])
plt.colorbar(label=r'$I/I_{max}$')
plt.xlabel('RA Offset (")')
plt.ylabel('Dec Offset (")')
plt.savefig('average.eps')
plt.clf()
'''
print('done')