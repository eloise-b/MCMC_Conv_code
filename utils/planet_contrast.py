import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import radmc3dPy as r3
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../pynrm')) #??? Should be in PYTHONPATH!!!
import mcinmc
import pickle
import opticstools as ot

#'irs48' then 'hd169142'
for star in ['irs48','hd169142']:
    if star == 'irs48':
        plt.clf()
        pa_sky = 96.0
        filename = star + '/IRS48_ims.fits'
        label = 'IRS 48'
    else:
        pa_sky = 15.0
        filename = star + '/good_ims_HD169142.fits'
        label = 'HD 169142'
    
    #Also saved as IRS_allparams.out. Came from image.out from Eloise's 12 Dec email.
    imag_obj=r3.image.readImage(star + '/image.out') 

    im=imag_obj.image[:,:,0]
    #(from the lnprob radmc)

    #from lnprob_conv_disk_radmc3d
    #Target images.
    target_ims = pyfits.getdata(filename,0)

    #PSF Library                    
    calib_ims = pyfits.getdata(filename,1)

    #Get the pa information for the object from the fits file
    bintab = pyfits.getdata(filename,2)
    pa_vert = bintab['pa'] 
    background = bintab['background'] 

    #Flip the target ims so 0,0 is in the bottom left, not the top left
    #Rotate the data so that you undo what the telescope rotation does, so that North is up and East is left
    tgt_ims = []
    for i in range(target_ims.shape[0]):
        f = np.flipud(target_ims[i])
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
    cal_ims_ft = mcinmc.imtools.ft_and_resample(cal_ims, empirical_background=True, resample=False)

    mcinmc.imtools.rotate_and_fit(im, pa_vert, pa_sky, cal_ims_ft, tgt_ims, 'test', model_chi_txt='',plot_ims=False,
        preconvolve=True, pxscale=0.01, save_im_data=True, make_sed=False, paper_ims=False, label='',
        model_chi_dir = './junk/',
        north_ims=False, rotate_present = True, bgnd=[360000.0], gain=4.0, rnoise=10.0, extn='.pdf',
        chi2_calc_hw=40, bgnd_cal=[360000.0], empirical_var=True, empirical_background=True,\
        make_synth=False, filename='', wave_min=3.5e-6, diam=10.0)
    
    #Great. Now we should have created the following files and we'll read them back in.
    rot_conv_sum = pickle.load(open('rot_conv_sum.pkl','rb')) 
    rot_conv_sum /= np.sum(rot_conv_sum)
    tgt_rot_sum = pickle.load(open('tgt_rot_sum.pkl','rb'))
    tgt_rot_sum /= np.sum(tgt_rot_sum)
    cal_rot_sum = pickle.load(open('cal_rot_sum.pkl','rb'))
    cal_rot_sum /= np.sum(cal_rot_sum)

    #We will compute the residual contrast with respect to the *total* flux (not the
    #star flux). A contrast map can be computed by a cross-correlation of the calibrator with
    #the residual.
    resid_rot_sum = tgt_rot_sum - rot_conv_sum 
    contrast_map = np.fft.irfft2(np.fft.rfft2(resid_rot_sum)*np.fft.rfft2(cal_rot_sum))
    contrast_map /= np.sum(cal_rot_sum**2)
    contrast_map = np.fft.fftshift(contrast_map)
    aa = ot.azimuthalAverage(contrast_map**2, center=[64,64], returnradii=True, binsize=1)
    plt.plot(0.01*aa[0][7:45], -np.log10(np.sqrt(aa[1][7:45])*5)*2.5, label=label)

    if star != 'irs48':
        plt.xlabel('Separation (arcsec)')
        plt.ylabel(r'5$\sigma$ Contrast (magnitudes)')
        plt.legend()
        plt.tight_layout()