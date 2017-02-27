from __future__ import print_function, division

"""
Code to read in pickles that come from the the lnprob and imtools code, and then make the
relevant images and organise them in such a way that they can go in the paper
"""

import scipy.ndimage as nd
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['fontname'] = "Arial"
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import radmc3dPy as r3
import astropy.io.fits as pyfits
import opticstools as ot
import pdb
import os
from os.path import exists
import pickle
from imtools import *

#read in the pickle files that are output by the main code
file = open('rot_mod.pkl','r')
model_im = pickle.load(file)
file.close()
file = open('model_sum.pkl','r')
conv_im = pickle.load(file)
file.close()
file = open('res_sum.pkl','r')
res_im = pickle.load(file)
file.close()
file = open('tgt_sum.pkl','r')
tgt_im = pickle.load(file)
file.close()

deconv = pyfits.getdata('deconv_image.fits')

#what type of model is it, sym, asym, planet?
label='Sym '
#chi squared of that model type
chi = ''

#constants from imtools so asinhplot still works
pxscale=0.01
mod_sz = model_im.shape[0]
sz = tgt_im.shape[1]
extent = [-pxscale*sz/2, pxscale*sz/2, -pxscale*sz/2, pxscale*sz/2]
stretch=0.01
mcmc_stretch=1e-4

#make the plots
arcsinh_plot(np.average(model_im, axis=0), mcmc_stretch, im_label=label+'Model', im_name='rot_im_paper.eps', extent=extent)
arcsinh_plot(tgt_im, stretch, asinh_vmin=0, im_label='Data', im_name='target_sum_paper_labelled.eps', extent=extent)
arcsinh_plot(tgt_im, stretch, asinh_vmin=0, im_name='target_sum_paper.eps', extent=extent)
arcsinh_plot(conv_im, stretch, asinh_vmin=0, im_label=label+'Conv Model', im_name='model_sum_paper.eps', extent=extent)
arcsinh_plot(tgt_im-conv_im, stretch, im_label=label+'Residual, D - M', res=True, im_name = 'resid_sum_paper.eps', extent=extent, scale_val=np.max(tgt_im))
#arcsinh_plot(tgt_sum/model_sum, stretch, im_label=label+'Ratio, Target/Model', im_name = 'ratio_paper.eps', extent=extent)#, scale_val=np.max(tgt_sum))        

arcsinh_plot(deconv, stretch, asinh_vmin=0, im_name='deconv.eps', extent=extent, north=True, angle=-(316.9*(np.pi/180)))
'''
plt.imshow(conv_im/tgt_im, interpolation='nearest', extent=extent, cmap=cm.cubehelix, vmin=0., vmax=2.)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)        
plt.xlabel('Offset (")',fontsize=23)
plt.ylabel('Offset (")',fontsize=23)
cbar = plt.colorbar(pad=0.0)
cbar.set_label('Model/Data',size=23)
cbar.ax.tick_params(labelsize=18)
plt.text(-0.6,0.6,label+'Ratio',color='white',ha='left',va='top',fontsize=23)
plt.savefig('ratio_paper.eps', bbox_inches='tight')
plt.clf()
'''
plt.imshow(conv_im/tgt_im, interpolation='nearest', extent=extent, cmap=cm.PiYG, vmin=0., vmax=2.)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)        
plt.xlabel('Offset (")',fontsize=23)
plt.ylabel('Offset (")',fontsize=23)
cbar = plt.colorbar(pad=0.0)
cbar.set_label('Model/Data',size=23)
cbar.ax.tick_params(labelsize=18)
plt.text(-0.6,0.6,label+'Ratio',color='black',ha='left',va='top',fontsize=23)
plt.text(0.6,0.6, r'$\chi^2 = $'+chi, color='black',ha='right',va='top',fontsize=23)
plt.savefig('ratio_paper_2.eps', bbox_inches='tight')
plt.clf()







