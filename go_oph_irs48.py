from __future__ import print_function, division
import multiprocessing
import numpy as np
import emcee
import pickle
from convolve_rotate_model_mcmc import lnprob_conv_disk_radmc3d

multiprocess=True

#nwalkers is set to be twice the number of parameters - should make this automatic
nwalkers = 24
print('nwalkers=',nwalkers)

#set parameters to 0.0 if you don't want them investigated. Have to set ipar_sig to zero also!
ipar = np.array([np.log(6.894e-3),np.log(3.012e-3),np.log(11.22),np.log(22.13),48.85,129.5,0.0,0.0,np.log(8000.0),0.0,0.0,0.0])
ipar = np.array([np.log(6.13e-4),np.log(1.46e-2),np.log(11.33),np.log(25.87),57.57,130.6,-0.14,0.04,np.log(10236.0),-17,0.0,1.0])

mode='test'
mode='mcmc'
#mode='plot'
kwargs = {"planet_temp":1500,"temperature":10000,"filename":"IRS48_ims.fits"}

#A test code block to see the effect of changing one parameter at a time.
if mode=='test':
    ntest = 9
    lnprob = np.empty(ntest)
    ipar_test = ipar.copy()
    ptests = np.log((0.6 + 0.05*np.arange(ntest))*1e-3)       
    for i in range(len(ptests)):
        ipar_test[ix] = ptests[i]
        #Examine our initial model...
        lnprob[i] = lnprob_conv_disk_radmc3d(ipar_test, remove_directory=False, **kwargs)
elif mode=='plot':
    lnprob = lnprob_conv_disk_radmc3d(ipar, remove_directory=False, plot_ims=True, **kwargs)
elif mode=='mcmc':
    ipar_sig = np.array([.01,.01,.01,.0,1,5,0.05,0.05,0.03,0.3,0.3,0.05])
    ndim = len(ipar)
    #Could use parameters of random.normal instead of below, if you prefer that.
    p0 = [ipar + np.random.normal(size=ndim)*ipar_sig for i in range(nwalkers)]

    if (multiprocess):
        threads = multiprocessing.cpu_count()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d, threads=threads, kwargs=kwargs)
        sampler.run_mcmc(p0,500)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d, kwargs=kwargs)
        sampler.run_mcmc(p0,50)

    chainfile = open('chainfile.pkl','w')
    pickle.dump((sampler.lnprobability,sampler.chain),chainfile)
    chainfile.close()

    #Useful things
    #np.max(sampler.flatlnprobability)
    #np.argmax(sampler.flatlnprobability)
    #sampler.flatchain[np.argmax(sampler.flatlnprobability)]
    #np.exp(sampler.flatchain[np.argmax(sampler.flatlnprobability)][0:5])