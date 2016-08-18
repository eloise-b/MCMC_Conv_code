from __future__ import print_function, division
import multiprocessing
import numpy as np
import emcee
import pickle
from convolve_rotate_model_mcmc import lnprob_conv_disk_radmc3d

multiprocess=False

#nwalkers is set to be twice the number of parameters - should make this automatic
nwalkers = 26
print('nwalkers=',nwalkers)

#set parameters to 0.0 if you don't want them investigated. Have to set ipar_sig to zero also!
ipar = np.array([np.log(6.894e-3),np.log(3.012e-3),np.log(11.22),np.log(22.13),48.85,129.5,0.0,0.0,np.log(8000.0),0.0,0.0,0.0])
ipar_sig = np.array([.01,.01,.01,.01,0.1,0.1,0.,0.,0.05,0.,0.,0.0])
ndim = len(ipar)
#Could use parameters of random.normal instead of below, if you prefer that.
p0 = [ipar + np.random.normal(size=ndim)*ipar_sig for i in range(nwalkers)]

if (multiprocess):
    threads = multiprocessing.cpu_count()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,threads=threads,kwargs={"planet_temp":2000})
    sampler.run_mcmc(p0,5)
else:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,kwargs={"planet_temp":2000})
    sampler.run_mcmc(p0,5)

chainfile = open('chainfile.pkl','w')
pickle.dump((sampler.lnprobability,sampler.chain),chainfile)
chainfile.close()

#Useful things
#np.max(sampler.flatlnprobability)
#np.argmax(sampler.flatlnprobability)
#sampler.flatchain[np.argmax(sampler.flatlnprobability)]
#np.exp(sampler.flatchain[np.argmax(sampler.flatlnprobability)][0:5])
