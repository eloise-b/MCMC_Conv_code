from __future__ import print_function, division
import multiprocessing
import numpy as np
import emcee
import pickle
from convolve_rotate_model_mcmc import lnprob_conv_disk_radmc3d
import pdb
import radmc3dPy as r3

multiprocess=False

#nwalkers is set to be twice the number of parameters - should make this automatic
nwalkers = 24
print('nwalkers=',nwalkers)

#set parameters to 0.0 if you don't want them investigated. Have to set ipar_sig to zero also!
ipar = np.array([np.log(0.6e-3), #Dust to gas
                 np.log(3e-3), #Gap Depletion
                 np.log(15.0),     #Inner Radius (AU)
                 np.log(25.0),     #Wall Radius (AU)
                 20,               #Inclination
                 0,                #Position Angle
                 0.0,0.0,          #Star position offset (AU)
                 np.log(8250.0),   #Stellar temperature (from Yeon Seok 2015)
                 0.0,0.0,0.0])     #Planet x, y, radius.

ntest = 12
lnprob = np.empty(ntest)
ipar_test = ipar.copy()
#A test code block to see the effect of changing one parameter at a time.
if (False):
    ptests = np.log((0.6 + 0.05*np.arange(ntest))*1e-3)   
    ix = 0            
    ptests = 30 * np.arange(ntest)     
    ix = 5         
    for i in range(len(ptests)):
        ipar_test[ix] = ptests[i]
        #Examine our initial model...
        lnprob[i] = lnprob_conv_disk_radmc3d(ipar_test, dist=145, remove_directory=False)

#pdb.set_trace()
                 
ipar_sig = np.array([.01,.01,.01,.01,0.1,0.1,0.,0.,0.05,0.,0.,0.0])
ndim = len(ipar)
#Could use parameters of random.normal instead of below, if you prefer that.
p0 = [ipar + np.random.normal(size=ndim)*ipar_sig for i in range(nwalkers)]

if (multiprocess):
    threads = multiprocessing.cpu_count()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,threads=threads,kwargs={"planet_temp":2000})
    sampler.run_mcmc(p0,3)
else:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d,kwargs={"planet_temp":2000})
    sampler.run_mcmc(p0,3)

chainfile = open('chainfile.pkl','w')
pickle.dump((sampler.lnprobability,sampler.chain),chainfile)
chainfile.close()

#Useful things
#np.max(sampler.flatlnprobability)
#np.argmax(sampler.flatlnprobability)
#sampler.flatchain[np.argmax(sampler.flatlnprobability)]
#np.exp(sampler.flatchain[np.argmax(sampler.flatlnprobability)][0:5])
