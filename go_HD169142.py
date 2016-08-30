from __future__ import print_function, division
import multiprocessing
import numpy as np
import emcee
import pickle
from mcinmc import lnprob_conv_disk_radmc3d
import pdb
import radmc3dPy as r3

multiprocess=True

#nwalkers is set to be twice the number of parameters - should make this automatic
nwalkers = 24
print('nwalkers=',nwalkers)

#set parameters to 0.0 if you don't want them investigated. Have to set ipar_sig to zero also!

#Issues: stellar temperature and dust to gas compete with each other. They only 
ipar = np.array([np.log(5.10e-4), #Dust to gas
                 np.log(2.24e-3), #Gap Depletion
                 np.log(12.44),     #Inner Radius (AU)
                 np.log(25.0),     #Wall Radius (AU)
                 28.7,               #Inclination
                 133.4,               #Position Angle
                 0.0,-0.6,          #Star position offset (AU)
                 np.log(7327.0),   #Stellar temperature (8250 from Yeon Seok 2015)
                 0.0,0.0,0.0])     #Planet x, y, radius.

ipar = np.array([  -7.559 + np.log(50),   -6.106,    2.519,    3.219,   29.32 ,  -90 + 134.982,
          0.037,   -0.542,    8.898,    0.   ,    0.   ,    0.   ])
ipar = np.array([  -7.559 + np.log(50),   -6.106,    2.519,    3.219,   19.32 ,  -90 + 134.982,
          0.037,   -0.542,    8.898,    0.   ,    0.   ,    0.   ])

#In [35]: sampler.flatchain[5996]
#Out[35]: 
#array([ -7.58035713e+00,  -6.10215030e+00,   2.52067287e+00,
#         3.21887582e+00,   2.87033559e+01,   1.33428656e+02,
#         3.29517973e-02,  -5.71955894e-01,   8.89932185e+00,
#         0.00000000e+00,   0.00000000e+00,   0.00000000e+00])
#
#In [36]: np.exp(sampler.flatchain[5996])
#Out[36]: 
#array([  5.10378917e-04,   2.23805007e-03,   1.24369623e+01,
#         2.50000000e+01,   2.92219432e+12,   8.85786000e+57,
#         1.03350072e+00,   5.64420412e-01,   7.32700307e+03,
#         1.00000000e+00,   1.00000000e+00,   1.00000000e+00])

mode='test'
mode='mcmc'
mode='plot'
kwargs = {"planet_temp":2000,"temperature":1000,"filename":"HD169142_2014_ims.fits","dist":145}

#A test code block to see the effect of changing one parameter at a time.
if mode=='test':
    ntest = 9
    lnprob = np.empty(ntest)
    ipar_test = ipar.copy()
    ptests = np.log((0.6 + 0.05*np.arange(ntest))*1e-3)   
    ix = 0            
    ptests = 30 * np.arange(ntest)     
    ix = 5       
    ptests = np.log(13 + np.arange(ntest)*0.5)    
    ix = 2         
    for i in range(len(ptests)):
        ipar_test[ix] = ptests[i]
        #Examine our initial model...
        lnprob[i] = lnprob_conv_disk_radmc3d(ipar_test, remove_directory=False, **kwargs)
elif mode=='plot':
    lnprob = lnprob_conv_disk_radmc3d(ipar, remove_directory=False, plot_ims=True, **kwargs)
elif mode=='mcmc':
    ipar_sig = np.array([.01,.01,.01,.0,1,5,0.05,0.05,0.05,0.,0.,0.0])
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

    #Print the key outputs.
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    nsamp = sampler.lnprobability.shape[1]
    ch = sampler.chain[:,nsamp//2:,:].reshape( (nsamp//2*nwalkers, len(ipar)) )
    print(np.mean(ch,axis=0))
    print(np.std(ch,axis=0))

