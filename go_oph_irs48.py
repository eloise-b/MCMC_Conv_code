from __future__ import print_function, division
import multiprocessing
import numpy as np
import emcee
import pickle
from lnprob_radmc3d import lnprob_conv_disk_radmc3d

multiprocess=False

#nwalkers is set to be twice the number of parameters - should make this automatic
nwalkers = 24
print('nwalkers=',nwalkers)

#set parameters to 0.0 if you don't want them investigated. Have to set ipar_sig to zero also!
#These parameters are from when the temperature was a parameter, now we are back to having an inner disc
#ipar = np.array([np.log(6.894e-3),np.log(3.012e-3),np.log(11.22),np.log(22.13),48.85,129.5,0.0,0.0,np.log(8000.0),0.0,0.0,0.0])
#ipar = np.array([np.log(6.13e-4),np.log(1.46e-2),np.log(11.33),np.log(25.87),57.57,130.6,-0.14,0.04,np.log(10236.0),-17,0.0,1.0])
#ipar_sig = np.array([.01,.01,.01,.0,1,5,0.05,0.05,0.03,0.3,0.3,0.05])

#For a planet
#ipar = np.array([ -7.399,  -4.218,   2.425,   3.253,  55.795,  37.778,   0.35 ,\
#         0.157,   9.235,  18.095,   1.042,   0.883])
#ipar = np.array([ -7.393,  -4.167,   2.441,   3.253,  56.514,  270+36.222,   0.372,\
#         0.046,   9.23 ,  17.196,   0.429,   0.941])
#ipar_sig = np.array([ 0.02 ,  0.041,  0.01 ,  0.02 ,  0.79 ,  0.754,  0.058,  0.065,\
#        0.007,  0.716,  0.515,  0.148])

#Assymetry Only
#ipar = np.array([  -6.349,   -4.951,    2.439,    3.305,   56.506,  128.062,
#         -0.269,    0.149,    9.196,    0.   ,    0.   ,    0.   ])
#ipar_sig = np.array([ 0.098,  0.081,  0.01 ,  0.017,  0.539,  0.684,  0.041,  0.051,
#        0.007,  0.   ,  0.   ,  0.   ])

#Parameters to be used when there is an inner disc
#Symmetric disc
ipar = np.array([np.log(6.894e-3),np.log(2.894e-8),np.log(3.012e-3),np.log(11.22),np.log(22.13),48.85,165.,0.0,0.0,0.0,0.0,0.0])
ipar_sig = np.array([.1,.1,.1,.1,.1,1.,5.,0.0,0.0,0.,0.,0.0])

#mode='test'
mode='mcmc'
#mode='plot'
kwargs = {"planet_temp":1500,"temperature":10000,"filename":"IRS48_ims.fits", "kappa":"['56e-3_pah']","Kurucz":True}

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
    #kwargs['nphot']="long(1.2e5)"
    lnprob = lnprob_conv_disk_radmc3d(ipar, remove_directory=False, plot_ims=True, save_im_data=True,\
                                      **kwargs)
elif mode=='mcmc':
    ndim = len(ipar)
    #Could use parameters of random.normal instead of below, if you prefer that.
    p0 = [ipar + np.random.normal(size=ndim)*ipar_sig for i in range(nwalkers)]

    if (multiprocess):
        threads = multiprocessing.cpu_count()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d, threads=threads, kwargs=kwargs)
        sampler.run_mcmc(p0,500)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_conv_disk_radmc3d, kwargs=kwargs)
        sampler.run_mcmc(p0,5)

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

    #Useful things
    #np.max(sampler.flatlnprobability)
    #np.argmax(sampler.flatlnprobability)
    #sampler.flatchain[np.argmax(sampler.flatlnprobability)]
    #np.exp(sampler.flatchain[np.argmax(sampler.flatlnprobability)][0:5])
