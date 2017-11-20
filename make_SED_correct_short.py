from __future__ import division

"""
Script to make an SED using Radmc3d and then make a spectrum and an image
"""

# import numpy and other things
import numpy as np
import matplotlib.pyplot as plt
import pdb
plt.ion
import astropy.units as u
#import specutils
from specutils import extinction

#be able to run from python
import os

#run the sed function - can set inclination (incl) and phi angle (posang?) (phi)
#os.system('radmc3d sed incl 50 phi 129')
#print "done SED"

#need file camera_wavelength_micron.inp - has same format as wavelength_micron.inp,
#will try using it as the same with a new name and also with it as just the wavelengths that 
#I have made models for

#makes the spectrum over the wavelengths in camera_wavelength_micron.inp
#os.system('radmc3d spectrum loadlambda incl 50 phi 129')
#overwrites the spectrum.out file created by the sed
#print "done Spectrum"

#create an image at a certain wavelength
#os.system('radmc3d image lambda 880')
#The image made here is no where near the quality it is when run properly to make an image
#in the usual way.
#print "done image"

#print "done."

#load spectrum.out into python

#'''
star = np.loadtxt('../IRS_star_SED/spectrum.out', skiprows=3)
#sym = np.loadtxt('../IRS_sym_SED/spectrum.out', skiprows=3)
#asym = np.loadtxt('../IRS_asym_SED/spectrum.out', skiprows=3)
#pla = np.loadtxt('../IRS_new_planet/spectrum.out', skiprows=3)
#brud = np.loadtxt('bruderer_points.txt')
#brud = np.loadtxt('IRS48_phot_data.txt')
#i_uncert = np.loadtxt('irs_data_uncert.txt')
#i_upper = np.loadtxt('irs_data_upper.txt')
#i_data = np.loadtxt('irs_data_plain.txt')
'''
star = np.loadtxt('../HD_star_SED/spectrum.out', skiprows=3)
#sym = np.loadtxt('../HD_sym_SED/spectrum.out', skiprows=3)
#asym = np.loadtxt('../HD_asym_SED/spectrum.out', skiprows=3)
#pla = np.loadtxt('../HD_new_planet/spectrum.out', skiprows=3)
#seok = np.loadtxt('seok_points.txt')
#seok = np.loadtxt('hd169142_phot_data.txt')
#uncert = np.loadtxt('hd_uncert.txt')
'''
#Plot the SED
#define c in cm/s
c = 2.99792458e10
dist = 120 # to convert to a distance 
#av and rv
av = 11.5
rv=5.5
#av = 0.31
#rv=3.1
#make frequency arrays, convert wavelength to cm (from micron) to calculate
#nu_sym = c / (sym[:,0]/1e4) 
nu_star = c / (star[:,0]/1e4) 
#nu_asym = c / (asym[:,0]/1e4) 
#nu_pla = c / (pla[:,0]/1e4) 

#nu_brud = c / (brud[:,0]/1e4)
#nu_seok = c / (seok[:,0]/1e4)

#nu_i_un = c / (i_uncert[:,0]/1e4)
#nu_i_da = c / (i_data[:,0]/1e4)
#nu_i_up = c / (i_upper[:,0]/1e4)


#unc = uncert*10**-23*nu_seok
#unc = i_uncert[:,2]*10**-23*nu_i_un
#data = i_data*10**-23*nu_i_da
#up = i_upper*10**-23*nu_i_up

#get the amount of extinction needed for the correction 
#wlgth = brud[:,0]*10**4
#wlgth = seok[:,0]*10**4
#wlgth = wlgth*u.angstrom
#ext = extinction.extinction_wd01(wave=wlgth, a_v=av, r_v=rv)

#wlgth_un = i_uncert[:,0]*10**4
#wlgth = seok[:,0]*10**4
#wlgth_un = wlgth_un*u.angstrom
#ext_un = extinction.extinction_wd01(wave=wlgth_un, a_v=av, r_v=rv)

#wlgth_up = i_upper[:,0]*10**4
#wlgth = seok[:,0]*10**4
#wlgth_up = wlgth_up*u.angstrom
#ext_up = extinction.extinction_wd01(wave=wlgth_up, a_v=av, r_v=rv)

#wlgth = i_data[:,0]*10**4
#wlgth = seok[:,0]*10**4
#wlgth = wlgth*u.angstrom
#ext = extinction.extinction_wd01(wave=wlgth, a_v=av, r_v=rv)

####de-redden the photometry
#read in the reddening file from weingartner and draine
#wgred = np.loadtxt('wg_red_5.5.txt')
#fix the wavelength to be in micron (reads in as the inverse)
#wgr_wave = 1./wgred[:,0]
#fix the reddening as in the file it has been moved by 0.2
#wgr_red = (10**0.2)*wgred[:,1]



#reddening for flux
#gamma
#g =1
#a = 5.6e-3
#plot the SEDs, divide flux by dist^2 to normalise it correctly
plt.loglog(star[:,0],nu_star*star[:,1]/dist**2, lw=2, ls=':', color='y', label='Star')
#plt.loglog(asym[:,0],nu_asym*asym[:,1]/dist**2, lw=2, ls='-.', color='green', label='Asym')
#plt.loglog(sym[:,0],nu_sym*(sym[:,1]/dist**2)*np.exp(a/sym[:,0]**g), lw=2, ls='--', color='blue', label='Sym')
#plt.loglog(sym[:,0],nu_sym*sym[:,1]/dist**2, lw=2, ls='--', color='blue', label='Sym')
#plt.loglog(pla[:,0],nu_pla*pla[:,1]/dist**2, lw=2, ls='-', color='magenta', label='Model')
#10^-23 in fluxes because the flux was originally in Janskys.
#plt.loglog(seok[:,0],seok[:,1]*1e-23*nu_seok*10**(0.4*ext),'kx', label='Data')
#plt.errorbar(seok[:,0],seok[:,1]*1e-23*nu_seok*10**(0.4*ext),yerr=unc, fmt='kx', label='Data')
#plt.errorbar(i_uncert[:,0],i_uncert[:,1]*1e-23*nu_i_un*10**(0.4*ext_un),yerr=unc, fmt='kx')
#plt.errorbar(i_upper[:,0],i_upper[:,1]*1e-23*nu_i_up*10**(0.4*ext_up), uplims=True, fmt='kv')
#plt.loglog(i_data[:,0],i_data[:,1]*1e-23*nu_i_da*10**(0.4*ext),'kx', label='Data')
#plt.loglog(brud[:,0],brud[:,1]*1e-23*nu_brud*10**(0.4*ext),'kx', label='Data')
#plt.plot([1e-1,30], [1e-9,1e-9], color='k', ls='-', lw=1)
#plt.plot([30,30], [1e-9,1e-7], color='k', ls='-', lw=1)
#plt.plot([1e-1,30], [3e-10,3e-10], color='k', ls='-', lw=1)
#plt.plot([30,30], [3e-10,3e-8], color='k', ls='-', lw=1)
#bbnu = c / bb[:,0]
#plt.loglog(bb[:,0],bbnu*bb[:,1],'--')
#plt.axis((1e-1,1e3,1e-14,1e-7))
plt.axis((1e-1,2e2,1e-11,1e-7))
#plt.axis((1e-1,30,3e-10,3e-8))
#plt.axis((1e-1,30,1e-9,1e-7))
plt.xlabel('Wavelength '+r'($\mu$m)', fontsize=23)
plt.ylabel(r'$\nu\,F_\nu$', fontsize=23)
plt.title('SED IRS 48 Dereddened Photometry', fontsize=23)
#plt.title('SED Oph IRS 48 Models', fontsize=23)
#plt.title('SED (No inner disc, 10AU ring)', fontsize=23)
plt.legend(prop={'size':18}, loc='lower center')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.savefig('IRS_SED_dered_err_zoom.pdf', bbox_inches='tight')
plt.savefig('IRS_SED_dered_uncert_zoom_new.pdf', bbox_inches='tight')
plt.clf()
'''
#reddening for flux
w_use = np.where(sym[:,0]<=1000)
wlgth2 = sym[w_use]
wlgth2 = wlgth2[:,0]*10**4
wlgth2 = wlgth2*u.angstrom
red = extinction.extinction_wd01(wave=wlgth2, a_v=av, r_v=rv)
nu_sym = nu_sym[w_use]
sym_use = sym[w_use]
nu_asym = nu_asym[w_use]
asym_use = asym[w_use]
nu_pla = nu_pla[w_use]
pla_use = pla[w_use]
nu_star = nu_star[w_use]
star_use = star[w_use]

plt.loglog(star_use[:,0],(nu_star*star_use[:,1]/dist**2)*10**(-0.4*red), lw=2, ls=':', color='y', label='Star')
#plt.loglog(asym_use[:,0],(nu_asym*asym_use[:,1]/dist**2)*10**(-0.4*red), lw=2, ls='-.', color='green', label='Asym')
#plt.loglog(sym[:,0],nu_sym*(sym[:,1]/dist**2)*np.exp(a/sym[:,0]**g), lw=2, ls='--', color='blue', label='Sym')
#plt.loglog(sym_use[:,0],(nu_sym*sym_use[:,1]/dist**2)*10**(-0.4*red), lw=2, ls='--', color='blue', label='Sym')
plt.loglog(pla_use[:,0],(nu_pla*pla_use[:,1]/dist**2)*10**(-0.4*red), lw=2, ls='-', color='magenta', label='Model')
#10^-23 in fluxes because the flux was originally in Janskys.
#plt.loglog(seok[:,0],seok[:,1]*1e-23*nu_seok,'kx', label='Data')
#plt.errorbar(seok[:,0],seok[:,1]*1e-23*nu_seok,yerr=unc, fmt='kx',label='Data')
plt.errorbar(i_uncert[:,0],i_uncert[:,1]*1e-23*nu_i_un,yerr=unc, fmt='kx')
plt.errorbar(i_upper[:,0],i_upper[:,1]*1e-23*nu_i_up, uplims=True, fmt='kv')
plt.loglog(i_data[:,0],i_data[:,1]*1e-23*nu_i_da,'kx', label='Data')
#plt.loglog(brud[:,0],brud[:,1]*1e-23*nu_brud,'kx', label='Data')
#plt.plot([1e-1,30], [1e-9,1e-9], color='k', ls='-', lw=1)
#plt.plot([30,30], [1e-9,1e-7], color='k', ls='-', lw=1)
#plt.plot([1e-1,30], [3e-10,3e-10], color='k', ls='-', lw=1)
#plt.plot([30,30], [3e-10,3e-8], color='k', ls='-', lw=1)
#bbnu = c / bb[:,0]
#plt.loglog(bb[:,0],bbnu*bb[:,1],'--')
plt.axis((1e-1,2e2,1e-14,1e-8))
#plt.axis((1e-1,1e3,1e-14,1e-7))
#plt.axis((1e-1,30,3e-10,3e-8))
#plt.axis((1e-1,30,1e-9,1e-7))
plt.xlabel('Wavelength '+r'($\mu$m)', fontsize=23)
plt.ylabel(r'$\nu\,F_\nu$', fontsize=23)
plt.title('SED IRS 48 Reddened Model', fontsize=23)
#plt.title('SED Oph IRS 48 Models', fontsize=23)
#plt.title('SED (No inner disc, 10AU ring)', fontsize=23)
plt.legend(prop={'size':18}, loc='lower center')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.savefig('IRS_SED_redden_data_err_zoom.pdf', bbox_inches='tight')
plt.savefig('IRS_SED_red_err_zoom_new.pdf', bbox_inches='tight')
plt.clf()
'''

