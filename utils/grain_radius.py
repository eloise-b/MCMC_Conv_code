"""
Find the equilibrium grain radius in nm.
"""

from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pdb
import astropy.constants as const
import astropy.units as units
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu

#Wavelength, kappa, sigma_scattering
wks = np.loadtxt('dustkappa_56e-3_pah.inp', skiprows=3)
#wks = np.loadtxt('dustkappa_carbon.inp', skiprows=3)

#F_lambda
wave = wks[:,0] * 1e-6 * units.m
kappa = wks[:,1] * (units.cm)**2 / units.g

#********** IRS 48 ***********
star_surface_F = np.pi * blackbody_lambda(wave, 9000) * (units.rad)**2
wf = np.loadtxt('spectrum_irs.out', skiprows=2)
flux_1pc = np.interp(wave.value*1e6, wf[:,0], wf[:,1])
flux_1pc *= units.erg/units.cm**2
flux_1pc *= const.c/wave**2

dust_F1 = star_surface_F * (2*const.R_sun)**2/(13.5*const.au)**2
#!!!This doesn't make sense.  Needs to be multiplied by pi
dust_F2 = flux_1pc * (const.pc)**2/(13.5*const.au)**2

dust_a = np.trapz(dust_F2*kappa, wave)/const.c
rho_d = 1.6 * units.g / (units.cm)**3
gas_accretion_rate = 4e-9 * const.M_sun/units.yr    #From Salyk (2013) 
m_star = 2 * const.M_sun #roughly...
r = 13.5 * const.au
a_gr = gas_accretion_rate * (const.G*m_star)**0.5 / (2 * np.pi)**1.5 / rho_d / r**2.5 / dust_a
print("IRS 48 grain radius (nm): {:5.2f}".format(a_gr.si.value * 1e9))

#********** HD 169142 ***********
star_surface_F = np.pi * blackbody_lambda(wave, 8250) * (units.rad)**2
wf = np.loadtxt('spectrum_hd.out', skiprows=2)
flux_1pc = np.interp(wave.value*1e6, wf[:,0], wf[:,1])
flux_1pc *= units.erg/units.cm**2
flux_1pc *= const.c/wave**2

dust_F3 = star_surface_F * (1.6*const.R_sun)**2/(7.9*const.au)**2
#!!!This doesn't make sense.  Needs to be multiplied by pi
dust_F4 = flux_1pc * (const.pc)**2/(7.9*const.au)**2

dust_a = np.trapz(dust_F4*kappa, wave)/const.c
rho_d = 1.6 * units.g / (units.cm)**3
gas_accretion_rate = 2.1e-9 * const.M_sun/units.yr    #From Salyk (2013) 
m_star = 1.65 * const.M_sun #roughly...
r = 7.9 * const.au
a_gr = gas_accretion_rate * (const.G*m_star)**0.5 / (2 * np.pi)**1.5 / rho_d / r**2.5 / dust_a
print("HD 169142 grain radius (nm): {:5.2f}".format(a_gr.si.value * 1e9))

plt.clf()
plt.plot(wave.value*1e6, dust_F1.si.value, label='F1')
plt.plot(wave.value*1e6, dust_F2.si.value, label='F2')
plt.plot(wave.value*1e6, dust_F3.si.value, label='F3')
plt.plot(wave.value*1e6, dust_F4.si.value, label='F4')
plt.legend()
plt.axis([0,2,0,1e4])