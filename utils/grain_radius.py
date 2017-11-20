"""
Find the equilibrium grain radius in nm.
"""

from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
import astropy.units as units
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu

#Wavelength, kappa, sigma_scattering
wks = np.loadtxt('dustkappa_56e-3_pah.inp', skiprows=3)
#wks = np.loadtxt('dustkappa_carbon.inp', skiprows=3)

#F_lambda
wave = wks[:,0] * 1e-6 * units.m
star_surface_F = np.pi * blackbody_lambda(wave, 9000) * (units.rad)**2
kappa = wks[:,1] * (units.cm)**2 / units.g

#********** IRS 48 ***********
dust_F = star_surface_F * (2*const.R_sun)**2/(13.5*const.au)**2
dust_a = np.trapz(dust_F*kappa, wave)/const.c
rho_d = 1.6 * units.g / (units.cm)**3
gas_accretion_rate = 4e-9 * const.M_sun/units.yr    #From Salyk (2013) 
m_star = 2 * const.M_sun #roughly...
r = 13.5 * const.au
a_gr = gas_accretion_rate * (const.G*m_star)**0.5 / 2 / np.pi**1.5 / rho_d / r**2.5 / dust_a
print("IRS 48 grain radius (nm): {:5.2f}".format(a_gr.si.value * 1e9))

#********** HD 169142 ***********
dust_F = star_surface_F * (2*const.R_sun)**2/(7.9*const.au)**2
dust_a = np.trapz(dust_F*kappa, wave)/const.c
rho_d = 1.6 * units.g / (units.cm)**3
gas_accretion_rate = 2.1e-9 * const.M_sun/units.yr    #From Salyk (2013) 
m_star = 1.65 * const.M_sun #roughly...
r = 7.9 * const.au
a_gr = gas_accretion_rate * (const.G*m_star)**0.5 / 2 / np.pi**1.5 / rho_d / r**2.5 / dust_a
print("HD 169142 grain radius (nm): {:5.2f}".format(a_gr.si.value * 1e9))