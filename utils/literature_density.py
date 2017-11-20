from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
import astropy.units as units

#******* Start with Oph IRS48 **********
print("Oph IRS 48")

gas_accretion_rate = 4e-9 * const.M_sun/units.yr    #From Salyk (2013) 
Sigma_c = 3.2e-2 * units.g / units.cm**2            #From Bruderer (2014)
r_c     = 60.0 * const.au                           #From Bruderer (2014)

depletion = 9e-3
nr = 1000
rs = np.linspace(0.4,20,nr) * const.au
sigmas = depletion * Sigma_c * (rs/r_c)**(-1)
disk_mass = 2*np.pi*np.trapz(rs*sigmas,rs)
depletion_time = (disk_mass/gas_accretion_rate).si
print("Inner disk depletion in years: {:5.1f}".format( (depletion_time.si/units.yr).si.value ) )

depletion = 8e-2
nr = 1000
rs = np.linspace(20,60,nr) * const.au
sigmas = depletion * Sigma_c * (rs/r_c)**(-1)
disk_mass = 2*np.pi*np.trapz(rs*sigmas,rs)
depletion_time = (disk_mass/gas_accretion_rate).si
print("Mid disk depletion in years: {:5.1f}".format( (depletion_time.si/units.yr).si.value ) )

#********* The same for HD 169142 *************
print("HD 169142")

gas_accretion_rate = 2.1e-9 * const.M_sun/units.yr  #From Wagner (2015)
Sigma_c = 6.5e-2 * units.g / units.cm**2            #From Fedele (2017)
r_c     = 100.0 * const.au                          #From Fedele (2017)

depletion = 0.025  #From Fedele (2017)
nr = 1000
rs = np.linspace(13,35,nr) * const.au
sigmas = depletion * Sigma_c * (rs/r_c)**(-1)
disk_mass = 2*np.pi*np.trapz(rs*sigmas,rs)
depletion_time = (disk_mass/gas_accretion_rate).si
print("Depleted disk depletion in years: {:5.1f}".format( (depletion_time.si/units.yr).si.value ) )

