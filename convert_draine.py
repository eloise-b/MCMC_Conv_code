from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

filename = 'data/small_PAH_C.txt'
#filename = 'data/tiny_PAH_C.txt'

grain_density = 1.6 #Guess of grain density in g/cm^3

outfile = 'dustkappa_tiny_PAH.inp'

with open(filename, 'r') as f:
    first_line = f.readline()
    
grain_radius_in_cm = float(first_line.split()[0]) * 1e-4
wQQQg = np.genfromtxt(filename, skip_header=2, delimiter=[9,10,10,10,10])

#Reverse the wavelength axis
wQQQg = wQQQg[::-1]

#Cross section per grain
sigma_abs=wQQQg[:,2] * np.pi * grain_radius_in_cm**2 
sigma_scat=wQQQg[:,3] * np.pi * grain_radius_in_cm**2 

#Cross section per gram, i.e. opacity.
grain_mass = grain_density * np.pi * 4.0/3.0 * grain_radius_in_cm**3
kappa_abs = sigma_abs/grain_mass
kappa_scat = sigma_scat/grain_mass

wave=wQQQg[:,0]

with open(outfile, 'w') as f:
    f.write("   2\n")
    f.write("   {0:d}\n".format(len(wave)))
    for i in range(len(wave)):
        f.write("{0:9.3e} {1:9.3e} {2:9.3e}\n".format(wave[i], kappa_abs[i], kappa_scat[i]))