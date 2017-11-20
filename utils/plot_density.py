""" TODO 

Add in quantum heating numbers.

hv_uv = 3 k_b T_d * (m_d / m_carbon) 

"""

from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const


dir = '/Users/mireland/python/mcinmc.old/37305' 
dusttogas = 0.000616
nr,ntheta,nphi=(117,80,60)
density_scale = 1.4
r_max_ring = 25.5 * const.au.cgs.value
r_max_depleted = 60.5 * const.au.cgs.value
radius_cm = 12 * const.au.cgs.value

#dir = '40742'
#dusttogas = 0.026
#nr,ntheta,nphi=(117,80,60)
#density_scale = 100
#r_max_ring = 24.9 * const.au.cgs.value
#r_max_depleted = 60.5 * const.au.cgs.value
#radius_cm = 12 * const.au.cgs.value


dust_kappa = 1e3
dust_dens = 1.6
a_gr = 1e-5 #Grain radius in cm
mass_gr = 4./3.*np.pi*a_gr**3*dust_dens
sigma_gr = mass_gr*dust_kappa

yr_in_s = 31557600.0

gridraw = np.loadtxt(dir + '/amr_grid.inp', skiprows=6)

#Load in the dust density here.
dd = np.loadtxt(dir + '/dust_density.inp', skiprows=3)
#Dimensions are (phi, theta, r)
dd = dd.reshape( (nphi,ntheta,nr) )

#Load in the dust temperature here.
tt = np.loadtxt(dir + '/dust_temperature.dat', skiprows=3)
#Dimensions are (phi, theta, r)
tt = tt.reshape( (nphi,ntheta,nr) )


#Load in the coordinates here.
gridraw = np.loadtxt(dir + '/amr_grid.inp', skiprows=6)
rgrid = gridraw[0:nr+1]
thetagrid = gridraw[nr+1:nr+ntheta+2]
phigrid = gridraw[nr+80+2:]
rplot =0.5*(rgrid[1:] + rgrid[:-1])
thetaplot =0.5*(thetagrid[1:] + thetagrid[:-1])
phiplot =0.5*(phigrid[1:] + phigrid[:-1])
dr = rgrid[1:] - rgrid[:-1]

#------------------------------
#Compute the optical depth of different parts of the disk.
las = np.loadtxt('dustkappa_carbon.inp',skiprows=3)
las_b = np.interp(0.4,las[:,0], las[:,1])
las_l = np.interp(3.7,las[:,0], las[:,1])

i = 0
j = 40

plt.clf()
plt.semilogy(rplot/const.au.cgs.value, dd[i,j,:])
plt.axis([10,100,5e-21,1e-19])

ww = np.where(rplot < 25.5*const.au.cgs.value)[0]
int1 = np.trapz(dd[i,j,ww], rplot[ww])
print("First integral (g/cm^2): {0:10.2e}".format(int1))
print("Tau: {0:10.2e}".format(int1*las_b))
print("Tau: {0:10.2e}".format(int1*las_l))
ww = np.where(rplot < 60.5*const.au.cgs.value)[0]
int2 = np.trapz(dd[i,j,ww], rplot[ww])
print("Second integral (g/cm^2): {0:10.2e}".format(int2))
print("Tau: {0:10.2e}".format(int2*las_b))
print("Tau: {0:10.2e}".format(int2*las_l))
ww = np.where(rplot < 100*const.au.cgs.value)[0]
int3 = np.trapz(dd[i,j,ww], rplot[ww])
print("Third integral (g/cm^2): {0:10.2e}".format(int3))
print("Tau: {0:10.2e}".format(int3*las_b))
print("Tau: {0:10.2e}".format(int3*las_l))


#------------------------------
#Surface density is integral of rho dz, which is the integral of 4* \int rho dtheta 
sigma = np.zeros(nr)
for i in range(nr):
    sigma[i] = rplot[i]*np.trapz(dd[0,:,i], thetaplot)
plt.clf()
plt.semilogy(rplot/const.au.cgs.value, sigma)
plt.axis([10,100,3e-7,3e-5])
plt.xlabel('Radius (AU)')
plt.ylabel(r'Surface Density (g cm$^{-3}$)')

#Total mass is 2*np.pi * \int r * sigma dr
dust_mass = 2 * np.pi * np.sum(sigma * rplot * dr)
total_mass = dust_mass/dusttogas/const.M_sun.cgs.value

ww = np.where(rplot < r_max_ring)[0]
dust_mass_ring = 2 * np.pi * np.sum(sigma[ww] * rplot[ww] * dr[ww])
dust_mass_ring_me  = dust_mass_ring/const.M_earth.cgs.value
gas_mass_ring = dust_mass_ring/dusttogas*density_scale
ring_timescale1 = gas_mass_ring/const.M_sun.cgs.value/4e-9

ww = np.where(rplot < r_max_depleted)[0]
dust_mass_ring2 = 2 * np.pi * np.sum(sigma[ww] * rplot[ww] * dr[ww])
dust_mass_ring2_me  = dust_mass_ring2/const.M_earth.cgs.value
gas_mass_ring = dust_mass_ring2/dusttogas*density_scale
ring_timescale2 = gas_mass_ring/const.M_sun.cgs.value/4e-9

#---
max_density = dd[0,ntheta//2,1]/dusttogas*density_scale
number_density = max_density/2/const.u.cgs.value

T_r = np.mean(np.mean(tt,axis=0),axis=0)
T_ref = np.mean(T_r[0:9])
T_gas = 200
v_kin = np.sqrt(T_gas*const.k_B.cgs.value/const.u.cgs.value/2/2)
collision_rate = v_kin*number_density*np.pi*a_gr**2
collision_power = collision_rate*0.5*T_gas*const.k_B.cgs.value

radiative_power = 4*np.pi*sigma_gr * const.sigma_sb.cgs.value * T_ref**4

nonlte_fact = radiative_power/collision_power

corrected_dusttogas = dusttogas/density_scale

outwards_force = radiative_power/const.c.cgs.value
pr_force = outwards_force*(10/3e5)
pr_acc = pr_force/mass_gr
outwards_acc = outwards_force/mass_gr
grav_acc = (const.G*const.M_sun*2/(11*const.au)**2).cgs.value

#Acceleration times stopping time is drift velociey
t_stop = dust_dens/max_density*a_gr/v_kin
drift_v = outwards_acc*t_stop
outflow_t_yr = radius_cm/drift_v/yr_in_s
outflow_acc_t_yr = np.sqrt(radius_cm/outwards_acc)/yr_in_s

