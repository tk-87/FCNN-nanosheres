#! /usr/bin/python3

import time
import cmath
import random
import numpy as np
from random import seed
from scipy import special as sp
from numpy import asarray
from numpy import save

# total cross-section (sum of TE and TM components)
def total_cs(a,omega,eps):

    # spherical cross-section
    def SCS(polarisation,v,a,omega,eps):

        z      = complex(0,1)
        M      = TM(polarisation,v,a,omega,eps)
        r      = ( M[0,0,:] - z*M[1,0,:] )/( M[0,0,:] + z*M[1,0,:] )
        coef   = (2*v + 1)*np.pi/( omega**2*eps[:,-1] )
        sigma  = coef * np.abs(1 - r)**2

        return sigma

    # transfer matrix
    def TM(polarisation,v,a,omega,eps):

        def subTM(polarisation,v,a,omega,eps1,eps2):
            k1   = omega * np.sqrt(eps1)
            k2   = omega * np.sqrt(eps2)
            x1   = a * k1
            x2   = a * k2

            j1   = sp.spherical_jn(v,x1,False)
            j1d  = sp.spherical_jn(v,x1,True)
            y1   = sp.spherical_yn(v,x1,False)
            y1d  = sp.spherical_yn(v,x1,True)
            j2   = sp.spherical_jn(v,x2,False)
            j2d  = sp.spherical_jn(v,x2,True)
            y2   = sp.spherical_yn(v,x2,False)
            y2d  = sp.spherical_yn(v,x2,True)

            if polarisation == 1:
                m1 = np.array( [[ y2d , -y2 ] , [ -j2d , j2 ]] )
                m2 = np.array( [[ j1 , y1 ] , [ j1d , y1d ]] )
            else:
                m1 = np.array( [[ y2d*eps1 , -y2 ] , [ -j2d*eps1 , j2 ]] )
                m2 = np.array( [[ j1 , y1 ] , [ j1d*eps2 , y1d*eps2 ]] )

            m = product(m1,m2)

            return m

        M = np.zeros([2,2,N])
        M[0,0,:] = 1
        M[1,1,:] = 1

        a = np.cumsum(a)
        for layer in range(k):
            eps1 = eps[:,layer]
            eps2 = eps[:,layer+1]
            m = subTM(polarisation,v,a[layer],omega,eps1,eps2)
            M = product(m,M)

        return M

    # takes the product of two sets of 2x2 matrices with dimentions (2,2,N)
    def product(m1,m2):

        m = np.array([[ m1[0,0,:]*m2[0,0,:] + m1[0,1,:]*m2[1,0,:] , m1[0,0,:]*m2[0,1,:] + m1[0,1,:]*m2[1,1,:] ] ,
                      [ m1[1,0,:]*m2[0,0,:] + m1[1,1,:]*m2[1,0,:] , m1[1,0,:]*m2[0,1,:] + m1[1,1,:]*m2[1,1,:] ]])

        return m


    sigma = 0
    for v in range(order):
        sigma = sigma + SCS(1,v,a,omega,eps) + SCS(2,v,a,omega,eps)

    return sigma


# generate random shell thicknesses
def thicknesses():

    a = np.empty([1,k], dtype=int)
    for i in range(k):
        a[0,i] = random.randrange(30,70)

    return a

# construct N-by-(k+1) permittivity matrix
def eps(Lambda):

    E_S     = 2.04 * np.ones([1,N])
    E_TiO2  = 5.913 + 0.2441/( (Lambda/1000)**2 - 0.0803 )

    array = np.ones([N,k+1])
    array[:,0], array[:,2], array[:,4], array[:,6] = E_S, E_S, E_S, E_S
    array[:,1], array[:,3], array[:,5], array[:,7] = E_TiO2, E_TiO2, E_TiO2, E_TiO2

    return array



lower      = 400                                    # lower limit in nm
upper      = 800                                    # upper limit in nm
N          = upper - lower                          # number of data points
order      = 10                                     # order of bessel function
k          = 8                                      # number of layers
Lambda     = np.linspace(lower, upper, N)           # 1D wavelength array
omega      = 2*np.pi*np.reciprocal( Lambda )        # 1D angular velocity arry
eps        = eps(Lambda)                            # 2D permittivity array
seed(42)



t = time.time()

n       = 50000
A       = np.empty([n,k], dtype=int)
spectra = np.empty([n,N])
for i in range(n):

    a   = thicknesses()
    CS  = np.array(total_cs(a,omega,eps))
    CS  = CS / (np.pi*np.sum(a)**2)

    A[i,:]       = a
    spectra[i,:] = CS


# uncomment these to save shell thickness and scattering cross section arrays to file
#save('../data/thicknesses.npy', A)
#save('../data/scatter_CS.npy', spectra)

t = time.time() - t
print(t)
