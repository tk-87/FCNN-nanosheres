#! /usr/bin/python3

import cmath
import numpy as np
from random import seed
from random import random
from scipy import special as sp
import matplotlib.pyplot as plt

# total cross-section (sum of TE and TM components)
def total_cs(a,omega,Lambda,eps):

    sigma = 0
    for v in range(order):
        sigma = sigma + SCS(1,v,a,omega,Lambda,eps) + SCS(2,v,a,omega,Lambda,eps)

    return sigma

# spherical cross-section
def SCS(k,v,a,omega,Lambda,eps):

    z      = complex(0,1)
    M      = np.array( TM(k,v,a,omega,eps) )
    r      = ( M[0,0,:] - z*M[1,0,:] )/( M[0,0,:] + z*M[1,0,:] )
    sigma  = Lambda**2*abs(1 - r)**2/(8*np.pi)

    return sigma

# transfer matrix
def TM(k,v,a,omega,eps):

    def subTM(k,v,a,omega,eps1,eps2):
        k1   = omega * np.sqrt(eps1)
        k2   = omega * np.sqrt(eps2)
        x1   = a * k1
        x2   = a * k2

        obj  = bessel()
        j1   = obj.first(v,x1,False)
        j1d  = obj.first(v,x1,True)
        y1   = obj.second(v,x1,False)
        y1d  = obj.second(v,x1,True)
        j2   = obj.first(v,x2,False)
        j2d  = obj.first(v,x2,True)
        y2   = obj.second(v,x2,False)
        y2d  = obj.second(v,x2,True)

        if k == 1:
            m1 = [ [ y2d , -j2d ] , [ -y2 , j2 ] ]
            m2 = [ [ j1 , j1d ] , [ y1 , y1d ] ]
        else:
            m1 = [ [ y2d*eps1 , -j2d*eps1 ] , [ -y2 , j2 ] ]
            m2 = [ [ j1 , j1d*eps2 ] , [ y1 , y1d*eps2 ] ]

        m1 = np.array(m1)
        m2 = np.array(m2)

        m = product(m1,m2)

        return m

    M = np.zeros((2,2,N))
    M[0,0,:] = 1
    M[1,1,:] = 1

    layers = 8
    a = np.cumsum(a)
    for layer in range(layers):
        eps1 = eps[:,layer]
        eps2 = eps[:,layer+1]
        m = subTM(k,v,a[layer],omega,eps1,eps2)
        M = product(M,m)

    return M

# takes the product of two sets of 2x2 matrices
def product(m1,m2):

    m1 = np.array(m1)
    m2 = np.array(m2)
    m = [ [ m1[0,0,:]*m2[0,0,:] + m1[0,1,:]*m2[1,0,:] , m1[1,0,:]*m2[0,0,:] + m1[1,1,:]*m2[1,0,:] ] ,
          [ m1[0,0,:]*m2[0,1,:] + m1[0,1,:]*m2[1,1,:] , m1[1,0,:]*m2[0,1,:] + m1[1,1,:]*m2[1,1,:] ] ]

    return m

# spherical bessel functions of the first and second kind
class bessel:

    def first(self,order,argument,derivative):
        return sp.spherical_jn(order,argument,derivative)

    def second(self,order,argument,derivative):
        return sp.spherical_yn(order,argument,derivative)

# generate random shell thicknesses
def thicknesses():

    seed(42)
    a = []
    for i in range(8):
        a.append( random() )

    r = 40e-9 / np.sum(a)
    a = np.array(a) * r
    return a

# calculate relative permittivity of TiO2
def E_TiO2(Lambda):
    E = 5.913 + 0.2441/( Lambda**2 - 0.0803 )
    return E

# construct N-by-(k+1) permittivity matrix
def eps(Lambda):

    array = []
    for i in range(N):
        E_s     = 2.04
        E0      = 8.85418782e-12
        U0       = 4*np.pi*1e-7
        row     = []
        for j in range(8):
            if (j % 2) == 0:
                row.append(E_s)
            else:
                row.append( E_TiO2(Lambda[i]) )

        row.append(1)
        row = np.array( row ) * E0 * U0
        array.append( row )

    array = np.array( array )

    return array

N          = 1000
order      = 10
step       = np.float( ( 800-400)/N )
a          = np.array([48,45,61,62,38,50,48,56]) * 1e-9
Lambda     = np.arange(400.,800.,step) * 1e-9
omega      = 2*np.pi*3e8*np.reciprocal( Lambda )
omega      = np.transpose( omega )

CS = total_cs(a,omega,Lambda,eps(Lambda))
CS =  CS / (np.pi*np.sum(a)**2)

plt.plot(Lambda,CS)
plt.title('Cross section including $\mu$')
plt.xlabel('$\lambda (nm)$')
plt.ylabel('$\sigma/\pi r^2$')
step = (800*1e-9 - 400*1e-9)/8
ticks = ('400', '450', '500', '550', '600', '650', '700', '750', '800')
plt.xticks(np.arange(400*1e-9, 850*1e-9, step), ticks)
plt.show()





















print("done")
