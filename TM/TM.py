#! /usr/bin/python3

import cmath
import numpy as np
from scipy import special as sp

# total cross-section (sum of TE and TM components)
def total_cs(a,omega,eps):
    sigma = 0
    order = 3
    for v in range(order):
        sigma = sigma + SCS(1,v,a,omega,eps) + SCS(2,v,a,omega,eps)

# spherical cross-section
def SCS(k,v,a,omega,eps):
    i      = complex(0,1)
    Lambda = 2*np.pi*3e8/omega
    M      = TM(k,v,a,omega,eps)
    r      = ( M[1,1] - i*M[2,1] )/( M[1,1] + i*M[2,1] )
    sigma  = Lambda**2*abs(1 - r)**2/(8*np.pi)

# transfer matrix
def TM(k,v,a,omega,eps):

    def subTM(k,v,a,omega,eps1,eps2):

        k1   = omega * np.sqrt(eps1)
        k2   = omega * np.sqrt(eps2)
        x1   = np.matmul(k1,a)
        x2   = np.matmul(k2,a)

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

        m = np.matmul(m1,m2)
        return m

    layers = 8
    M = np.identity(2)
    for layer in range(layers):
        a = np.cumsum(a)
        m = subTM(k,v,a,omega,eps1,eps2)
        M = np.matmul(M,m)

    return M

# spherical bessel functions of the first and second kind
class bessel:

    def first(self,order,argument,derivative):
        return sp.spherical_jn(self,order,argument,derivative)

    def second(self,order,argument,derivative):
        return sp.spherical_yn(self,order,argument,derivative)


thickness  = (70 - 30)/7
a          = thickness * np.array([1,1,1,1,1,1,1,1]) * 1e-9
omega      = 2*np.pi*3e8*np.reciprocal( np.array([400.,600.,800.]) * 1e-9 )
omega      = np.transpose(omega)
E0         = 8.85418782e-12
eps        = [ [  ,  ,  ,  ,  ,  ,  ,  , 1 ] ,
               [  ,  ,  ,  ,  ,  ,  ,  , 1 ] ,
               [  ,  ,  ,  ,  ,  ,  ,  , 1 ] ,
               ] * E0

CS = total_cs(a,omega,eps)
print(CS)
