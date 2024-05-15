import dolfinx.fem as xfem
import dolfinx.fem.petsc as xpetsc
import ufl
import scipy
import numpy as np

import src.helperNS as helper


def calcAA(mesh):
    u, v, b, facet_tag, V, ds, fdim = helper.overhead(mesh)
      
    ### --- --- ###
    aa1 = xfem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    
    AA1 = xpetsc.assemble_matrix(aa1)
    AA1.assemble()
    
    ai, aj, av = AA1.getValuesCSR()
    AA1_csr = scipy.sparse.csr_matrix((av, aj, ai))
    AA1 = scipy.sparse.csc_matrix(AA1_csr)
    ### --- --- ###
    aa2 = xfem.form(ufl.inner(b, ufl.grad(u)) * v * ufl.dx)
    
    AA2 = xpetsc.assemble_matrix(aa2)
    AA2.assemble()
    
    ai, aj, av = AA2.getValuesCSR()
    AA2_csr = scipy.sparse.csr_matrix((av, aj, ai))
    AA2 = scipy.sparse.csc_matrix(AA2_csr)
    ### --- --- ###
    
    return AA1, AA2
  

def calcThetaA():
    thetaA1 = lambda mu: 1. / mu[3]
    thetaA2 = lambda mu: 1
    
    return thetaA1, thetaA2


def helperF(x, facets):
    values = np.zeros((x.shape[1], ))
    values[facets] = 1
    
    return values


def calcUD(V, facet_tag, num):
    facets = facet_tag.find(num)
    u_D = xfem.Function(V)
    u_D.interpolate(lambda x: helperF(x, facets))
    return u_D

def calcFPartial(phi, v, b):
    FF1 = xfem.form(- ufl.inner(ufl.grad(phi), ufl.grad(v)) * ufl.dx)
    ff1 = xpetsc.create_vector(FF1)
    xpetsc.assemble_vector(ff1, FF1)
    
    FF2 = xfem.form(- ufl.inner(b, ufl.grad(phi)) * v * ufl.dx)
    ff2 = xpetsc.create_vector(FF2)
    xpetsc.assemble_vector(ff2, FF2)
    
    return ff1, ff2


def calcFF(mesh):
    u, v, b, facet_tag, V, ds, fdim = helper.overhead(mesh)
    
    # Should be equivalent to 
    #   f1 = - ufl.inner(values, v) * ds(1)
    u_D1 = calcUD(V, facet_tag, 1)
    u_D2 = calcUD(V, facet_tag, 2)
    u_D3 = calcUD(V, facet_tag, 3)
    
    ff1, ff2 = calcFPartial(u_D1, v, b)
    ff3, ff4 = calcFPartial(u_D2, v, b)
    ff5, ff6 = calcFPartial(u_D3, v, b)
    
    return ff1, ff2, ff3, ff4, ff5, ff6
        
    
def calcThetaF():
    thetaF1 = lambda mu: mu[0] / mu[3]
    thetaF2 = lambda mu: mu[0]
    thetaF3 = lambda mu: mu[1] / mu[3]
    thetaF4 = lambda mu: mu[1]
    thetaF5 = lambda mu: mu[2] / mu[3]
    thetaF6 = lambda mu: mu[2]
    
    return thetaF1, thetaF2, thetaF3, thetaF4, thetaF5, thetaF6