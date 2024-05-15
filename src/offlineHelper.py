import numpy as np
import src.boundaryCondition.defBoundaryCondition as defBc


import dolfinx
import dolfinx.mesh as xmesh
import dolfinx.fem as xfem
import dolfinx.fem.petsc as xpetsc
import dolfinx.io as xio
import dolfinx.io as xio


import ufl
from mpi4py import MPI 
import os
import sys

    
def readMesh(fileName):
    mesh, cell_markers, facet_markers = xio.gmshio.read_from_msh(fileName, MPI.COMM_WORLD, gdim=2)
    return mesh, cell_markers, facet_markers


def createFunctionSpace(mesh):
    V = xfem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    return V, u, v


def setUpBc():
    boundaries = [(1, defBc.outerBoundary),
                  (2, defBc.firstWall),
                  (3, defBc.secondWall),
                  (4, defBc.thirdWall),
                  (5, defBc.inlet),
                  (6, defBc.outlet)]

    return boundaries


def tagBc(mesh, boundaries):
    facet_indices, facet_markers = [], []
    
    fdim = mesh.topology.dim - 1
    
    for (marker, locator) in boundaries:
        facets = xmesh.locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    
    facet_tag = xmesh.meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    
    return facet_tag


def saveBc2Mesh(mesh, facet_tag):
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    
    with xio.XDMFFile(mesh.comm, "./mesh/facet_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tag, mesh.geometry)


def constantBcFunction(mu):
    return lambda x: dolfinx.default_scalar_type(mu) * np.ones((x.shape[1], ))

def constantBcValues(mesh, mu):
    return xfem.Constant(mesh, dolfinx.default_scalar_type(mu))
        

