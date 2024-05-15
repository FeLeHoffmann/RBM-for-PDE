### Helper File for Advection Diffusion Reaction ###
####################################################
import src.flowField.flowField as calcB
import src.boundaryCondition.defBoundaryCondition as defBc
import src.boundaryCondition.BoundaryCondition as bc

import numpy as np
import scipy

from mpi4py import MPI

import dolfinx 
from dolfinx.io import gmshio
import dolfinx.fem as xfem
import dolfinx.mesh as xmesh
import dolfinx.plot as xplot
import dolfinx.io as xio
import dolfinx.fem.petsc as xpetsc

from petsc4py import PETSc

import pyvista
import ufl

def readMesh(fileName):
    mesh, cell_markers, facet_markers = gmshio.read_from_msh(fileName, MPI.COMM_WORLD, gdim=2)
    return mesh, cell_markers, facet_markers

def createFunctionSpace(mesh):
    V = xfem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    return V, u, v

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
    
    return facet_tag, fdim

def saveBcMesh(mesh, facet_tag):
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with xio.XDMFFile(mesh.comm, "./mesh/facet_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tag, mesh.geometry)
        
def constantBcFunction(mu):
    return lambda x: dolfinx.default_scalar_type(mu) * np.ones((x.shape[1], ))

def constantBcValues(mesh, mu):
    return xfem.Constant(mesh, dolfinx.default_scalar_type(mu))

def performBoundaryConditions(boundary_conditions, FF):
    bcs = []
    for condition in boundary_conditions:
        if condition.type == "Dirichlet":
            bcs.append(condition.bc)
        else:
            FF -= condition.bc
            
    return FF, bcs

def plotter(uh, V, name):
    pyvista.start_xvfb()
    pyvista_cells, cell_types, geometry = xplot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
    grid.point_data["u"] = uh.x.array
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()
    plotter.add_text("uh", position="upper_edge", font_size=14, color="black")
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.save_graphic("./solutions/" + name + ".pdf")

def overhead(mesh):
    # Define weak formulation
    x = ufl.SpatialCoordinate(mesh)
    V, u, v = createFunctionSpace(mesh)

    scaling = 7.5
    bx, by = calcB.calcB(x[0], x[1])
    b  = ufl.as_vector((1./scaling * bx, 1./scaling * by))

    # Set Boundary Conditions
    boundaries = [(1, defBc.outerBoundary),
                  (2, defBc.firstWall),
                  (3, defBc.secondWall),
                  (4, defBc.thirdWall),
                  (5, defBc.inlet),
                  (6, defBc.outlet)]


    # Tag the boundaries
    facet_tag, fdim = tagBc(mesh, boundaries)

    # Write Boundaries to Mesh File
    saveBcMesh(mesh, facet_tag)

    # Integration measure
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)  
    
    return u, v, b, facet_tag, V, ds, fdim

def calcBc(mesh, u, v, facet_tag, V, ds, fdim):
    bc1 = bc.BoundaryCondition("Dirichlet", 1, constantBcFunction(0),     mesh, V, v, ds, facet_tag)
    bc5 = bc.BoundaryCondition("Dirichlet", 5, constantBcFunction(0),     mesh, V, v, ds, facet_tag)
    bc6 = bc.BoundaryCondition("Neumann"  , 6, constantBcValues(mesh, 0), mesh, V, v, ds, facet_tag)
    
    return bc1, bc5, bc6

def calcPDE(mu, mesh, u, v, b, facet_tag, V, ds, fdim, bc1, bc5, bc6, plotting=False):
    k  = xfem.Constant(mesh, dolfinx.default_scalar_type(1/mu[3]))
    FF = (k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + 
          ufl.inner(b, ufl.grad(u)) * v * ufl.dx)

    # Define Boundary Condition Functions
    boundary_conditions = [bc1, # outer Boundary
                           bc.BoundaryCondition("Dirichlet", 2, constantBcFunction(mu[0]), mesh, V, v, ds, facet_tag), # first Wall
                           bc.BoundaryCondition("Dirichlet", 3, constantBcFunction(mu[1]), mesh, V, v, ds, facet_tag), # second Wall
                           bc.BoundaryCondition("Dirichlet", 4, constantBcFunction(mu[2]), mesh, V, v, ds, facet_tag), # third Wall
                           bc5, # inlet
                           bc6] # outlet

    FF, bcs = performBoundaryConditions(boundary_conditions, FF)

    # Solve
    a = xfem.form(ufl.lhs(FF))
    L = xfem.form(ufl.rhs(FF))
    
    A = xpetsc.assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = xpetsc.create_vector(L)
    xpetsc.assemble_vector(b, L)
    xpetsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    xpetsc.set_bc(b, bcs)
    
    ai, aj, av = A.getValuesCSR()
    A_csr = scipy.sparse.csr_matrix((av, aj, ai))
    b_csr = b.getArray()
    
    
    uh = xfem.Function(V)
    
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    solver.solve(b, uh.vector)
    
    # Save Plotter
    if plotting:
        plotter(uh, V, "solution_" + str(np.round(mu[0], 2)) + "_" 
                                          + str(np.round(mu[1], 2)) + "_" 
                                          + str(np.round(mu[2], 2)) + "_" 
                                          + str(np.round(mu[3], 2)) )
    
    return uh.x.array, A_csr, b_csr