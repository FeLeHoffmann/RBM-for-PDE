import numpy as np
import matplotlib.pyplot as plt

import pyvista
import dolfinx
import dolfinx.mesh as xmesh
import dolfinx.fem as xfem
import dolfinx.fem.petsc as xpetsc
import dolfinx.io as xio
import dolfinx.plot as xplot

import ufl
def plotFunctions(sourceTerm, dirichletBoundary, alpha, beta):
    # Plots the functions "sourceTerm" and "dirichletBoundary"
    #   on the unit square [-1, 1]² with parameters alpha / beta.
    ##############################################################
    
    ### Define Domain
    x = np.linspace(0, 1, 250)
    y = np.linspace(0, 1, 250)
    
    XX, YY = np.meshgrid(x, y)
    
    
    ### Evaluate Functions
    f  = sourceTerm([XX, YY], alpha)
    ud = dirichletBoundary([XX, YY], beta)
    
    
    ### Plot everything
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))
    
    axs[0].imshow(f, extent=[0,1,0,1])
    axs[0].set_title(f"Source Function - $f(x)$")
    axs[1].imshow(ud, extent=[0,1,0,1])
    axs[1].set_title(f"Boundary Condition - $u_D(x)$")
    
    plt.show()


def plotTriangulation(domain):
    # Plot the domain
    ###########################
    
    pyvista.start_xvfb()
    pyvista.set_jupyter_backend('html')
    
    tdim = domain.topology.dim
    fdim = tdim - 1

    topology, cell_types, geometry = xplot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.show()
    
    
def plotSolution(uh, V):
    pyvista.start_xvfb()
    pyvista.set_jupyter_backend('html')
    
    pyvista_cells, cell_types, geometry = xplot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()
    plotter.add_text("uh", position="upper_edge", font_size=14, color="black")
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.show()
    plotter.save_graphic("./" + "test" + ".pdf")


def offlineComputation():
    # Perform all computations which are independent 
    #   of mu. THese can be performed offline. 
    #################################################
    
    import src.offlineHelper as offH
    import src.flowField.flowField as flowField

    # Read in the mesh
    fileName = "src/mesh/mixer.msh"
    mesh, cell_markers, facet_markers =  offH.readMesh(fileName)

    # Create FunctionSpace & Test-/Trialfunction & B-Field
    V, u, v = offH.createFunctionSpace(mesh)
    b = flowField.returnB(mesh)
    
    # Define and Perform Boundaries
    boundaries = offH.setUpBc()
    facet_tag = offH.tagBc(mesh, boundaries)
    offH.saveBc2Mesh(mesh, facet_tag)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag) 

    return V, u, v, b, mesh, ds, facet_tag


def onlineComputation(mu, V, u, v, b, mesh, ds, facet_tag):
    # Perform all computations which are depending 
    #   on mu. These need to be performed after mu is feeded. 
    #########################################################
    
    import src.onlineHelper as onH
    
    # Define Weak Formulation
    FF = onH.defineWeakForm(mu, u, v, b)

    # Apply Boundary Conditions
    boundary_conditions = onH.evalBc(mu, mesh, V, v, ds, facet_tag)
    FF, bcs = onH.performBoundaryConditions(boundary_conditions, FF)
    
    ### Solve the Problem
    # Define LHS / RHS
    a = ufl.lhs(FF)
    L = ufl.rhs(FF)
    
    # Specify Solver
    problem = xpetsc.LinearProblem(a, L, bcs=bcs)
    
    # Solve the actual problem
    uh = problem.solve()

    return uh
