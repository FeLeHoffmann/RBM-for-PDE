import src.boundaryCondition.BoundaryCondition as bc
import src.offlineHelper as oH

import ufl


def evalBc(mu, mesh, V, v, ds, facet_tag):
    boundary_conditions = [bc.BoundaryCondition("Dirichlet", 1, oH.constantBcFunction(0)     , mesh, V, v, ds, facet_tag),     # outer Boundary
                           bc.BoundaryCondition("Dirichlet", 2, oH.constantBcFunction(mu[0]) , mesh, V, v, ds, facet_tag), # first Wall
                           bc.BoundaryCondition("Dirichlet", 3, oH.constantBcFunction(mu[1]) , mesh, V, v, ds, facet_tag), # second Wall
                           bc.BoundaryCondition("Dirichlet", 4, oH.constantBcFunction(mu[2]) , mesh, V, v, ds, facet_tag), # third Wall
                           bc.BoundaryCondition("Dirichlet", 5, oH.constantBcFunction(0)     , mesh, V, v, ds, facet_tag),     # inlet
                           bc.BoundaryCondition("Neumann"  , 6, oH.constantBcValues(mesh, 0) , mesh, V, v, ds, facet_tag)] # outlet

    return boundary_conditions


def performBoundaryConditions(boundary_conditions, FF):
    bcs = []
    
    for condition in boundary_conditions:
        if condition.type == "Dirichlet":
            bcs.append(condition.bc)
        elif condition.type == "Neumann":
            FF -= condition.bc
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(condition.type))
            
    return FF, bcs


def defineWeakForm(mu, u, v, b):
    # Diffusion Part
    FF1 = 1/mu[3] * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # Flow Field Part
    FF2 = ufl.inner(b, ufl.grad(u)) * v * ufl.dx

    FF = FF1 + FF2

    return FF