import dolfinx.fem as xfem
import ufl

class BoundaryCondition():
    def __init__(self, type, marker, values, mesh, V, v, ds, facet_tag):        
        self._type = type
        if type == "Dirichlet":
            u_D = xfem.Function(V)
            u_D.interpolate(values)
            fdim = mesh.topology.dim - 1
            facets = facet_tag.find(marker)
            dofs = xfem.locate_dofs_topological(V, fdim, facets)
            self._bc = xfem.dirichletbc(u_D, dofs)
        elif type == "Neumann":
            self._bc = ufl.inner(values, v) * ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type