import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
import dolfinx.io as iox
import ufl
import meshio

def sigmoid(x, mu, s):
    # Sigmoid Function (Smooth Step Function)
    #
    #   mu is shift
    #   s is scaling of steep the function is
    #   Alternative to "if x > mu"
      
    return 1 / (1 + ufl.exp(- s * (x - mu)))


def thetaBox(x, y, xMin, xMax, yMin, yMax, scale=25):
    # Combined Sigmoid Function (Smooth Step Box Function)
    #
    #   creates a box with 1 inside [xMin, xMax]x[yMin, yMax] 
    #       and 0 outside
    #   scale is scaling of steep the function is
    #   Alternative to "if xMin < x < xMax and yMin < y < yMin"
    
    sigX = sigmoid(x, xMin, scale) * sigmoid(x, xMax, -scale)
    sigY = sigmoid(y, yMin, scale) * sigmoid(y, yMax, -scale) 
    sigXY = sigX * sigY
    return sigXY    


def circle(x, y, centerX=0., centerY=0., scale=1., ellipseX=1., ellipseY=1.):
    # Create a pointed Vector Field in a circle shape
    #
    #   Center of the Circle at [centerX, centerY]
    #   Radius of the Circle is Scale
    #   Resize Circle to Ellipse with (a,b) = (ellipseX, ellipseY)
    
    r = ufl.sqrt((x - centerX)**2 + (y - centerY)**2)
    theta = ufl.acos((x - centerX) / r)
    
    scale = scale * ufl.sqrt((x - centerX)**2 + (y - centerY)**2 + 0.1)
    
    cx = scale * ellipseX * (-ufl.sign(y - centerY)) * ufl.sin(theta)
    cy = scale * ellipseY * ufl.cos(theta)
    
    return np.array([cx, cy])


def calcB(x, y):
    # Define the flow field b
    #
    #   By Eye we reconstruct the Flow Field. 
    #   We have a wave like flow through the mixer.
    #       Additional we have some turbulences at the inlet and outlet.
    #       We model a reduction in the flow field speed (Friction)
    #
    #   NB: This model is NOT exact, this is just an approximation and
    #       should be refined by solving the Navier-Stokes.
    #   NB: For technical reasons it's not possible to use if-else
    #       statements with ufl/dolfinx functions. That's why we're
    #       using the sigmoid/thetaBox function here. It also gives a 
    #       smoother appearance. 
     
    bx, by = (0, 0)
    
    bx, by = (thetaBox(x, y, -99, -2, -99, 99) * np.array([2, 0]) +     # Inlet Vault
              thetaBox(x, y,   2, 99, -99, 99) * np.array([1, 0]) +     # Outlet Vault
              thetaBox(x, y,  -2, -1,   0, 99) * circle(x, y, centerX=-1.5, centerY= 0.50, scale= 2.00) +                               # Turbulence Inlet
              thetaBox(x, y,  -2,  0, -99,  0) * circle(x, y, centerX=-1.0, centerY=-0.25, scale= 1.75, ellipseX=1, ellipseY=0.75) +
              thetaBox(x, y,  -1,  1,   0, 99) * circle(x, y, centerX= 0.0, centerY= 0.25, scale=-1.50, ellipseX=1, ellipseY=0.75) +
              thetaBox(x, y,   0,  2, -99,  0) * circle(x, y, centerX= 1.0, centerY= 0.25, scale= 1.25, ellipseX=1, ellipseY=0.75) +
              thetaBox(x, y,   1,  2,   0, 99) * circle(x, y, centerX= 1.5, centerY= 0.50, scale= 1.00))                                # Turbulence Outlet
                  
    return bx, by


def returnB(mesh):
    # Calculate and return the b field for the Mixer.
    # We are scaling the vectorfield bei 1/7.5.
    x = ufl.SpatialCoordinate(mesh)
    
    scaling = 7.5
    bx, by = calcB(x[0], x[1])
    b  = ufl.as_vector((1./scaling * bx, 1./scaling * by))
    
    return b


def plotB(fileName):
    # Load Mesh-File (.msh) with meshio instead io.gmesh
    # This has the reason that meshio has the easier interface
    mesh = meshio.read(fileName)

    # Extract Vertices and Faces
    points = mesh.points
    cells = mesh.cells[-1]  


    # Plot the mesh
    # We plot each triangle individually. There's for sure an easier way
    # But ChatGPT gave me this code, it's running, so I'm not touching it.
    plt.figure(figsize=(8, 6))
    midPts = np.zeros((cells.data.shape[0], 3))
    i = 0
    for triangle in cells.data:
        # Plot Triangle
        vertices = points[triangle]
        plt.fill(vertices[:, 0], vertices[:, 1], 'b', edgecolor='k', alpha=.3, linewidth=0.5)

        # Plot Quiver
        midPts[i, :] = np.mean(points[triangle], axis=0)
        midPt = np.mean(points[triangle], axis=0)

        cB = np.vectorize(calcB)
        bx, by = cB(midPt[0], midPt[1])
        bx = bx.astype(np.float64)
        by = by.astype(np.float64)

        plt.quiver(midPt[0], midPt[1], bx, by, scale=75, width=0.0025, headwidth=1, headlength=1, headaxislength=1)
       
        i = i + 1

    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')
    plt.title('Mesh Plot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.plot()
