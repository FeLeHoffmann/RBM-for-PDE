import numpy as np
import ufl

def sigmoid(x, mu, s):
    return 1 / (1 + ufl.exp(- s * (x - mu)))


def thetaBox(x, y, xMin, xMax, yMin, yMax, scale=25):
    sigX = sigmoid(x, xMin, scale) * sigmoid(x, xMax, -scale)
    sigY = sigmoid(y, yMin, scale) * sigmoid(y, yMax, -scale) 
    sigXY = sigX * sigY
    return sigXY    


def circle(x, y, centerX=0., centerY=0., scale=1., ellipseX=1., ellipseY=1.):
    r = ufl.sqrt((x - centerX)**2 + (y - centerY)**2)
    theta = ufl.acos((x - centerX) / r)
    
    scale = scale * ufl.sqrt((x - centerX)**2 + (y - centerY)**2 + 0.1)
    
    cx = scale * ellipseX * (-ufl.sign(y - centerY)) * ufl.sin(theta)
    cy = scale * ellipseY * ufl.cos(theta)
    
    return np.array([cx, cy])

def circleNP(x, y, centerX=0., centerY=0., scale=1., ellipseX=1., ellipseY=1.):
    r = ufl.sqrt((x - centerX)**2 + (y - centerY)**2)
    theta = ufl.acos((x - centerX) / r)
    
    scale = scale * ufl.sqrt((x - centerX)**2 + (y - centerY)**2 + 0.1)
    
    cx = scale * ellipseX * (-ufl.sign(y - centerY)) * ufl.sin(theta)
    cy = scale * ellipseY * ufl.cos(theta)
    
    return np.array([cx, cy], dtype=np.float64)


def calcB(x, y):
    bx, by = (0, 0)
    
    bx, by = (thetaBox(x, y, -99, -2, -99, 99) * np.array([2, 0]) +     # Inlet Vault
              thetaBox(x, y,   2, 99, -99, 99) * np.array([1, 0]) +     # Outlet Vault
              thetaBox(x, y,  -2, -1,   0, 99) * circle(x, y, centerX=-1.5, centerY= 0.50, scale= 2.00) +
              thetaBox(x, y,  -2,  0, -99,  0) * circle(x, y, centerX=-1.0, centerY=-0.25, scale= 1.75, ellipseX=1, ellipseY=0.75) +
              thetaBox(x, y,  -1,  1,   0, 99) * circle(x, y, centerX= 0.0, centerY= 0.25, scale=-1.50, ellipseX=1, ellipseY=0.75) +
              thetaBox(x, y,   0,  2, -99,  0) * circle(x, y, centerX= 1.0, centerY= 0.25, scale= 1.25, ellipseX=1, ellipseY=0.75) +
              thetaBox(x, y,   1,  2,   0, 99) * circle(x, y, centerX= 1.5, centerY= 0.50, scale= 1.00))
                  
    return bx, by
