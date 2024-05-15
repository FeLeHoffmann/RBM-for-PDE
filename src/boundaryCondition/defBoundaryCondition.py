import numpy as np


# Define the Boundaries
def outerBoundary(x):
    return np.logical_or(np.logical_or(np.logical_or(np.isclose(x[1], 1), 
                                                     np.isclose(x[1], -1)),
                                       np.logical_or(np.isclose(x[0], 2),  
                                                     np.isclose(x[0], -2))),
                         np.logical_or(np.logical_and(np.isclose(x[1],  0.25),
                                                      np.logical_or(x[0] < -1.5, 
                                                                    x[0] > 1.5)),
                                       np.logical_and(np.isclose(x[1], -0.25), 
                                                      np.logical_or(x[0] < -1.5, 
                                                                    x[0] > 1.5))))


def firstWall(x):
    return np.logical_and(np.logical_or(np.logical_or(np.isclose(x[0], -1.1), 
                                                      np.isclose(x[0], -0.9)),
                                        np.isclose(x[1], -0.25)), 
                          np.logical_and(x[0] > -1.5, x[0] < -0.5))


def secondWall(x):
    return np.logical_and(np.logical_or(np.logical_or(np.isclose(x[0], -0.1), 
                                                      np.isclose(x[0],  0.1)),
                                        np.isclose(x[1], 0.25)), 
                          np.logical_and(x[0] > -0.5, x[0] < 0.5))


def thirdWall(x):
    return np.logical_and(np.logical_or(np.logical_or(np.isclose(x[0], 0.9), 
                                                      np.isclose(x[0], 1.1)),
                                        np.isclose(x[1], -0.25)), 
                          np.logical_and(x[0] > 0.5, x[0] < 1.5))


def inlet(x):
    return np.isclose(x[0], -2.5)


def outlet(x):
    return np.isclose(x[0], 2.5)
