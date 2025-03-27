import numpy as np

def d_points2line(Ps, P1, P2):
    """
    Computes the shortest distances from a set of points to a line segment defined by two endpoints.
    
    Parameters:
        Ps (np.ndarray): Array of shape (3, n) representing n points in 3D space.
        P1 (np.ndarray): 3D coordinates of the first endpoint of the line segment.
        P2 (np.ndarray): 3D coordinates of the second endpoint of the line segment.
    
    Returns:
        np.ndarray: An array of distances from each point in Ps to the line defined by P1 and P2.
    """
    # Compute unit direction vector of the line segment
    V = P2 - P1
    V /= np.linalg.norm(V)
    
    # Compute distances using cross product
    n = Ps.shape[1]
    Vs = np.tile(V.reshape(-1, 1), (1, n))
    vcross = np.cross(Ps - P1[:, None], Vs, axis=0)
    distances = np.linalg.norm(vcross, axis=0)
    
    return distances
