import numpy as np

def distPP(p1, p2):
    """
    Computes the Euclidean distance between two points in 3D space.
    
    Parameters:
        p1 (np.ndarray): A (3,) array representing the first point.
        p2 (np.ndarray): A (3,) array representing the second point.
    
    Returns:
        float: The Euclidean distance between p1 and p2.
    """
    return np.linalg.norm(p2 - p1)