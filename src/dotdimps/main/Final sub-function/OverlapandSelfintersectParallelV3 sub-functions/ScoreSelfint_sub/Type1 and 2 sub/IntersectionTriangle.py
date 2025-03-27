import numpy as np
from scipy.linalg import lstsq

def intersection_triangle_line_segment(p0, p1, p2, Lstart, Lslut):
    """
    Determines whether a line segment intersects a triangle in 3D space using least squares.
    
    Parameters:
        p0, p1, p2 (np.ndarray): 3D coordinates of the triangle vertices.
        Lstart, Lslut (np.ndarray): 3D coordinates defining the start and end of the line segment.
    
    Returns:
        tuple:
            - int: 1 if intersection occurs, 0 otherwise.
            - np.ndarray: Intersection parameters (barycentric and segment parameter).
    """
    # Construct the system of equations for solving intersection
    A = np.array([p1 - p0, p2 - p0, -Lslut + Lstart]).T
    b = (Lstart - p0)
    
    # Solve for intersection parameters (u, v, t)
    uvt, _, _, _ = lstsq(A, b, rcond=None)
    uvt = uvt.reshape(-1, 1)
    
    # Check if intersection conditions are met
    intersection = (
        (uvt[0, 0] >= 0) and 
        (uvt[1, 0] >= 0) and 
        (uvt[0, 0] + uvt[1, 0] <= 1) and 
        (0 <= uvt[2, 0] <= 1)
    )
    
    return int(intersection), uvt
