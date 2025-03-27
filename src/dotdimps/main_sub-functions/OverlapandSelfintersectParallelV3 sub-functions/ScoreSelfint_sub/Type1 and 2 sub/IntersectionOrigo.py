import numpy as np

def intersection_origo_triangle_line_segment(pts, Lstart, Lslut):
    """
    Determines if a line segment intersects a triangle in 3D space.
    
    Parameters:
        pts (np.ndarray): A (3x3) matrix where columns represent the triangle vertices.
        Lstart (np.ndarray): A (3x1) array representing the start point of the line segment.
        Lslut (np.ndarray): A (3x1) array representing the end point of the line segment.
    
    Returns:
        bool: True if the line segment intersects the triangle, otherwise False.
    """
    # Solve for intersection using barycentric coordinates and segment parameterization
    uvt = np.linalg.solve(np.column_stack((pts, -Lslut + Lstart)), Lstart)
    
    # Check intersection conditions:
    # uvt[0], uvt[1] define barycentric coordinates -> they should be non-negative and sum to at most 1
    # uvt[2] is the segment parameter -> should be between 0 and 1
    intersection = (
        (uvt[0] >= 0) & (uvt[1] >= 0) & (uvt[0] + uvt[1] <= 1) & 
        (uvt[2] >= 0) & (uvt[2] <= 1)
    )
    
    return intersection