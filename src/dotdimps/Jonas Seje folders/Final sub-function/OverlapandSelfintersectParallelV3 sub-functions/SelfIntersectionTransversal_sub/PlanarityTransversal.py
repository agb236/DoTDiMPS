import numpy as np

def PlanarityTransversal(a0, a1, b0, b1):
    """
    Computes the parameter values where two moving line segments are planar (intersect in 3D space).
    
    Parameters:
        a0, a1 (np.ndarray): 3x2 arrays defining the first line segment at t=0 and t=1.
        b0, b1 (np.ndarray): 3x2 arrays defining the second line segment at t=0 and t=1.
    
    Returns:
        list: 
            - List of real roots t where the line segments are planar (0 <= t <= 1).
            - Corresponding derivative values at those points.
    """
    a, b = a0[:, 0], a0[:, 1] - a0[:, 0]
    da, db = a1[:, 0] - a, a1[:, 1] - a1[:, 0] - b
    c, d = b0[:, 0], b0[:, 1] - b0[:, 0]
    dc, dd = b1[:, 0] - c, b1[:, 1] - b1[:, 0] - d
    
    if (np.dot(da, da) + np.dot(db, db) + np.dot(dc, dc) + np.dot(dd, dd)) < 1.0e-25:
        return [[], []]  # Both line segments do not move
    
    r, dr = c - a, dc - da
    M1 = np.column_stack((dr, r, dr, r))
    M2 = np.column_stack((db, db, b, b))
    d_matrix = np.array([np.transpose(d), np.transpose(dd)])
    M_matrix = np.array([
        M1[1, :] * M2[2, :] - M1[2, :] * M2[1, :],
        M1[2, :] * M2[0, :] - M1[0, :] * M2[2, :],
        M1[0, :] * M2[1, :] - M1[1, :] * M2[0, :]
    ])
    M = np.dot(d_matrix, M_matrix)
    
    # Compute polynomial roots for intersections
    p = np.array([M[1, 0], M[0, 0] + M[1, 1] + M[1, 2], M[0, 1] + M[0, 2] + M[1, 3], M[0, 3]])
    roots = np.roots(p)
    roots = roots[np.isreal(roots)].real  # Keep only real roots
    roots = roots[(roots >= 0) & (roots <= 1)]  # Filter roots within valid range
    
    if roots.size > 0:
        dp = np.array([3, 2, 1]) * p[:3]
        derivatives = np.dot(np.column_stack((roots ** 2, roots, np.ones(len(roots)))), dp)
        return [roots, derivatives]
    return [[], []]
