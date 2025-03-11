import numpy as np
from intersection_origo_triangle_line_segment import intersection_origo_triangle_line_segment
import d_points2line as dpl

def IsContractableType1ReparametrizationParallel(M, M0, M1, i, P, P1, maxlen, chain_change):
    """
    Determines whether a Type 1 reparametrization is contractable based on geometric conditions.
    
    Parameters:
        M, M0, M1 (np.ndarray): Matrices containing reparametrization data.
        i (int): Index of the point to check.
        P, P1 (np.ndarray): 3D coordinate sets.
        maxlen (float): Maximum allowed reparametrization length.
        chain_change (np.ndarray): Array specifying chain modifications.
    
    Returns:
        list: [contractability_score, looplength] or [0, 0] if obstruction is found.
    """
    # Compute weighted position interpolation
    sav = M0[i, 7]
    P = ((1 - sav) * P + sav * P1).T
    
    # Compute segment bounds
    mint1, maxt1 = M0[i, 4], M0[i, 3]
    mint2, maxt2 = M1[i, 4], M1[i, 3]
    looplength = max(maxt1 - mint1, maxt2 - mint2)
    
    # If loop is too long, return immediately
    if looplength > maxlen:
        return [0, 0]
    
    # Compute average segment position
    mint, maxt = M[i, 4], M[i, 3]
    avt = (mint + maxt) / 2
    n1av, tav = int(np.floor(avt)), avt - np.floor(avt)
    
    # Define bounding indices and segment interpolation
    n1, n2 = int(np.floor(mint)), int(np.ceil(maxt))
    a, b = mint - n1, maxt - np.floor(M[i, 3])
    pts = np.column_stack(((1 - a) * P[:, n1] + a * P[:, n1+1], P[:, (n1+1):(n2)], (1 - b) * P[:, n2-1] + b * P[:, n2]))
    
    # Ensure closed loop (avoid gaps in the chain)
    if np.linalg.norm(pts[:, 0] - pts[:, -1]) > 1e-7:
        print("WARNING: No Intersection detected.")
        return [0, 0]
    
    # Compute central reference and radius check
    center = np.mean(pts[:, :-1], axis=1)
    pts -= center[:, None]
    rdisk = np.max(np.linalg.norm(pts, axis=0))
    pmidt = (1 - tav) * P[:, n1av] + tav * P[:, n1av + 1] - center
    
    # Define start and end positions for intersection checking
    Lstart = P[:, np.r_[0:n1 - 1, n2:P.shape[1] - 1]]
    Lend = P[:, np.r_[1:n1, n2 + 1:P.shape[1]]]
    Lmidt = np.linalg.norm((Lstart + Lend) / 2, axis=0)
    
    # Identify potentially intersecting segments
    LineSegmentLength = np.linalg.norm(Lstart - Lend, axis=0)
    ex = np.where(Lmidt <= rdisk + LineSegmentLength / 2)[0]
    
    # Remove false obstructions based on chain changes
    nums_to_remove1 = chain_change[chain_change < n1]
    nums_to_remove2 = chain_change[chain_change > n2] - (n2 - n1) - 1
    ex = ex[~np.isin(ex, nums_to_remove1.astype(int))]
    ex = ex[~np.isin(ex, nums_to_remove2.astype(int))]
    
    # Check for intersections with triangles
    for j in ex:
        for k in range(pts.shape[1] - 1):
            if intersection_origo_triangle_line_segment(pts[:, [k, k + 1]], Lstart[:, j], Lend[:, j]):
                return [0, 0]
    
    # Compute contractability score
    contractability_score = np.sum(dpl.d_points2line(pts[:, 1:-1], pts[:, 0], pmidt)) * 2
    return [contractability_score, looplength]
