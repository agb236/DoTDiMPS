import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "Type1 and Type2 sub"))

import numpy as np
import bisect
from intersection_origo_triangle_line_segment import intersection_origo_triangle_line_segment
import IntersectionTriangle_LineSegment as itls
import d_points2line as dp2l

def IsContractableType2ReparametrizationParallel(M, M0, M1, i, makker, P, P1, maxlen, chain_change):
    """
    Determines whether a Type 2 reparametrization is contractable based on geometric conditions.
    
    Parameters:
        M, M0, M1 (np.ndarray): Matrices containing reparametrization data.
        i, makker (int): Indices of the points to check.
        P, P1 (np.ndarray): 3D coordinate sets.
        maxlen (float): Maximum allowed reparametrization length.
        chain_change (np.ndarray): Array specifying chain modifications.
    
    Returns:
        list: [contractability_score, looplength] or [0, 0] if obstruction is found.
    """
    # Compute segment lengths
    lengRep = np.sum(np.maximum(np.abs(M0[i, [3, 4]] - M0[makker, [3, 4]]), np.abs(M1[i, [3, 4]] - M1[makker, [3, 4]])))
    
    # If reparametrization length is too long, return immediately
    if lengRep > maxlen:
        return [0, 0]
    
    # Compute weighted position interpolation
    ts1, ts2 = M[i, 7], M[makker, 7]
    sav = (ts1 + ts2) / 2
    P = ((1 - sav) * P + sav * P1).T
    
    # Compute segment interpolation bounds
    mint, maxt = np.min(M[[i, makker], 3:5], axis=0), np.max(M[[i, makker], 3:5], axis=0)
    n1, n2 = np.floor(mint[1]), np.ceil(maxt[1])
    n3, n4 = np.floor(mint[0]), np.ceil(maxt[0])
    
    # Compute midpoint and radius of potential obstruction region
    center = np.mean(P[:, int(n1):int(n2)], axis=1, keepdims=True)
    P -= center
    rdisk = np.max(np.linalg.norm(P, axis=0))
    
    # Identify potential intersecting segments
    Lindex = np.concatenate((np.arange(0, int(n1)), np.arange(int(n2), int(n3 - 1)), np.arange(int(n4), P.shape[1] - 1)))
    Lstart, Lend = P[:, Lindex], P[:, Lindex + 1]
    
    Lmidt = np.linalg.norm((Lstart + Lend) / 2, axis=0)
    LineSegmentLength = np.linalg.norm(Lstart - Lend, axis=0)
    ex = np.where(Lmidt <= rdisk + LineSegmentLength / 2)[0]
    
    # Remove false obstructions based on chain changes
    nums_to_remove1 = chain_change[chain_change < n1]
    nums_to_remove2 = chain_change[chain_change > n2] - (n2 - n1) - 1
    ex = ex[~np.isin(ex, nums_to_remove1.astype(int))]
    ex = ex[~np.isin(ex, nums_to_remove2.astype(int))]
    
    # Check for intersections with triangles
    for j in ex:
        for k in range(P.shape[1] - 1):
            if intersection_origo_triangle_line_segment(P[:, [k, k + 1]], Lstart[:, j], Lend[:, j]):
                return [0, 0]
    
    # Compute contractability score
    dists = dp2l.d_points2line(P, P[:, 0], P[:, -1])
    return [np.sum(dists) * 2, lengRep]
