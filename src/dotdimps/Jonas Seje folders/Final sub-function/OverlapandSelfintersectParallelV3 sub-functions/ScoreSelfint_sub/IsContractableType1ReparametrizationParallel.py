import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "Type1 and 2 sub"))

import numpy as np
from IntersectionOrigo import intersection_origo_triangle_line_segment
import dist_pts_to_line as dpl

def IsContractableType1ReparametrizationParallel(M, M0, M1, i, P, P1, maxlen, chain_change):
    # Initialize the flag for finding the number of Omega1_2 obstructions
    FindNumberOfOmega1_2Obstructions = 0
    
    # Get the saved value from M0
    sav = M0[i, 7]
    
    # Interpolate between P and P1 using the saved value
    P = ((1 - sav) * P + sav * P1).T
    
    # Get the min and max t values for the first and second loops
    mint1 = M0[i, 4]
    maxt1 = M0[i, 3]
    leng1 = maxt1 - mint1
    mint2 = M1[i, 4]
    maxt2 = M1[i, 3]
    leng2 = maxt2 - mint2
    
    # Determine the maximum loop length
    looplength = max(leng1, leng2)
    
    # If the loop length exceeds the maximum length, return [0, 0]
    if looplength > maxlen:
        return [0, 0]

    # Get the min and max t values for the current loop
    mint = M[i, 4]
    maxt = M[i, 3]
    avt = (mint + maxt) / 2
    n1av = int(np.floor(avt))
    tav = avt - n1av

    n1 = int(np.floor(mint))
    n2 = int(np.ceil(maxt))
    a = mint - n1
    b = maxt - np.floor(M[i, 3])

    # Create points for the loop
    pts = np.column_stack(((1 - a) * P[:, n1] + a * P[:, n1+1], P[:, (n1+1):(n2)], (1 - b) * P[:, n2-1] + b * P[:, n2]))
    
    # Check if the distance between the first and last points is greater than a small threshold
    if np.sum((pts[:, 0] - pts[:, -1]) ** 2) > 10**(-15):
        pointdistance = np.sum((pts[:, 0] - pts[:, -1]) ** 2) ** 0.5
        print('WARNINGNoIntersection distance', pointdistance)
        return [0, 0]

    # Calculate the center of the points and adjust the points to be centered around the origin
    center = np.sum(pts[:, 0:-1], axis=1) / (pts.shape[1] - 1)
    pts = pts - np.tile(center.reshape(-1, 1), (1, pts.shape[1]))
    
    # Calculate the radius of the disk
    rdisk = np.max(np.sum(pts ** 2, axis=0) ** 0.5)
    
    # Calculate the midpoint of the loop
    pmidt = (1 - tav) * P[:, n1av] + tav * P[:, n1av + 1] - center

    # Get the number of triangles
    NbrTriangles = pts.shape[1] - 1
    
    # Adjust P to be centered around the origin
    P = P - np.tile(center.reshape(-1, 1), (1, P.shape[1]))
    
    # Create line segments
    Lstart = P[:, np.r_[0:n1 - 1, n2:P.shape[1] - 1]]
    Lend = P[:, np.r_[1:n1, n2 + 1:P.shape[1]]]
    
    # Calculate the midpoint of the line segments
    Lmidt = np.sum(((Lstart + Lend) / 2) ** 2, axis=0) ** 0.5

    # Calculate the length of the line segments
    LineSegmentLength = np.sum((Lstart - Lend) ** 2, axis=0) ** 0.5
    
    # Find the indices of the line segments that are within the disk
    ex = np.where(Lmidt <= rdisk + LineSegmentLength / 2)[0]

    # Remove false lines
    nums_to_remove1 = chain_change[chain_change < n1]
    nums_to_remove2 = chain_change[chain_change > n2] - (n2 - n1) - 1
    ex = ex[~np.isin(ex, nums_to_remove1.astype(int))]
    ex = ex[~np.isin(ex, nums_to_remove2.astype(int))]

    # Update the line segments to only include the valid ones
    Lstart = Lstart[:, ex]
    Lend = Lend[:, ex]
    NbrL = Lstart.shape[1]
    NbrIntc = 0

    # Check for intersections
    if NbrL > 0:
        if FindNumberOfOmega1_2Obstructions:
            for j in range(NbrL):
                for k in range(NbrTriangles):
                    NbrIntc += intersection_origo_triangle_line_segment(pts[:, [k, k + 1]], Lstart[:, j], Lend[:, j])
        else:
            slet = np.column_stack((ex, Lmidt[ex]))
            index = np.argsort(slet[:, 1])
            for j in index:
                for k in range(NbrTriangles):
                    if intersection_origo_triangle_line_segment(pts[:, [k, k + 1]], Lstart[:, j], Lend[:, j]):
                        return [0, 0]

    # If there are intersections, return [0, 0]
    if NbrIntc > 0:
        return [0, 0]

    # Calculate the distance from the points to the line and return the result
    return [np.sum(dpl.d_points2line(pts[:, 1:-1], pts[:, 0], pmidt)) * 2, looplength]