import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "Type1 and 2 sub"))

import numpy as np
# import intersection_origo_triangle_line_segment as iotls
# import IntersectionTriangle_LineSegment as itls
import dist_pts_to_line as dpl
import bisect
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from IntersectionOrigo import intersection_origo_triangle_line_segment
import plotly.graph_objects as go

def IsContractableType2ReparametrizationParallel(M, M0, M1, i, makker, P, P1, maxlen, chain_change):
    # Initialize variables
    casea = "Error"
    FindNumberOfOmega1_2Obstructions = 0
    printout = 0
    printoutobstruction = 0

    # Calculate the length of the loop
    leng = np.sum(np.abs(M[i, [3, 4]] - M[makker, [3, 4]]))
    lengRep = np.sum(np.maximum(np.abs(M0[i, [3, 4]] - M0[makker, [3, 4]]), np.abs(M1[i, [3, 4]] - M1[makker, [3, 4]])))

    # If the loop length exceeds the maximum length, return [0, 0]
    if lengRep > maxlen:
        ud = [0, 0]
        return ud

    # Print debug information if printout is enabled
    if printout == 1:
        print([i, makker])
        print(M[[i, makker], :])

    # Get the saved values from M
    ts1 = M[i, 7]
    ts2 = M[makker, 7]
    sav = (ts1 + ts2) / 2

    # Interpolate between P and P1 using the saved value
    P0 = P
    P = ((1 - sav) * P + sav * P1).T

    # Calculate the points for the loop
    Pts1 = ((1 - ts1) * P0 + ts1 * P1).T
    Pts2 = ((1 - ts2) * P0 + ts2 * P1).T
    sa = M[i, 4]
    na = np.floor(sa)
    a = sa - na
    sa2 = M[i, 3]
    na2 = np.floor(sa2)
    a2 = sa2 - na2
    sb = M[makker, 4]
    nb = np.floor(sb)
    b = sb - nb
    sb2 = M[makker, 3]
    nb2 = np.floor(sb2)
    b2 = sb2 - nb2

    # Calculate the line segments
    La1 = (1 - a) * Pts1[:, int(na)] + a * Pts1[:, int(na+1)]
    La2 = (1 - a) * Pts2[:, int(na)] + a * Pts2[:, int(na+1)]
    La12 = (1 - a2) * Pts1[:, int(na2)] + a2 * Pts1[:, int(na2+1)]
    La22 = (1 - a2) * Pts2[:, int(na2)] + a2 * Pts2[:, int(na2+1)]
    Lb1 = (1 - b) * Pts1[:, int(nb)] + b * Pts1[:, int(nb+1)]
    Lb2 = (1 - b) * Pts2[:, int(nb)] + b * Pts2[:, int(nb+1)]
    Lb12 = (1 - b2) * Pts1[:, int(nb2)] + b2 * Pts1[:, int(nb2+1)]
    Lb22 = (1 - b2) * Pts2[:, int(nb2)] + b2 * Pts2[:, int(nb2+1)]

    # Determine the start and end indices for the loop
    mins = np.atleast_2d(np.min(M[[i, makker], 3:5], axis=0))
    maxs = np.atleast_2d(np.max(M[[i, makker], 3:5], axis=0))

    if M[i, 4] == mins[0,1]: 
        istart = i
        islut = makker
    else:
        istart = makker
        islut = i
    n1 = np.floor(M[istart, 4])
    n2 = np.ceil(M[islut, 4])
    a = M[istart, 4] - n1
    b = M[islut, 4] - np.floor(M[islut, 4])
    
    # Create points for the loop
    col1 = La1
    col2 = np.array((1 - a) * P[:, int(n1)] + a * P[:, int(n1)+1])
    if int(n1+1) == int(n2-1):
        col3 = np.array([P[:, int(n1+1)]])
    else:
        col3 = np.array(P[:, int(n1+1):int(n2)])
    col4 = np.array((1 - b) * P[:, int(n2 - 1)] + b * P[:, int(n2)])
    col5 = Lb2
    if len(col3) == 1:
        pts1 = np.concatenate([col1.reshape(-1,1), col2.reshape(-1,1), col3.reshape(-1,1), col4.reshape(-1,1), col5.reshape(-1,1)],axis=1)
    else:
        pts1 = np.concatenate([col1.reshape(-1,1), col2.reshape(-1,1), col3, col4.reshape(-1,1), col5.reshape(-1,1)],axis=1)

    if M[islut, 3] < M[istart, 3]:
        n3 = np.floor(M[islut, 3])
        n4 = np.ceil(M[istart, 3])
        a = M[islut, 3] - np.floor(M[islut, 3])
        b = M[istart, 3] - np.floor(M[istart, 3])
        
        col1 = ((1 - a) * P[:, int(n3)] + a * P[:, int(n3+1)]).reshape(-1,1)
        if int(n3+1) == int(n4-1):
            col2 = np.array([P[:, int(n3+1)]])
        else:
            col2 = (P[:, int(n3+1):int(n4)])
        col3 = ((1 - b) * P[:, int(n4 - 1)] + b * P[:, int(n4)]).reshape(-1,1)
        if len(col2) == 1:
            pts2 = np.concatenate((col1, col3), axis=1)
        else: 
            pts2 = np.concatenate((col1, col2, col3), axis=1)
    else:
        n3 = np.floor(M[istart, 3])
        n4 = np.ceil(M[islut, 3])
        a = M[istart, 3] - np.floor(M[istart, 3])
        b = M[islut, 3] - np.floor(M[islut, 3])

        col1 = ((1 - a) * P[:, int(n3)] + a * P[:, int(n3+1)]).reshape(-1,1)
        if int(n3+1) == int(n4-1):
            col2 = np.array([P[:, int(n3+1)]]).reshape(-1,1)
        else:
            col2 = (P[:, int(n3+1):int(n4)])
        col3 = ((1 - b) * P[:, int(n4 -  1)] + b * P[:, int(n4)]).reshape(-1,1)
        
        pts2 = np.concatenate((col1, col2, col3), axis=1)
        n = pts2.shape[1]
        pts2 = pts2[:, n::-1]
    ns = [n3, n4]
    n3 = np.min(ns)
    n4 = np.max(ns)
    pts = np.concatenate((pts1, pts2), axis=1)

    # Center the points around the origin
    center = (np.sum(pts, axis=1) / pts.shape[1]).reshape(-1,1)
    pts = pts - np.tile(center, (1, pts.shape[1]))
    rdisk = np.max(np.sum(pts ** 2, axis=0)) ** 0.5
    pts = np.concatenate((pts, pts[:, 0].reshape(-1, 1)), axis=1)
    
    # Get the number of triangles
    NbrTriangles = pts.shape[1] - 1
    P = P - np.tile(center, (1, P.shape[1]))
    Lindex = np.concatenate((np.arange(0, int(n1)), np.arange(int(n2), int(n3 - 1)), np.arange(int(n4), P.shape[1] - 1)))
    Lstart = P[:, Lindex]
    Lend = P[:, 1 + Lindex]

    # Calculate the midpoint of the line segments
    Lmidt = np.sqrt(np.sum(((Lstart + Lend) / 2) ** 2, axis=0))
    LineSegmentLength = np.sqrt(np.sum((Lstart - Lend) ** 2, axis=0))
    distdiff = Lmidt - LineSegmentLength / 2
    ex = np.where(distdiff <= rdisk)[0]

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

    if NbrL > 0:
        slet = np.column_stack((ex.T, distdiff[ex]))
        index = np.argsort(slet[:, 1])

        for j in (index):
            for k in range(NbrTriangles):
                if (intersection_origo_triangle_line_segment(pts[:, [k, k+1]], Lstart[:, j], Lend[:, j])):
                    chain1 = bisect.bisect_left(chain_change, na)
                    chain2 = bisect.bisect_left(chain_change, na2)
                    chain3 = bisect.bisect_left(chain_change, ex[j])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter3d( x=[i[0] for i in P[:,int(min(na,nb)-5):int(max(na,nb)+5)].T], 
                                                y=[i[1] for i in P[:,int(min(na,nb)-5):int(max(na,nb)+5)].T], 
                                                z=[i[2] for i in P[:,int(min(na,nb)-5):int(max(na,nb)+5)].T], mode='lines', line=dict(width=9, color = "blue"), name = "Chain" + str(chain1)))
                    
                    fig.add_trace(go.Scatter3d( x=[i[0] for i in P[:,int(min(na2,nb2)-5):int(max(na2,nb2)+5)].T], 
                                                y=[i[1] for i in P[:,int(min(na2,nb2)-5):int(max(na2,nb2)+5)].T], 
                                                z=[i[2] for i in P[:,int(min(na2,nb2)-5):int(max(na2,nb2)+5)].T], mode='lines', line=dict(width=9, color = "red"), name = "Chain" + str(chain2)))
                    
                    fig.add_trace(go.Scatter3d(x=[i[0] for i in (P[:, Lindex][:, ex[j]-5:ex[j]+5]).T],
                                              y=[i[1] for i in (P[:, Lindex][:, ex[j]-5:ex[j]+5]).T],
                                              z=[i[2] for i in (P[:, Lindex][:, ex[j]-5:ex[j]+5]).T],mode='lines', line=dict(width=9, color = "black"), name= "Obstruction from chain" + str(chain3)))

                    for b in range(NbrTriangles):
                        x = list(pts[0, [b, b+1]])+[0]
                        y = list(pts[1, [b, b+1]])+[0]
                        z = list(pts[2, [b, b+1]])+[0]

                        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, opacity=0.4, color='cyan'))
                    
                    x = list(pts[0, [b-1, b]])+[0]
                    y = list(pts[1, [b-1, b]])+[0]
                    z = list(pts[2, [b-1, b]])+[0]

                    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, opacity=0.4, color='cyan'))

                    fig.update_layout(title_text="Chain with intersections")
                    ud = [0, 0]
                    return ud

    # Calculate the distance from the points to the line and return the result
    dists = dpl.d_points2line(pts, pts[:, 0], pts[:, pts1.shape[1]-1])
    ud = [np.sum(dists) * 2, lengRep]

    return ud