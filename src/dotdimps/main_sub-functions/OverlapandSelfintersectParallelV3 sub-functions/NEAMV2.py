import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "NEAM_sub"))

import numpy as np
from ResidueMinDist import MakeDminProteinReparametrizedParallel as mdprp
from PairwiseDiff import MakeDP as mdp

def NEAMReparametrizationParallelV2(P1, P2, RePar1, RePar2, IsAligned, Smoothning):

    # Dmin = mdprp.MakeDminProteinReparametrizedParallel(RePar1, RePar2, Smoothning)
    dP1 = mdp(P1)
    dP2 = mdp(P2)
    l1 = np.sqrt(np.sum(dP1**2, axis=2))
    ddPNormSqr = np.sum((dP1 - dP2)**2, axis=2)
    dot12 = np.sum(dP1 * dP2, axis=2)
    crossPNormSqr = np.sum(np.cross(dP2, dP1, axis=2)**2, axis=2)
    dminSqr = crossPNormSqr / ddPNormSqr
    t = (l1**2 - dot12) / ddPNormSqr
    tstar = np.maximum(np.minimum(t, 1), 0)
    sEffSq = (tstar - t)**2
    DSqr = sEffSq * ddPNormSqr + dminSqr

    D = np.sqrt(DSqr)

    #---------------------------------

    # Calculate the squared distances between consecutive points in P1 and P2
    L1S = np.sum((P1[:-1, :] - P1[1:, :])**2, axis=1)
    L2S = np.sum((P2[:-1, :] - P2[1:, :])**2, axis=1)

    # Calculate the maximum squared distances
    LmaxS = np.maximum(L1S, L2S)
    Lmax = np.sqrt(LmaxS)

    # Test conditions
    test0 = (D[:-1, :] + D[1:, :] <= np.atleast_2d(Lmax).T)  # The spheres are too small / line segment is too long d=0
    test1 = (np.abs(D[:-1, :] - D[1:, :]) > np.atleast_2d(Lmax).T)  # The spheres contain the line segment and do not intersect

    # Calculate d1
    d1 = np.maximum(D[:-1, :], D[1:, :]) - np.atleast_2d(Lmax).T

    # Calculate Xnormalized
    taeller = DSqr[:-1, :] - DSqr[1:, :] + np.atleast_2d(LmaxS).T
    Xnormalized = taeller / (2 * np.atleast_2d(LmaxS).T)
    testx = Xnormalized < 1
    testxcomplement = Xnormalized >= 1

    # Calculate X
    X = np.maximum(0, taeller / (2 * np.atleast_2d(Lmax).T))

    # Calculate f2
    f2 = np.sqrt(np.maximum(DSqr[:-1, :] - X**2, 0))

    # Initialize d2
    Dend = D[1:, :]
    d2 = np.zeros_like(d1)

    # Update d2 based on test conditions
    d2[testx] = f2[testx]
    d2[testxcomplement] = Dend[testxcomplement]
    d2[test1] = d1[test1]
    d2[test0] = 0

    # Calculate the output
    D_check = (d2[:, :-1] + d2[:, 1:])
    L_check = Lmax.reshape(1, -1)
    out = (d2[:, :-1] + d2[:, 1:]) < Lmax.reshape(1, -1)
    out = np.maximum(out, out.T)  # Symmetrization
    #------------------------------------------

    
    # overlap = Dmin - np.sqrt(dminSqrSegment)
    # overlapalt = Dmin - np.sqrt(np.sum(dP1**2, axis=2))
    # overlap[ddPNormSqr < 1.0E-15] = overlapalt[ddPNormSqr < 1.0E-15]
    # overlap = np.maximum(overlap, 0)

    
    # alignedaligned = np.outer(IsAligned, IsAligned)
    # overlapaligned = overlap * alignedaligned
    # tmp1 = np.diff(RePar1)
    # weight1 = 0.5 * (np.concatenate(([1], tmp1)) + np.concatenate((tmp1, [1])))
    # tmp2 = np.diff(RePar2)
    # weight2 = 0.5 * (np.concatenate(([1], tmp2)) + np.concatenate((tmp2, [1])))
    # overlapGap = overlap - overlapaligned
    # overlapGapWeight = overlapGap * np.outer(weight1, weight2)
    

    return out

# P1 = data = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/P1.txt")
# P2 = data = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/P2.txt")
# RePar1 = data = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/RePar1.txt")
# RePar2 = data = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/RePar2.txt")
# IsAligned = data = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/IsAligned.txt")

# out = NEAMReparametrizationParallelV2(P1, P2, RePar1, RePar2, IsAligned, 1)

# print(out)
# print(out.shape)
# print(out[:,5].astype(int))