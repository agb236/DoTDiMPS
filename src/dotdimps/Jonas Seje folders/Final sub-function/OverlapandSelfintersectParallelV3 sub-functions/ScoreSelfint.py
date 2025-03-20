import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "ScoreSelfint_sub"))


import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from IsContractableType1ReparametrizationParallel import IsContractableType1ReparametrizationParallel
from IsContractableType2ReparametrizationParallel import IsContractableType2ReparametrizationParallel
from PriceEstEndContraction import PriceEstEndContraction
from scipy.interpolate import splrep, PPoly
from distPP import distPP
from maxWeightMatching import maxWeightMatching

def ScoreSelfIntcWeightedMatchingReparametrizisedParallelTMP(selfintc, selfintcu, selfintcv, selfintcs, length, P, P1, RePar1, RePar2, IsAligned, chain1, chain2, maxendcontraction, maxlen, chain_change):
    """
    Evaluates and scores self-intersections in a reparametrized matching scenario, incorporating local transformations.
    
    Parameters:
        selfintc, selfintcu, selfintcv, selfintcs: Sparse matrices of self-intersection properties.
        length: Length parameter for the transformations.
        P, P1: 3D coordinate representations.
        RePar1, RePar2: Reparametrization arrays.
        IsAligned: Array indicating aligned structures.
        chain1, chain2: Chain indices for evaluation.
        maxendcontraction: Maximum allowed end contraction.
        maxlen: Maximum allowed length for reparametrization.
        chain_change: Array specifying chain changes.
    
    Returns:
        tuple: (score metrics, essential self-intersections, transformed matrix M)
    """
    # Extract sparse matrix data in sorted order
    row, col, data = sparse.find(sparse.tril(selfintc, 0))
    sorted_indices = np.lexsort((row, col))
    C = data[sorted_indices]
    
    row, col, data = sparse.find(sparse.tril(selfintcu, 0))
    d = data[np.lexsort((row, col))]
    
    row, col, data = sparse.find(sparse.tril(selfintcv, 0))
    e = data[np.lexsort((row, col))]
    
    row, col, data = sparse.find(sparse.tril(selfintcs, 0))
    f = data[np.lexsort((row, col))]
    A, B = row[np.lexsort((row, col))], col[np.lexsort((row, col))]
    
    # Construct matrix M with self-intersection properties
    M = np.column_stack((A-B, A, B, A+d, B+e, C))
    min_values = np.min([(M[:,4]-1)**2, (M[:,3]-M[:,4])**2/(4*np.pi), (length-M[:,3])**2], axis=0)
    M = np.column_stack((M, 0.001*(3.8**2 * min_values), f))
    M0, M1 = M.copy(), M.copy()
    
    # Compute interpolated parameterizations
    pp1 = PPoly.from_spline(splrep(np.arange(RePar1.shape[0]), RePar1, k=3))
    pp2 = PPoly.from_spline(splrep(np.arange(RePar2.shape[0]), RePar2, k=3))
    M0[:,1:5] = pp1(M[:,1:5])
    M1[:,1:5] = pp2(M[:,1:5])
    
    # Compute self-intersection metrics
    Nbr = M.shape[0]
    cost1, maxCost1 = np.zeros(2), np.zeros(2)
    cost2, maxCost2 = np.zeros((1, 2)), np.zeros(2)
    sumsignraw = np.sum(M[:,5])
    
    O1 = np.zeros((Nbr, 2))
    for j in range(Nbr):
        tmp = IsContractableType1ReparametrizationParallel(M, M0, M1, j, P, P1, maxlen, chain_change)
        if tmp[0]:
            tmp[0] = min(tmp[0], PriceEstEndContraction(M[j,4]-1), PriceEstEndContraction(length-M[j,3]-1))
        O1[j,:] = tmp
    
    # Type 2 reparametrization analysis
    O2 = np.zeros((Nbr*(Nbr-1)//2, 4))
    paircount = 0
    for i in range(Nbr-1):
        for j in range(i+1, Nbr):
            if M[i,5] + M[j,5] == 0 and not (M[j,3] < M[i,4] or M[j,4] > M[i,3]):
                tmp = IsContractableType2ReparametrizationParallel(M, M0, M1, i, j, P, P1, maxlen, chain_change)
                if tmp[0]:
                    O2[paircount, :] = [i, j] + tmp
                    paircount += 1
    O2 = O2[:paircount, :]
    
    # Compute weighted matching
    epsilon = 0.5 * (np.sum(O1[:,0]) + np.sum(O2[:,2]))**-1
    WVertex = epsilon * O1[:,0] + (O1[:,0] == 0)
    Wedge = -epsilon * O2[:,2] + WVertex[O2[:,0].astype(int)] + WVertex[O2[:,1].astype(int)]
    
    edgeData = np.column_stack((O2[:,0:2], Wedge))
    result = maxWeightMatching(edgeData) if edgeData.shape[0] > 0 else np.array([-1])
    
    # Calculate essential intersections
    Essentials = [i for i in range(result.shape[0]) if result[i] < 0]
    ud_essentials = M[Essentials, [1, 2]] if Essentials else np.zeros((0, 2))
    
    # Compute final score metrics
    RMSsum = np.sum(np.sqrt(np.sum((P - P1) ** 2, axis=1)))
    ud = [len(Essentials), RMSsum, sumsignraw, np.sum(M[:, 5])]
    return ud, ud_essentials, M
