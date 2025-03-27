import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "ScoreSelfint_sub"))

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from IsContractableType1ReparametrizationParallel import IsContractableType1ReparametrizationParallel
from IsContractableType2ReparametrizationParallel import IsContractableType2ReparametrizationParallel
from EndContractionPriceEst import PriceEstEndContraction
from scipy.interpolate import splrep, PPoly
from max_weight_matching import maxWeightMatching


def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))
def ScoreSelfIntcWeightedMatchingReparametrizisedParallelTMP(selfintc, selfintcu, selfintcv, selfintcs, len, P, P1, RePar1, RePar2, IsAligned, chain1, chain2, maxendcontraction, maxlen, chain_change):
    # Ouptut order:
    #  1 number of local reidemeister type one moves performed
    #  2 total cost of local reidemeister type one moves performed
    #  3 total length of local reidemeister type one moves performed
    #  4 maximal cost of one local reidemeister type one moves performed
    #  5 maximal length of one local reidemeister type one moves performed

    #  6 number of local reidemeister type one moves performed
    #  7 total cost of local reidemeister type two moves performed
    #  8 total length of local reidemeister type two moves performed
    #  9 maximal cost of one local reidemeister type two moves performed
    # 10 maximal length of one local reidemeister type two moves performed
    # 11 morph length (L1) of enddeformations (pre and post morphs)
    # 12 morph length (L1) of the non-contracted part
    # 13 morph length (L1) of the ends during the original morph
    # 14 maximal depth of essencial self-intersections
    # 15 number of essencial self-intersections
    # 16 Sum of signs of the self-intersections
    # 17 Sum of signs of the self-intersections after local reidemeister moves

    # 18 Number of Essential Aligned-Aligned self-intersections
    # 19 Number of Aligned-Aligned self-intersections
    # 20 Number of Essential Aligned-Gap self-intersections
    # 21 Number of Aligned-Gap self-intersections
    # 22 Number of Essential Gap-Gap self-intersections
    # 23 Number of Gap-Gap self-intersections

    # Re 18 to 23: A self-intersection between parameters (a,b) is defined as
    #         aligned-aligned if   IsAlignedSpline(a)+IsAlignedSpline(b) >= 1.5
    #         aligned-gap if 1.5 > IsAlignedSpline(a)+IsAlignedSpline(b) >  0.5
    #         gap-gap     if       IsAlignedSpline(a)+IsAlignedSpline(b) <  0.5

    #C = sparse.find(sparse.tril(selfintc, 0))[2]
    
    
    row, col, data = sparse.find(sparse.tril(selfintc, 0))
    sorted_indices = np.lexsort((row, col))
    C = data[sorted_indices]
    
    row, col, data = sparse.find(sparse.tril(selfintcu, 0))
    sorted_indices = np.lexsort((row, col))
    d = data[sorted_indices]
    
    row, col, data = sparse.find(sparse.tril(selfintcv, 0))
    sorted_indices = np.lexsort((row, col))
    e = data[sorted_indices]
    
    row, col, data = sparse.find(sparse.tril(selfintcs, 0))
    sorted_indices = np.lexsort((row, col))
    A = row[sorted_indices]
    B = col[sorted_indices]
    f = data[sorted_indices]
    
    M = np.column_stack((A-B, A, B, A+d, B+e, C))
    M = np.column_stack((M, 0.001*(3.8**2*np.min([(M[:,4]-1)**2, (M[:,3]-M[:,4])**2/(4*np.pi), (len-M[:,3])**2], axis=0)), f))
    ud_M = M
    M0 = M.copy()
    M1 = M.copy()
    n1 = np.atleast_2d(RePar1).shape[1]
    tck = splrep(np.arange(n1), RePar1, k = 3)
    pp = PPoly.from_spline(tck)
    
    n2 = np.atleast_2d(RePar2).shape[1]
    tck = splrep(np.arange(n2), RePar2, k = 3)
    pp2 = PPoly.from_spline(tck)
    
    M0[:,1:5] = pp(M[:,1:5])
    # M0[:,1] -= 1
    # M0[:,2] -= 1
    # M0[:,3] -= 1
    # M0[:,4] -= 1
    
    M1[:,1:5] = pp2(M[:,1:5])
    # M1[:,1] -= 1
    # M1[:,2] -= 1
    # M1[:,3] -= 1
    # M1[:,4] -= 1
    
    n3 = np.atleast_2d(IsAligned).shape[1]
    tck = splrep(np.arange(n3), IsAligned, k = 3)
    IsAlignedSpline = PPoly.from_spline(tck)
    
    #IsAlignedSpline = spmak(np.arange(n3+2), IsAligned[0])


#############################################
    Nbr = M.shape[0]
    NbrOmega1 = 0
    NbrOmega2 = 0
    cost1 = np.zeros(2)
    maxCost1 = np.zeros(2)
    cost2 = np.atleast_2d(np.zeros(2))
    maxCost2 = np.zeros(2)
    sumsignraw = np.sum(M[:,5]) # sum of crossing changes
    NbrSelfIntc = M.shape[0]
    
    O1 = np.zeros((NbrSelfIntc, 2))
    
    if chain1 == chain2:
        for j in range(NbrSelfIntc): # +1
            tmp = IsContractableType1ReparametrizationParallel(M, M0, M1, j, P, P1, maxlen, chain_change)
            if tmp[0]:
                tmp[0] = np.min([tmp[0], PriceEstEndContraction(M[j,4]-1), PriceEstEndContraction(len-M[j,3]-1)])
            else:
                enddist = np.min([M[j,4]-1, len-M[j,3]])
                if enddist < maxendcontraction:
                    tmp = [PriceEstEndContraction(enddist), enddist*2]
            O1[j,:] = tmp
    paircount = 0
    PotentialType2 = 0
    Obstructed = 0

    O2 = np.zeros((Nbr*(Nbr-1)//2, 4))
    for i in range(Nbr-1):
        for j in range(i+1, Nbr):
            if M[i,5]+M[j,5] == 0: # have opposite signs
                if not (M[j,3] < M[i,4] or M[j,4] > M[i,3]):
                    tmp = IsContractableType2ReparametrizationParallel(M, M0, M1, i, j, P, P1, maxlen, chain_change)
                    PotentialType2 += 1
                    if tmp[0] == 0:
                        Obstructed += 1
                        
                    if tmp[0]:
                        paircount += 1
                        O2[paircount-1,:] = [i, j] + tmp # Indices of self-intersections saved in python format (0-indexing)
    if PotentialType2 > 0:
        print("Procent not solved Type 2: " + str(Obstructed/PotentialType2*100) + "%")
    O2 = O2[:paircount,:]

    
    epsilon = 0.5*(np.sum(O1[:,0]) + np.sum(O2[:,2]))**(-1)
    WVertex = epsilon*O1[:,0] + (O1[:,0] == 0)
    int_O2 = O2[:,0:2].astype(int)
    Wedge = -epsilon*O2[:,2] + WVertex[int_O2[:,0]] + WVertex[int_O2[:,1]]

    edgeData = np.column_stack((O2[:,0:2], Wedge))
    if edgeData.shape[0] == 0:
        print("edgeData is empty!")
        result = np.array([-1])
    else:
        result = np.array(maxWeightMatching(edgeData)[0:]) # maybe remove for inter-chain?
    NbrEssential = 0
    Essentials = []
    Nbr2 = result.shape[0]
    
    for i in range(Nbr2):
        if result[i] > 0:
            if result[i] < i:
                edge = np.where((O2[:,0] == result[i]) & (O2[:,1] == i))[0]
                cost2var = O2[edge,2:4]
                #rows = cost2var.shape[0]
                
                if cost2.shape[0] > cost2var.shape[0]:
                    rows = cost2.shape[0]
                    for j in range(rows):
                        cost2[j,:] = cost2[j,:] + cost2var
                else:
                    rows = cost2var.shape[0]
                    for k in range(rows):
                        cost2var[k,:] = cost2var[k,:] + cost2
                
                cost2 = cost2var
                cost2var = O2[edge,2:4]
                NbrOmega2 += 1
                maxCost2 = np.maximum(maxCost2, cost2var)
        else:
            if WVertex[i] < 1:
                cost1var = O1[i,:]
                cost1 += cost1var
                NbrOmega1 += 1
                maxCost1 = np.maximum(maxCost1, cost1var)
            else:
                NbrEssential += 1
                Essentials.append(i)

    for i in range(Nbr2, Nbr):
        if WVertex[i] < 1:
            cost1var = O1[i,:]
            cost1 += cost1var
            NbrOmega1 += 1
            maxCost1 = np.maximum(maxCost1, cost1var)
        else:
            NbrEssential += 1
            Essentials.append(i)

    slet = IsAlignedSpline(M[:,3]) + IsAlignedSpline(M[:,4])
    AlignedAligned = slet >= 1.5 # Nephew
    AlignedGap = (slet < 1.5) & (slet > 0.5)
    GapGap = slet <= 0.5
    NbrEssentialAlignedAligned = np.sum(AlignedAligned[Essentials])
    NbrAlignedAlignedTotal = np.sum(AlignedAligned)
    NbrEssentialAlignedGap = np.sum(AlignedGap[Essentials])
    NbrAlignedGapTotal = np.sum(AlignedGap)
    NbrEssentialGapGap = np.sum(GapGap[Essentials])
    NbrGapGapTotal = np.sum(GapGap)

    PerformEndDeformations = 0

    if PerformEndDeformations == 0:
        RMSsum = np.sum(np.sqrt(np.sum((P - P1) ** 2, axis=1)))
        if NbrEssential == 0:
            ud = [NbrOmega1, cost1, maxCost1, NbrOmega2, cost2, maxCost2, 0, RMSsum, 0, 0, 0,
                  sumsignraw, 0, NbrEssentialAlignedAligned, NbrAlignedAlignedTotal,
                  NbrEssentialAlignedGap, NbrAlignedGapTotal, NbrEssentialGapGap,
                  NbrGapGapTotal]
            ud_essentials = np.zeros((0, 2))
        else:
            M = M[Essentials, :]
            ud_essentials = M[:, [1, 2]]
            antal = M.shape[0]
            ud = [NbrOmega1, cost1, maxCost1, NbrOmega2, cost2, maxCost2, 0,
                  RMSsum, 0, 0, antal, sumsignraw, np.sum(M[:, 5]),
                  NbrEssentialAlignedAligned, NbrAlignedAlignedTotal,
                  NbrEssentialAlignedGap, NbrAlignedGapTotal, NbrEssentialGapGap,
                  NbrGapGapTotal]
    
    return ud, ud_essentials, ud_M


""" 
IsAligned = np.loadtxt("Test txt/SSIWMRPTMP/IsAligned.txt")
len = np.loadtxt("Test txt/SSIWMRPTMP/len.txt")
maxendcontraction = np.loadtxt("Test txt/SSIWMRPTMP/maxendcontraction.txt")
maxlen = np.loadtxt("Test txt/SSIWMRPTMP/maxlen.txt")
P = np.loadtxt("Test txt/SSIWMRPTMP/P.txt")
P1 = np.loadtxt("Test txt/SSIWMRPTMP/P1.txt")
RePar1 = np.loadtxt("Test txt/SSIWMRPTMP/RePar1.txt")
RePar2 = np.loadtxt("Test txt/SSIWMRPTMP/RePar2.txt")
selfintc = np.loadtxt("Test txt/SSIWMRPTMP/selfintc.txt")
selfintcs = np.loadtxt("Test txt/SSIWMRPTMP/selfintcs.txt")
selfintcu = np.loadtxt("Test txt/SSIWMRPTMP/selfintcu.txt")
selfintcv = np.loadtxt("Test txt/SSIWMRPTMP/selfintcv.txt")
P1org = np.loadtxt("Test txt/SSIWMRPTMP/P1org.txt")
P2org = np.loadtxt("Test txt/SSIWMRPTMP/P2org.txt")

ud,ud_essentials, ud_M = ScoreSelfIntcWeightedMatchingReparametrizisedParallelTMP(selfintc, selfintcu, selfintcv, selfintcs, len, P, P1, RePar1, RePar2, IsAligned, P1org, P2org, maxendcontraction, maxlen)

 """
