import numpy as np
import MakeDminProteinReparametrizedParallel as mdprp
import MakeDP as mdp

def NEAMReparametrizationParallel(P1, P2, RePar1, RePar2, IsAligned, Smoothning):
    """
    Computes overlap measures for two reparametrized protein structures, evaluating alignment 
    and gaps based on geometric transformations.

    Parameters:
        P1, P2 (np.ndarray): Matrices representing point sets of two proteins.
        RePar1, RePar2 (np.ndarray): Reparametrization arrays for both proteins.
        IsAligned (np.ndarray): Binary vector indicating aligned points.
        Smoothning (int): Determines smoothing method for distance constraints.
    
    Returns:
        tuple: (overlap, overlapaligned, overlapGap, overlapGapWeight)
    """
    # Compute minimal allowed distances between residues
    Dmin = mdprp.MakeDminProteinReparametrizedParallel(RePar1, RePar2, Smoothning)
    
    # Compute pairwise differences for both protein structures
    dP1, dP2 = mdp.MakeDP(P1), mdp.MakeDP(P2)
    l1 = np.linalg.norm(dP1, axis=2)  # Compute Euclidean norms
    ddPNormSqr = np.sum((dP1 - dP2) ** 2, axis=2)
    dot12 = np.sum(dP1 * dP2, axis=2)
    
    # Compute cross-product norms and derived distance measures
    crossPNormSqr = np.sum(np.cross(dP2, dP1, axis=2) ** 2, axis=2)
    dminSqr = crossPNormSqr / ddPNormSqr
    t = (l1 ** 2 - dot12) / ddPNormSqr
    tstar = np.clip(t, 0, 1)  # Ensure values are within [0,1]
    
    # Compute segment-wise minimal distances
    sEffSq = (tstar - t) ** 2
    dminSqrSegment = sEffSq * ddPNormSqr + dminSqr
    overlap = Dmin - np.sqrt(dminSqrSegment)
    overlapalt = Dmin - np.sqrt(np.sum(dP1 ** 2, axis=2))
    overlap[ddPNormSqr < 1.0E-15] = overlapalt[ddPNormSqr < 1.0E-15]
    overlap = np.maximum(overlap, 0)
    
    # Compute alignment-based overlap corrections
    alignedaligned = np.outer(IsAligned, IsAligned)
    overlapaligned = overlap * alignedaligned
    
    # Compute weighting factors for alignment gaps
    weight1 = np.pad(np.diff(RePar1), (1, 0), constant_values=1)
    weight2 = np.pad(np.diff(RePar2), (1, 0), constant_values=1)
    overlapGap = overlap - overlapaligned
    overlapGapWeight = overlapGap * np.outer(weight1, weight2)
    
    return overlap, overlapaligned, overlapGap, overlapGapWeight
