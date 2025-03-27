import numpy as np

def MakeDP(P):
    """
    Computes pairwise differences between 3D coordinate points in P.
    
    Parameters:
        P (np.ndarray): An (n x 3) matrix representing n points in 3D space.
    
    Returns:
        np.ndarray: An (n x n x 3) matrix containing pairwise differences along each axis.
    """
    # Ensure input is an (n x 3) matrix
    if P.shape[1] != 3:
        print("Error: Input matrix must have exactly 3 columns.")
        return np.array([])
    
    # Compute pairwise differences efficiently
    dP = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return dP