import numpy as np
from scipy.interpolate import splrep, PPoly

def MakeDminProteinReparametrizedParallel(RePar1, RePar2, Smoothning):
    """
    Computes the minimal allowed distance between residues (i, j) based on 
    the smaller arclength distance between the i-th and j-th points 
    in two given reparametrizations.

    Parameters:
        RePar1 (np.ndarray): First reparametrization array.
        RePar2 (np.ndarray): Second reparametrization array.
        Smoothning (int): Determines which distance parameters to use.
    
    Returns:
        np.ndarray: Matrix of minimal allowed distances between residues.
    """
    SCALEFACTOR = 1  # Scaling factor (default 1)
    nbr_points = len(RePar1)
    
    # Ensure both reparametrizations have the same length
    if nbr_points != len(RePar2):
        print("Error: Function requires equal length reparametrizations.")
        return None
    
    # Define minimal distances based on smoothing parameter
    if Smoothning == 1:
        default_values = np.array([1.0, 2.1, 3.0, 3.4, 3.6, 3.7, 3.7])
    else:
        default_values = np.array([2.8, 4.5, 3.86, 3.47, 3.52, 3.48, 3.6])
    
    # Extend minimal distances to match the number of points
    mind = np.full(nbr_points, 3.7)
    mind[:min(7, nbr_points)] = default_values[:min(7, nbr_points)]
    mind *= SCALEFACTOR  # Apply scaling factor
    
    # Create spline representation of minimal distances
    tck = splrep(np.arange(nbr_points + 1), np.hstack((0, mind)), k=3)
    pp = PPoly.from_spline(tck)
    
    # Compute absolute differences in arclength distances
    diff_arclength1 = np.abs(np.subtract.outer(RePar1, RePar1))
    diff_arclength2 = np.abs(np.subtract.outer(RePar2, RePar2))
    
    # Compute minimal allowed distance matrix
    Dminalt = pp(np.minimum(diff_arclength1, diff_arclength2))
    
    return Dminalt