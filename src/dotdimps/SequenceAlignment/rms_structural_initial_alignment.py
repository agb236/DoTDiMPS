import numpy as np
from rms import rms
from rms_euclid_out import rms_euclid_out

def rms_structural_initial_alignment(P1, P2, maxgaplength=0):
    """
    Performs exhaustive search to align the shorter structure (P2) optimally
    within the longer structure (P1), allowing one inner gap of length <= maxgaplength.

    Parameters:
    P1, P2 (np.ndarray): Arrays of shape (n, 3) and (m, 3)
    maxgaplength (int): Maximum allowed gap length in alignment

    Returns:
    tuple:
        - P1 (np.ndarray): Centered version of the first full structure
        - P (np.ndarray): Aligned (rotated and centered) version of P2
        - I1 (tuple): Start and end indices in P1
        - I2 (tuple): Start and end indices in P2
        - rmsmin (float): Minimum RMSD found
    """
    n, m = len(P1), len(P2)
    flip = False

    # Ensure P1 is the longer chain
    if n < m:
        P1, P2 = P2, P1
        n, m = m, n
        flip = True

    rmsmin = np.inf
    k1min = 0
    k2min = 0
    jmin = 0

    for k1 in range(n - m + 1):
        current_rms = rms(P1[k1:k1 + m], P2)
        if current_rms < rmsmin:
            k1min, k2min, jmin = k1, 0, 0
            rmsmin = current_rms

        for k2 in range(1, min(n - m - k1 + 1, maxgaplength + 1)):
            for j in range(1, m):
                part1 = P1[k1:k1 + j]
                part2 = P1[k1 + k2 + j:k1 + k2 + m]
                if part1.shape[0] + part2.shape[0] == m:
                    P1_subset = np.vstack((part1, part2))
                    current_rms = rms(P1_subset, P2)
                    if current_rms < rmsmin:
                        k1min, k2min, jmin = k1, k2, j
                        rmsmin = current_rms

    if k2min == 0:
        P1_aligned = P1[k1min:k1min + m]
    else:
        P1_aligned = np.vstack((
            P1[k1min:k1min + jmin],
            P1[k1min + k2min + jmin:k1min + k2min + m]
        ))

    p1_center, p2_center, R, _ = rms_euclid_out(P1_aligned, P2)

    # Center full P1 and P2
    P1_centered = P1 - p1_center
    P2_centered = P2 - p2_center
    P_rotated = (R @ P2_centered.T).T

    I1 = (k1min, k1min + m)
    I2 = (0, m)

    if k2min != 0:
        I1 = (k1min, k1min + jmin + (m - jmin))
        I2 = (0, m)

    if flip:
        # Swap back if we flipped input
        P1_centered, P_rotated = P_rotated, P1_centered
        I1, I2 = I2, I1

    return P1_centered, P_rotated, I1, I2, rmsmin
