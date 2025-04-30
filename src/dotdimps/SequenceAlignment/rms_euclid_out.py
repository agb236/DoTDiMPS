import numpy as np

def rms_euclid_out(P1, P2):
    """
    Computes RMSD after optimal superposition of two point clouds,
    and returns the aligned structures and rotation matrix.

    Parameters:
    P1, P2 (np.ndarray): Arrays of shape (n, 3)

    Returns:
    tuple:
        - p1 (np.ndarray): Center of P1
        - p2 (np.ndarray): Center of P2
        - R (np.ndarray): Optimal rotation matrix (3x3)
        - rms (float): Root-mean-square deviation
    """
    if P1.shape != P2.shape:
        return None, None, None, np.inf

    n = P1.shape[0]

    # Compute centroids
    p1 = np.mean(P1, axis=0)
    p2 = np.mean(P2, axis=0)

    # Center structures
    P1_centered = P1 - p1
    P2_centered = P2 - p2

    # Compute optimal rotation
    C = P1_centered.T @ P2_centered
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(C))
    R = U @ np.diag([1, 1, d]) @ Vt

    # Apply rotation
    P2_rotated = (R @ P2_centered.T).T

    # Compute RMSD
    rms = np.sqrt(np.sum((P1_centered - P2_rotated) ** 2) / n)

    return p1, p2, R, rms
