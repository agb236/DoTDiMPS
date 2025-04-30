import numpy as np

def rms(P1, P2):
    """
    Computes the root-mean-square deviation (RMSD) between two point clouds
    after optimal translation and rotation using SVD.

    Parameters:
    P1, P2 (np.ndarray): Arrays of shape (n, 3)

    Returns:
    float: RMSD value (or np.inf if point sets are mismatched)
    """
    if P1.shape != P2.shape:
        return np.inf

    n = P1.shape[0]

    # Remove translation
    P1_centered = P1 - np.mean(P1, axis=0)
    P2_centered = P2 - np.mean(P2, axis=0)

    # Optimal rotation using SVD
    C = P1_centered.T @ P2_centered
    _, S, _ = np.linalg.svd(C)

    det_C = np.linalg.det(C)
    correction = np.sign(det_C)

    # Compute RMSD
    rmsd = np.sqrt(
        abs(
            (np.sum(P1_centered ** 2) +
             np.sum(P2_centered ** 2) -
             2 * (S[0] + S[1] + correction * S[2])) / n
        )
    )

    return rmsd
