import numpy as np

def make_dp_nm(P1, P2):
    """
    Constructs a (n x m x 3) array of difference vectors between P1 and P2.

    Parameters:
    P1 (np.ndarray): Array of shape (n, 3)
    P2 (np.ndarray): Array of shape (m, 3)

    Returns:
    np.ndarray: Difference vectors of shape (n, m, 3) such that:
                result[i, j, :] = P1[i, :] - P2[j, :]
    """
    if P1.shape[1] != 3 or P2.shape[1] != 3:
        raise ValueError("Input arrays must have shape (N, 3)")

    n, m = P1.shape[0], P2.shape[0]
    dP = np.zeros((n, m, 3))
    for i in range(3):
        dP[:, :, i] = np.expand_dims(P1[:, i], axis=1) - P2[:, i]

    return dP
