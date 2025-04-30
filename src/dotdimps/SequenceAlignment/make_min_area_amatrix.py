import numpy as np

def make_min_area_amatrix(Type1, Type2):
    """
    Constructs the minimal area cost matrix A based on Type1 and Type2 cost matrices.
    Inspired by the minArea algorithm (Falicov & Cohen, J. Mol. Biol. 1996).

    Parameters:
    Type1 (np.ndarray): Cost of extending alignment in structure 1 direction
    Type2 (np.ndarray): Cost of extending alignment in structure 2 direction

    Returns:
    np.ndarray: Cost matrix A of shape (n+1, m)
    """
    n, m = Type1.shape[0] + 1, Type2.shape[1]
    A = np.zeros((n, m))

    # First column: cumulative Type1
    A[1:, 0] = np.cumsum(Type1[:, 0])

    # First row: cumulative Type2
    A[0, 1:] = np.cumsum(Type2[0, :])

    for i in range(2, n + m - 1):
        for k in range(max(0, i - n + 1), min(i - 1, m - 1)):
            row = i - k
            col = k + 1
            A[row, col] = min(
                A[row - 1, col] + Type1[row - 1, col],
                A[row, col - 1] + Type2[row, col - 1]
            )

    return A
