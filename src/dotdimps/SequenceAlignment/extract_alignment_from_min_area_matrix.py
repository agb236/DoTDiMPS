import numpy as np

def extract_alignment_from_min_area_matrix(A):
    """
    Extracts optimal alignment path and reparametrizations from a min-area cost matrix.

    Parameters:
    A (np.ndarray): Cost matrix (e.g., from MakeMinArea_Amatrix)

    Returns:
    tuple of np.ndarray: (RePar1, RePar2, Alignment)
    """
    n, m = A.shape
    i, j = n - 1, m - 1
    count = n + m - 2  # Python: 0-indexed
    Alignment = np.zeros((4, n + m - 1))
    Alignment[:, count] = [i, j, 0, A[i, j]]

    while i > 0 and j > 0:
        if A[i - 1, j] < A[i, j - 1]:  # Type 1 move
            count -= 1
            i -= 1
            Alignment[:, count] = [i, j, 1, A[i, j]]
        else:  # Type 2 move
            count -= 1
            j -= 1
            Alignment[:, count] = [i, j, 2, A[i, j]]

    # Fill remaining entries (only horizontal or vertical moves)
    if i > 0:
        for k in range(i - 1, -1, -1):
            count -= 1
            Alignment[:, count] = [k, 0, 1, A[k, 0]]
    elif j > 0:
        for k in range(j - 1, -1, -1):
            count -= 1
            Alignment[:, count] = [0, k, 2, A[0, k]]

    Alignment[2, -1] = 1.5  # Mark the final point

    # Reparametrizations
    RePar1 = []
    RePar2 = []
    iold, jold = int(Alignment[0, 0]), int(Alignment[1, 0])
    RePar1.append(iold)
    RePar2.append(jold)

    for k in range(1, Alignment.shape[1]):
        i_curr = int(Alignment[0, k])
        j_curr = int(Alignment[1, k])
        di = i_curr - iold
        dj = j_curr - jold
        if di * dj > 0:
            nbr = max(abs(di), abs(dj))
            steps = np.arange(nbr)
            RePar1.extend(iold + steps * (di / nbr))
            RePar2.extend(jold + steps * (dj / nbr))
            iold = i_curr
            jold = j_curr

    # Final interpolation
    di = int(Alignment[0, -1]) - iold
    dj = int(Alignment[1, -1]) - jold
    nbr = max(abs(di), abs(dj))
    if nbr > 0:
        RePar1.extend(np.linspace(iold, iold + di, nbr + 1))
        RePar2.extend(np.linspace(jold, jold + dj, nbr + 1))

    return np.array(RePar1), np.array(RePar2), Alignment
