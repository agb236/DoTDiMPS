import numpy as np
from make_dp_nm import make_dp_nm

def make_type1_and_2_objective_function_tmav_score(P1, P2, n, m):
    """
    Computes cost matrices (Type1 and Type2) for triangulated alignment scoring
    using TM-score-inspired objectives.

    Parameters:
    P1, P2 (np.ndarray): Structures (n x 3) and (m x 3)
    n, m (int): Lengths of the respective chains

    Returns:
    tuple of np.ndarray: Type1 and Type2 cost matrices
    """
    DP = make_dp_nm(P1, P2)  # Shape: (n, m, 3)
    Dsqr = np.sum(DP ** 2, axis=2)  # Shape: (n, m)
    Lav = (n + m) / 2

    if Lav <= 21:
        Lsqr = 0.5
    else:
        Lsqr = (1.24 * ((Lav - 15) ** (1/3) - 1.8)) ** 2

    TMObj = -1 / (1 + Dsqr / Lsqr)

    Type1 = TMObj[:-1, :] + TMObj[1:, :]     # Sum along P1 direction
    Type2 = TMObj[:, :-1] + TMObj[:, 1:]     # Sum along P2 direction

    return Type1, Type2
