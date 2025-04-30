import numpy as np
from make_type1_and_2_objective_function_tmav_score import make_type1_and_2_objective_function_tmav_score
from make_min_area_amatrix import make_min_area_amatrix
from extract_alignment_from_min_area_matrix import extract_alignment_from_min_area_matrix

def fill_alignment_gaps(P1, P2, Spairs, optionsFig):
    """
    Fills alignment gaps between aligned residue pairs using interpolation or optimal triangulation.

    Parameters:
    P1, P2 (np.ndarray): Point clouds (n x 3) and (m x 3)
    Spairs (np.ndarray): 2 x N array of aligned residue indices
    optionsFig (dict): Options (e.g., from set_options())

    Returns:
    tuple: RePar1, RePar2 (np.ndarray) — reparametrized coordinate paths
    """
    DS = Spairs[:, 1:] - Spairs[:, :-1]
    DS2 = ((DS == 1).astype(int))
    DS2 = np.hstack(([0, 0], DS2.flatten())) + np.hstack((DS2.flatten(), [0, 0]))
    SpairsT = Spairs[:, np.any(DS == 1, axis=0)]

    if np.sum(SpairsT[0, 0] == Spairs[0, :]) < 2:
        SpairsT = np.hstack((Spairs[:, [0]], SpairsT))
    if np.sum(SpairsT[0, -1] == Spairs[0, :]) < 2:
        SpairsT = np.hstack((SpairsT, Spairs[:, [-1]]))

    I1 = [Spairs[0, 0], Spairs[0, -1]]
    I2 = [Spairs[1, 0], Spairs[1, -1]]

    # Terminal extension
    if optionsFig["MaximalSequenceAlignmentExtension"] > 0:
        chain1startpre = max(0, Spairs[0, 0] - optionsFig["MaximalSequenceAlignmentExtension"])
        chain2startpre = max(0, Spairs[1, 0] - optionsFig["MaximalSequenceAlignmentExtension"])

        if chain1startpre < Spairs[0, 0] and chain2startpre < Spairs[1, 0]:
            newNterminal = True
            chain1start = chain1startpre
            chain2start = chain2startpre
        else:
            newNterminal = False
            chain1start = Spairs[0, 0]
            chain2start = Spairs[1, 0]

        chain1endpre = min(P1.shape[0], Spairs[0, -1] + optionsFig["MaximalSequenceAlignmentExtension"])
        chain2endpre = min(P2.shape[0], Spairs[1, -1] + optionsFig["MaximalSequenceAlignmentExtension"])

        if chain1endpre > Spairs[0, -1] and chain2endpre > Spairs[1, -1]:
            newCterminal = True
            chain1end = chain1endpre
            chain2end = chain2endpre
        else:
            newCterminal = False
            chain1end = Spairs[0, -1]
            chain2end = Spairs[1, -1]

        SpairsT[0, :] -= chain1start
        SpairsT[1, :] -= chain2start

        P1 = P1[chain1start:chain1end, :]
        P2 = P2[chain2start:chain2end, :]

        if newNterminal:
            SpairsT = np.hstack((np.array([[0], [0]]), SpairsT))
        if newCterminal:
            SpairsT = np.hstack((SpairsT, np.array([[P1.shape[0] - 1], [P2.shape[0] - 1]])))

    else:
        chain1start = I1[0]
        chain2start = I2[0]
        P1 = P1[I1[0]:I1[1], :]
        P2 = P2[I2[0]:I2[1], :]
        SpairsT[0, :] -= I1[0]
        SpairsT[1, :] -= I2[0]

    # Now fill gaps between anchors in SpairsT
    RePar1 = []
    RePar2 = []
    for i in range(1, SpairsT.shape[1]):
        d1 = SpairsT[0, i] - SpairsT[0, i - 1]
        d2 = SpairsT[1, i] - SpairsT[1, i - 1]

        if d1 <= 1 and d2 <= 1:
            RePar1.append(SpairsT[0, i])
            RePar2.append(SpairsT[1, i])
        elif d1 > 1 and d2 <= 1:
            RePar1.extend(range(SpairsT[0, i - 1], SpairsT[0, i] + 1))
            RePar2.extend(np.linspace(SpairsT[1, i - 1], SpairsT[1, i], d1 + 1))
        elif d1 <= 1 and d2 > 1:
            RePar1.extend(np.linspace(SpairsT[0, i - 1], SpairsT[0, i], d2 + 1))
            RePar2.extend(range(SpairsT[1, i - 1], SpairsT[1, i] + 1))
        else:
            # Gap in both chains: triangulation
            P1gap = P1[SpairsT[0, i - 1]:SpairsT[0, i] + 1]
            P2gap = P2[SpairsT[1, i - 1]:SpairsT[1, i] + 1]
            Type1, Type2 = make_type1_and_2_objective_function_tmav_score(P1gap, P2gap, P1.shape[0], P2.shape[0])
            A = make_min_area_amatrix(Type1, Type2)
            Re1, Re2, _ = extract_alignment_from_min_area_matrix(A)
            RePar1.extend(Re1[1:] + SpairsT[0, i - 1])
            RePar2.extend(Re2[1:] + SpairsT[1, i - 1])

    RePar1 = np.array(RePar1) + chain1start
    RePar2 = np.array(RePar2) + chain2start

    return RePar1, RePar2
