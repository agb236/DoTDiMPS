import numpy as np
from tm_alignment_to_blocks import tm_alignment_to_blocks
from nbr_residues_in_string import nbr_residues_in_string 

def alignment_based_reparametrization(line1, line2, line3, align_ends=True):
    """
    Reparametrizes two curves (line1 and line3) based on the alignment given in line2.
    """
    M = tm_alignment_to_blocks(line2)
    if M.shape[1] < 3:
        print("Error in alignment_based_reparametrization: Alignment Failed")
        return [], [], []

    blockstart = 0
    blockend = M.shape[0] - 1
    index1 = 0
    index3 = 0

    if align_ends:
        if M[0, 2] == 0:
            blockstart = 1
            index1 = nbr_residues_in_string(line1[M[0, 0]:M[0, 1]+1])
            index3 = nbr_residues_in_string(line3[M[0, 0]:M[0, 1]+1])
        if M[-1, 2] == 0:
            blockend -= 1

    alignment_length = M[blockend, 1] - M[blockstart, 0]
    RePar1 = np.zeros(alignment_length)
    RePar3 = np.zeros(alignment_length)
    IsAligned = np.zeros(alignment_length)

    indexA = 0
    for i in range(blockstart, blockend + 1):
        start_idx, end_idx = M[i, 0], M[i, 1]
        nr1 = nbr_residues_in_string(line1[start_idx:end_idx+1])
        nr3 = nbr_residues_in_string(line3[start_idx:end_idx+1])
        align_len = max(nr1, nr3)

        RePar1[indexA:indexA+align_len] = np.linspace(index1, index1 + nr1, align_len + 2)[1:-1]
        index1 += nr1

        RePar3[indexA:indexA+align_len] = np.linspace(index3, index3 + nr3, align_len + 2)[1:-1]
        index3 += nr3

        IsAligned[indexA:indexA+align_len] = M[i, 2]
        indexA += align_len

    return RePar1[:indexA], RePar3[:indexA], IsAligned[:indexA]
