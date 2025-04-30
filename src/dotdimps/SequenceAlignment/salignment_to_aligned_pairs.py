import numpy as np

def salignment_to_aligned_pairs(Salign, options):
    """
    Extracts aligned residue pairs from sequence alignment.

    Parameters:
    Salign (list or tuple of 3 str): [seq1, alignment, seq2]
    options (dict): Dictionary with at least the key:
                    'TrimSeqenceAlignment': 0 or 1

    Returns:
    tuple:
        - Spairs (np.ndarray): 2 x N array of exactly aligned residue indices (marked '|')
        - Sother_pairs (np.ndarray): 2 x M array of weakly aligned indices (marked ':')
    """
    seq1, alignment_line, seq2 = Salign

    if options.get("TrimSeqenceAlignment", 0) == 1:
        match_flags = np.array([1.0 if c == '|' else 0.0 for c in alignment_line])
        smooth = np.convolve(match_flags, np.ones(3)/3, mode='same')
        indices = np.where(smooth == 1)[0]
        if len(indices) > 0:
            start = max(0, indices[0] - 1)
            stop = min(len(alignment_line), indices[-1] + 2)
            alignment_line = alignment_line[:start] + ' ' * (stop - start) + alignment_line[stop:]

    # Map from alignment index to sequence index
    idx1 = np.cumsum([c != '-' for c in seq1]) - 1
    idx2 = np.cumsum([c != '-' for c in seq2]) - 1

    idx1 = np.where([c != '-' for c in seq1])[0]
    index1_map = np.zeros(len(seq1), dtype=int)
    index1_map[idx1] = np.arange(1, len(idx1)+1)

    idx2 = np.where([c != '-' for c in seq2])[0]
    index2_map = np.zeros(len(seq2), dtype=int)
    index2_map[idx2] = np.arange(1, len(idx2)+1)

    ex1 = [i for i, c in enumerate(alignment_line) if c == '|']
    ex2 = [i for i, c in enumerate(alignment_line) if c == ':']

    Spairs = np.array([
        index1_map[:len(alignment_line)][ex1],
        index2_map[:len(alignment_line)][ex1]
    ])
    Sother_pairs = np.array([
        index1_map[:len(alignment_line)][ex2],
        index2_map[:len(alignment_line)][ex2]
    ])

    return Spairs, Sother_pairs
