import numpy as np

def tm_alignment_to_blocks(line2):
    """
    Converts the second line of a TM-align alignment (consisting of ':' and '.' for aligned,
    and ' ' for unaligned) into a list of aligned/non-aligned blocks with start, end, and label.

    Parameters:
    line2 (str): Second alignment line from TM-align output

    Returns:
    np.ndarray: Nx3 array where each row is [start_index, end_index, aligned_flag]
                aligned_flag: 1 for aligned ('.' or ':'), 0 for not aligned (' ')
    """
    aligned = np.array([(c in [':', '.']) for c in line2], dtype=int)
    n = len(aligned)

    if n == 0:
        return np.empty((0, 3), dtype=int)
    if n == 1:
        return np.array([[0, 0, aligned[0]]])

    blocks = []
    istart = 0
    old = aligned[0]

    for i in range(1, n):
        if aligned[i] != old:
            blocks.append([istart, i - 1, old])
            istart = i
            old = aligned[i]

    blocks.append([istart, n - 1, old])
    return np.array(blocks)
