def nbr_residues_in_string(s):
    """
    Counts the number of non-gap residues in a string.
    Gaps are typically represented by '-' in sequence alignments.

    Parameters:
    s (str): Alignment substring

    Returns:
    int: Number of residues (non-gap characters)
    """
    return sum(1 for c in s if c != '-')
