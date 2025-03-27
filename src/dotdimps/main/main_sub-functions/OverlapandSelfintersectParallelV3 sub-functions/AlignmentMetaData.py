import numpy as np

def AlignmentMetaData(RePar1, RePar2, IsAligned):
    """
    Computes metadata about sequence alignment between two chains.
    
    Parameters:
        RePar1 (np.ndarray): Reparametrization array for the first chain.
        RePar2 (np.ndarray): Reparametrization array for the second chain.
        IsAligned (np.ndarray): Binary array indicating aligned residues.
    
    Returns:
        list: [number of aligned residues, aligned window length (chain 1), 
               aligned window length (chain 2), total residues in chain 1].
    """
    return [
        np.sum(IsAligned),
        RePar1[-1] - RePar1[0] + 1,
        RePar2[-1] - RePar2[0] + 1,
        len(RePar1)
    ]
