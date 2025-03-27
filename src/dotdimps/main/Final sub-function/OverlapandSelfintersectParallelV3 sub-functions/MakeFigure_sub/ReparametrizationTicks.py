import numpy as np

def MakeReParTicks(RePar, n):
    """
    Generates reparametrization tick labels and their corresponding indices.
    
    Parameters:
        RePar (np.ndarray): Reparametrization array.
        n (int): Number of tick labels to generate.
    
    Returns:
        tuple:
            - np.ndarray: Tick labels.
            - np.ndarray: Corresponding indices in the RePar array.
    """
    min_val, max_val = RePar[0], RePar[-1]
    inner_labels = np.unique(np.ceil(np.linspace(min_val, max_val, n) / 10) * 10)
    inner_labels[0], inner_labels[-1] = min_val, max_val
    
    inner_indices = np.array([np.argmax(RePar >= label) + 1 for label in inner_labels])  # Adjust for 1-indexing from MATLAB
    
    return inner_labels, inner_indices
