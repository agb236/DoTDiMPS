import numpy as np

def PriceEstEndContraction(X):
    """
    Estimates the cost of end contraction based on an input value X.
    
    Parameters:
        X (float or np.ndarray): Input value(s) for cost estimation.
    
    Returns:
        float or np.ndarray: Computed cost estimate based on given input.
    """
    return 0.623529412 * np.minimum(X, 17)**2 + 3.09852941 * np.minimum(X, 17) + np.maximum(X - 17, 0) * 25
