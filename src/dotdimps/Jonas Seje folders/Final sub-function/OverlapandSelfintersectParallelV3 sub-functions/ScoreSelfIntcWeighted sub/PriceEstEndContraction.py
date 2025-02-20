import numpy as np

def PriceEstEndContraction(X):
    ud = 0.623529412 * np.minimum(X, 17)**2 + 3.09852941 * np.minimum(X, 17) + np.maximum(X - 17, 0) * 25
    return ud

#PriceEstEndContraction(83.6272)