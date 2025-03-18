import numpy as np

# Symmetric Log Transformation
def symmetric_log(x, C):
    return np.sign(x) * np.log1p(np.abs(x) + C)

# Inverse of the symmetric log
def inverse_symmetric_log(y, C):
    return np.sign(y)*(np.exp(np.sign(y)*y) - C - 1)
