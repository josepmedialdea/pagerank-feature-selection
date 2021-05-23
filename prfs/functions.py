import numpy as np


def correlation(x1, x2):
    covariance = np.cov(x1, x2)[0, 1]

    x1_variance = np.var(x1, ddof=1)
    x2_variance = np.var(x2, ddof=1)

    return np.abs(covariance/(np.sqrt(x1_variance*x2_variance)))

def uncorrelation(x1, x2):
    return 1 - correlation(x1, x2)

def sparse_correlation(x1, x2):
    x1_x2_correlation = correlation(x1, x2)

    if x1_x2_correlation >= 0.5:
        return x1_x2_correlation
    else:
        return 0

def sparse_uncorrelation(x1, x2):
    x1_x2_uncorrelation = uncorrelation(x1, x2)
    
    if x1_x2_uncorrelation >= 0.5:
        return x1_x2_uncorrelation
    else:
        return 0
