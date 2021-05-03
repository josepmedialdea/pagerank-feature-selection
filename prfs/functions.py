import numpy as np


def correlation(x1, x2):
    covariance = np.cov(x1, x2)[0, 1]

    x1_variance = np.var(x1, ddof=1)
    x2_variance = np.var(x2, ddof=1)

    return np.abs(covariance/(np.sqrt(x1_variance*x2_variance)))

def uncorrelation(x1, x2):
    return 1 - correlation(x1, x2)
