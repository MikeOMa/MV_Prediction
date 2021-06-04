"""
NLL,
RMSE,
UV-RMSE
95\% CI
CI area
"""

import numpy as np
from scipy.stats import chi2
from probdrift.MVN_helpers import in_ci
from numba import jit
from scipy.special import gamma

# All functions take a scipy distribution for each point and a array of data with that many rows
# Return in the desired metric


def NLL(dists, y):
    N = y.shape[0]
    nlls = [-dists[i].logpdf(y[i, :]) for i in range(N)]
    return nlls


@jit(fastmath=True)
def _es_p1_est(X):
    m = X.shape[0]
    psum = 0
    counter = 0
    for i in range(m):
        for j in range(i):
            psum += ((X[i, 0] - X[j, 0]) ** 2 + (X[i, 1] - X[j, 1]) ** 2) ** 0.5
            counter += 1
    ran = m * (m - 1) / 2
    assert ran == counter
    return psum / ran


@jit(fastmath=True)
def _es_p2_est(X, y):
    psum = 0
    m = X.shape[0]
    for i in range(m):
        k = (X[i, 0] - y[0]) ** 2 + (X[i, 1] - y[1]) ** 2
        psum += k ** 0.5
    return psum / m


@jit
def ES_sample(X, y):
    # E||X-y||
    p2 = _es_p2_est(X, y)
    # E||X-\tilde(X)||
    p1 = _es_p1_est(X)

    return p2 - 0.5 * p1


def ES(dists, y, nsamp=500):
    n = len(dists)
    es = np.zeros(n)
    for i in range(n):
        X = dists[i].rvs(nsamp)
        es[i] = ES_sample(X, y[i, :])
    return es


def MSE_x(dists, y):
    N = y.shape[0]
    se = [np.square(dists[i].mean[0] - y[i, 0]) for i in range(N)]
    se = np.array(se)
    return se


def MSE_y(dists, y):
    N = y.shape[0]
    se = [np.sum(np.square(dists[i].mean[1] - y[i, 1])) for i in range(N)]
    se = np.array(se)
    return se


def coverage(dists, y, alpha=0.90):
    N = y.shape[0]
    covs = [dists[i].cov for i in range(N)]
    diffs = [y[i, :] - dists[i].mean for i in range(N)]
    ret = [in_ci(diff, cov, alpha=alpha) for diff, cov in zip(diffs, covs)]
    return ret


def matrix_area(mat, mult):
    p = mat.shape[0]
    return (
        2
        * (np.pi ** (p / 2))
        / gamma(p / 2)
        / p
        * (np.linalg.det(mat) ** 0.5)
        * mult ** (p / 2)
    )


def area(dists, y, alpha: float = 0.90):
    N = y.shape[0]
    covs = [dists[i].cov for i in range(N)]
    # TODO: Check this is correct for the area
    area = [matrix_area(cov, chi2.ppf(alpha, cov.shape[1])) for cov in covs]
    return area


METRIC_LIST = {
    "NLL": NLL,
    "MSE_x": MSE_x,
    "MSE_y": MSE_y,
    "ES": ES,
    "coverage": coverage,
    "area": area,
}
