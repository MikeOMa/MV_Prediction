from scipy.stats import chi2
from math import pi
import numpy as np


def elipse_points(sigma, alpha: float = 0.95) -> np.ndarray:
    """

    Parameters
    ----------
    alpha: the confidence level for the elipse.
    L: The lower triangular of the covariance matrix.

    Returns
    -------
    A set of points Nx2 array which define the boundary of the circle.

    """
    crit_val = chi2.ppf(alpha, sigma.shape[0])
    L = np.linalg.cholesky(sigma)
    mult = np.sqrt(crit_val)
    grid = np.linspace(-pi, pi, 500)
    circle = np.vstack([np.sin(grid), np.cos(grid)])
    elipse = mult * L @ circle
    return elipse.T


def in_ci(error, sigma, alpha=0.95):
    crit_val = chi2.ppf(alpha, sigma.shape[0])
    sigma_inv = np.linalg.inv(sigma)
    n = error.T @ sigma_inv @ error
    return n < crit_val


def cov_to_sigma(cov=None, L=None):
    assert not (cov is None and L is None), ValueError("Must specify one of L or cov")
    if L is not None:
        cov = np.linalg.inv(L @ L.T)
    sigma_1 = np.sqrt(cov[0, 0])
    sigma_2 = np.sqrt(cov[1, 1])
    rho = cov[0, 1] / sigma_1 / sigma_2
    return np.array([sigma_1, sigma_2, rho])
