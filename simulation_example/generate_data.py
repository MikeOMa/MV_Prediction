"""
Example taken from Using Neural Networks to Model Conditional Multivariate Densities
Peter Martin Williams 1996
Replication of Figure 3.
"""

import matplotlib.pyplot as plt
import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal
from scipy.stats import multivariate_normal


def simulate_data(N=1000, sort=False, k=[1, 1, 0.25, 0.25], vanilla = False):
    x = np.random.rand(N) * np.pi
    if sort:
        x = np.sort(x)
    means = np.zeros((N, 2))
    means[:, 0] = k[0] * np.sin(2.5 * x) * np.sin(1.5 * x)
    means[:, 1] = k[1] * np.cos(3.5 * x) * np.cos(0.5 * x)

    if not vanilla:
        means[:, 0] = means[:, 0] + x
        means[:, 1] = means[:, 1] - x**2
    cov = np.zeros((N, 2, 2))
    cov[:, 0, 0] = 0.01 + k[2] * (1 - np.sin(2.5 * x)) ** 2
    cov[:, 1, 1] = 0.01 + k[3] * (1 - np.cos(3.5 * x)) ** 2
    corr = np.sin(2.5 * x) * np.cos(0.5 * x)
    off_diag = corr * np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
    cov[:, 0, 1] = off_diag
    cov[:, 1, 0] = off_diag
    scipy_dists = [
        multivariate_normal(mean=means[i, :], cov=cov[i, :, :]) for i in range(N)
    ]
    rvs = np.array([dist.rvs(1) for dist in scipy_dists])
    x = x.reshape(-1, 1)
    return x, rvs, scipy_dists


def cov_to_sigma(cov_mat):
    """
    Parameters:
        cov_mat: Nx2x2 numpy array
    Returns:
        sigma: (N,2) numpy array containing the variances
        corr: (N,) numpy array the correlation [-1,1] extracted from cov_mat
    """

    sigma = np.sqrt(np.diagonal(cov_mat, axis1=1, axis2=2))
    corr = cov_mat[:, 0, 1] / (sigma[:, 0] * sigma[:, 1])
    return sigma, corr


def plot_res(X, pred_dist, true_dist, **kwargs):
    order = np.argsort(X.flatten())
    x_plot = X.copy().flatten()
    x_plot.sort()
    # Extract parameters for plotting
    mean = np.array([pred_dist[i].mean for i in order])
    covs = np.array([pred_dist[i].cov for i in order])
    sigma, corrs = cov_to_sigma(covs)

    true_mean = np.array([true_dist[i].mean for i in order])
    true_cov_mat = np.array([true_dist[i].cov for i in order])
    true_sigma, true_corrs = cov_to_sigma(true_cov_mat)

    # Plot the parameters in the sigma, correlation representation
    fig, axs = plt.subplots(5, 1, sharex=True, **kwargs)
    colors = ["blue", "red"]
    axs[4].set_xlabel("X")
    for i in range(2):
        axs[i].set_title("Mean Dimension:" + str(i))
        axs[i].plot(x_plot, mean[:, i], label="fitted")
        axs[i].plot(x_plot, true_mean[:, i], label="true")

        axs[2 + i].set_title("Marginal Standard Deviation Dimension: " + str(i))
        axs[2 + i].plot(x_plot, sigma[:, i], label="fitted")
        axs[2 + i].plot(x_plot, true_sigma[:, i], label="true")
    axs[4].set_title("Correlation")
    axs[4].plot(x_plot, corrs, label="fitted")
    axs[4].plot(x_plot, true_corrs, label="true")
    for i in range(5):
        axs[i].legend()
    fig.tight_layout()
    fig.show()


def plot_data(X, Y):
    data_figure, data_axs = plt.subplots()
    data_axs.plot(X, Y[:, 0], "o", label="Dim 1")
    data_axs.plot(X, Y[:, 1], "o", label="Dim 2")
    data_axs.set_xlabel("X")
    data_axs.set_ylabel("Y")
    data_axs.set_title("Input Data")
    data_axs.legend()
    data_figure.show()


if __name__ == "__main__":
    X, Y, true_dist = simulate_data(sort=False)
    X = X.reshape(-1, 1)
    dist = MultivariateNormal(2)

    X_val, Y_val, _ = simulate_data(1000, sort=True)
    X_val = X_val.reshape(-1, 1)
    ngb = NGBRegressor(
        Dist=dist, verbose=True, n_estimators=2000, natural_gradient=True
    )
    ngb.fit(X, Y, X_val=X_val, Y_val=Y_val, early_stopping_rounds=5)
    y_dist = ngb.pred_dist(X, max_iter=ngb.best_val_loss_itr)
    plot_res(X, y_dist.scipy_distribution(), true_dist)
