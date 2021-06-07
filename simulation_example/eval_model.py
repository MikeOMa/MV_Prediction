import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from fittingmethods import fit_indep_ngboost, fit_ngboost, fit_nn, fit_skGB
from generate_data import simulate_data
from probdrift.metrics import METRIC_LIST

N_rep = 50
# randomseed
SEED = 230521
N = int(sys.argv[1])
np.random.seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)

#Controls if we use the original simulation from williams 1996.
vanilla = "vanilla" in sys.argv[1:]

def scipy_to_tfp(scipy_dist_list):
    return [
        tfp.distributions.MultivariateNormalTriL(
            dist.mean, tf.linalg.cholesky(dist.cov)
        )
        for dist in scipy_dist_list
    ]


def mean_KL(d1, d2):
    return np.mean(
        [tfp.distributions.kl_divergence(dist1, dist2) for dist1, dist2 in zip(d1, d2)]
    )

NN_LAYERS = [[20], [50], [100], [20, 20], [50, 20], [100, 20]]


def gen_nn(k):
    name = "NN"
    hidden_layers = NN_LAYERS[k]
    for i in hidden_layers:
        name = name + str(i) + "_"
    fit_kwargs = {"hidden_layers": hidden_layers}
    return name, fit_kwargs


res_dict = {"name": [], "logllh": [], "KL_div": [], "time": []}
for key in METRIC_LIST.keys():
    res_dict[key] = []

def fill_dict(name, pred_dist, t):
    ## Using true_tfp global and Y_test
    print(name)
    res_dict["name"].append(name)
    res_dict["logllh"].append(
        np.mean([pred_dist[i].logpdf(Y_test[i, :]) for i in range(len(pred_dist))])
    )
    res_dict["KL_div"].append(mean_KL(scipy_to_tfp(pred_dist), true_tfp))
    for key, metric_fn in METRIC_LIST.items():
        res_dict[key].append(np.mean(metric_fn(pred_dist, Y_test)))

    # res_dict['par_dist'].append()
    res_dict["time"].append(t)


for i in range(N_rep):
    X, Y, _ = simulate_data(N, vanilla=vanilla)
    X_valid, Y_valid, _ = simulate_data(300, vanilla=vanilla)
    X_test, Y_test, true_dists = simulate_data(1000, vanilla=vanilla)
    true_tfp = scipy_to_tfp(true_dists)
    name = 'skGB'
    model, t = fit_skGB(X, Y, X_valid, Y_valid)
    out = model.scipy_distribution(X_test)
    fill_dict(name, out, t)

    for natural in [True, False]:
        if natural:
            name = "NGB"
        else:
            name = "GB"
        name = name
        model, t = fit_ngboost(
            X, Y, X_valid, Y_valid, natural_gradient=natural
        )
        pred = model.pred_dist(X_test)
        out = pred.scipy_distribution()
        fill_dict(name, out, t)

    for natural in [True, False]:
        if natural:
            name = "indep_NGB"
        else:
            name = "indep_GB"
        model, t = fit_indep_ngboost(
            X, Y, X_valid, Y_valid, natural_gradient=natural
        )
        out = model.scipy_distribution(X_test)
        fill_dict(name, out, t)

    for i in range(6):
        name, fit_kwargs = gen_nn(i)
        model, t = fit_nn(
            X, Y, X_valid, Y_valid, **fit_kwargs, cp_name=str(N) + str(i)
        )
        out = model.scipy_distribution(X_test)
        fill_dict(name, out, t)


if vanilla:
    out_path = f"results_{N_rep}_{SEED}_V/"
else:
    out_path = f"results_{N_rep}_{SEED}/"

if not os.path.exists(out_path):
    os.makedirs(out_path)
pd.DataFrame(res_dict).to_csv(out_path + f"res_{N}.csv", index=False)
