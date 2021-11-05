import pickle
from probdrift.mvn_models import mv_lgbm, mv_gbm, mvn_ngboost, mvn_neuralnetwork, indep_ngboost
import pandas as pd
from pathlib import Path
from probdrift import X_VAR, Y_VAR
import yaml
import os
from sklearn import preprocessing
import sys
import numpy as np

name_to_model = {
    "LGBM": mv_lgbm,
    "GBM": mv_gbm,
    "mvn_ngboost": mvn_ngboost,
    "mvn_ngboost_og": mvn_ngboost,
    "nn": mvn_neuralnetwork,
    "indep_ngboost": indep_ngboost
}
debug = False
if debug:
    data_dir = "../tests/data/test_dat.csv"
    data = pd.read_csv(data_dir)
else:
    data_dir = os.environ["data_file"]
    data = pd.read_hdf(data_dir)
    data.dropna(inplace=True)
    data.reset_index(inplace=True)


def random_id_subset(ids, pc=0.1):
    unique_id = np.unique(ids)
    N_unique = len(unique_id)
    np.random.shuffle(unique_id)
    in_test = int(N_unique * pc)
    test_ids = unique_id[:in_test]
    test_mask = np.in1d(ids, test_ids)
    train_mask = np.invert(test_mask)
    return train_mask, test_mask


def run_and_save(params, X, y, ids, path_base, X_test):
    mod_name = params["modelname"]
    if mod_name=='nn':
        mod = name_to_model[mod_name](**params)
    else:
        mod = name_to_model[mod_name](params)
    train_mask, valid_mask = random_id_subset(ids, pc=0.1)

    output_dist_name = path_base + "scipy_dists.p"
    # Only run if the file does not already exist
    if not os.path.isfile(output_dist_name):
        mod.fit(X[train_mask, :], y[train_mask, :], X[valid_mask, :], y[valid_mask, :])

        if mod_name != 'nn':
            # Original gradient schemes tend to have large models (60MB)
            # We do not neccesarily need them
            if 'og' not in params['fname']:
                pickle.dump(mod, open(path_base + "model.p", "wb"))
        else:
            mod.model.save(path_base + "model.h5")
        preds = mod.scipy_distribution(X_test, cmat_output=True)
        pickle.dump(preds, open(path_base + "scipy_dists.p", "wb"))



if __name__=='__main__':

    num_to_run = int(sys.argv[-1])
    cfg_dir = "config_files_leaf/"
    allfiles = os.listdir(cfg_dir)
    allfiles.sort()
    config_file = allfiles[num_to_run]


    par_dict = yaml.load(
        open(cfg_dir + allfiles[num_to_run], "r"), Loader=yaml.FullLoader
    )
    fname = par_dict["fname"]
    shuffle_seed = 500
    if 'undrogue' in data_dir:
        path_base = f"results19may_{shuffle_seed}_undrg/" + fname
    else:
        path_base = f"results19may_{shuffle_seed}/" + fname

    res_path = Path(path_base).mkdir(parents=True, exist_ok=True)
    ids = data["id"].values
    X = data[X_VAR].values
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # Mutliply by 100 for numerical stability
    # Methods model the inverse covariance or the log scale
    Y = data[Y_VAR].values * 100
    del data

    N_runs = 20
    np.random.seed(shuffle_seed)
    splits = [random_id_subset(ids) for _ in range(N_runs)]

    for i, (train_mask, test_mask) in enumerate(splits):
        fold_path = path_base + "/fold_" + str(i)
        X_train = X[train_mask, :]
        X_test = X[test_mask, :]
        Y_train = Y[train_mask, :]
        train_ids = ids[train_mask]
        run_and_save(par_dict, X_train, Y_train, train_ids, fold_path, X_test)
