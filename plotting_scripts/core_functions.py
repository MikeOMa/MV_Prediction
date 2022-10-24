from probdrift import X_VAR, Y_VAR, mpl_config
from sklearn import preprocessing
import os
import pandas as pd
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import sys

import sys

sys.path.append("../experiments/")
os.environ["data_file"] = os.environ["drift_files"]
from dispatcher import random_id_subset
import pickle


class oof_predict:
    def __init__(self, folds_dict, models):
        self.folds_dict = folds_dict
        self.models = models

    def predict(self, X, drift_id, drogue=True):
        models = self.models[drogue]
        ids_to_fold = self.folds_dict[drogue]
        try:
            current_model = models[ids_to_fold[drift_id]]
        except KeyError:
            print("Defaulting to last fold")
            current_model = models[-1]
        return current_model.model.pred_dist(X)


def read_models(model_dir, k=5):
    models = [
        pickle.load(open(model_dir + "fold_" + str(i) + "model.p", "rb"))
        for i in range(k)
    ]
    return models


data = pd.read_hdf(os.environ["drift_files"])
ids = data["id"].values
## 500 is the seed used to generate the splits
np.random.seed(500)
folds = [random_id_subset(ids, 0.1) for i in range(10)]

multivariate_models_dir = "../experiments/results19may_500/045mvn_ngboost/"
indep_models_dir = "../experiments/results19may_500/029indep_ngboost/"
models_list = read_models(multivariate_models_dir)
indep_models_list = read_models(indep_models_dir)

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(data[X_VAR])
