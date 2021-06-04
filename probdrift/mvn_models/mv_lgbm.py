from .model_types import mv_predictor
import lightgbm
import numpy as np
import pandas as pd
import os

DEFAULT_LGBM = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "n_estimators": 4000,
    "verbosity": 1,
    "num_leaves": 31,
}


class mv_lgbm(mv_predictor):
    def fit(self, X_train, Y_train, X_valid, Y_valid, retrain=False):
        params = getattr(self, "params", DEFAULT_LGBM)
        num_threads = os.environ.get("num_threads", 2)
        print(num_threads)
        params["n_jobs"] = num_threads
        self.models = []
        p = Y_train.shape[1]
        for i in range(p):
            tr_data = lightgbm.Dataset(
                X_train, label=Y_train[:, i], free_raw_data=False
            )
            valid_data = lightgbm.Dataset(
                X_valid, label=Y_valid[:, i], free_raw_data=False
            )
            mod = lightgbm.train(
                params,
                train_set=tr_data,
                valid_sets=[valid_data],
                early_stopping_rounds=100,
            )
            if retrain:
                del tr_data, valid_data
                best_iter = mod.best_iteration
                print("fitting full")
                y=np.vstack([Y_train, Y_valid])[:,i]
                tr_data = lightgbm.Dataset(
                    np.vstack([X_train, X_valid]),
                    label=y,
                    free_raw_data=False,
                )
                new_params = params.copy()
                new_params['n_estimators']=best_iter
                mod = lightgbm.train(
                    new_params, num_boost_round=best_iter, train_set=tr_data
                )
            self.models.append(mod)
        self._estimate_err(X_train, Y_train)
        # Estimate the error
