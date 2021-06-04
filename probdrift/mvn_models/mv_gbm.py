from .model_types import mv_predictor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import os
from .mvn_ngboost import DEFAULT_BASE, MAX_ITER

DEFAULT_GBM = DEFAULT_BASE.copy()
del DEFAULT_GBM["splitter"]


def predict_GBR(model, X, stage):
    preds = (
        np.sum([est[0].predict(X) for est in model.estimators_[:stage]], axis=0)
        * model.learning_rate
    )
    preds = preds + model._raw_predict_init(X).flatten()
    return preds


class sklearn_earlystop:
    """
    Class for earlystopping in sklearn
    set model.valid_data = [train, test] prior to training
    """

    def __init__(self, early_stopping_rounds=50, X_valid=None, Y_valid=None):
        self.count = 0
        self.min_mse = np.inf
        self.mse_track = []
        self.patience = early_stopping_rounds
        self.best_iter = 0
        self.X_valid = X_valid
        self.Y_valid = Y_valid

    def __call__(self, i, cmodel, local_pars):
        """
        Format of monitor
        Gets called every iteration
        """
        self.count += 1
        if i > 0:
            preds = predict_GBR(cmodel, self.X_valid, i)
            mse = np.mean(np.square(self.Y_valid - preds))
            self.mse_track.append(mse)

            # Check if mse is getting better
            if self.min_mse > mse:
                self.min_mse = mse
                self.best_iter = i
                self.count = 0

        out = self.count > self.patience
        if out:
            print("Early stopping, best iteration is " + str(self.best_iter))
        return out


class mv_gbm(mv_predictor):
    def fit(self, X_train, Y_train, X_valid, Y_valid, early_stopping_rounds=50, retrain=False):
        p = Y_train.shape[1]
        params = getattr(self, "params", DEFAULT_GBM)
        params["n_estimators"] = 1000
        print(params)
        self.models = [GradientBoostingRegressor(**params) for i in range(p)]

        for dim in range(p):
            es_monitor = sklearn_earlystop(
                early_stopping_rounds, X_valid, Y_valid[:, dim]
            )
            self.models[dim].fit(X_train, Y_train[:, dim], monitor=es_monitor)
            # Drop the estimators up to the best iteration
            # Forces the .predict method to just use the right number of iterations
            if retrain:
                X_full = np.vstack([X_train, X_valid])
                Y_full = np.vstack([Y_train, Y_valid])
                new_params = params.copy()
                new_params['n_estimators']=es_monitor.best_iter
                self.models[dim] = GradientBoostingRegressor(**new_params)
                self.models[dim].fit(X_full, Y_full[:,dim])
            else:
                ## Cut off all the extra trees.
                self.models[dim].estimators_ = self.models[dim].estimators_[
                    : es_monitor.best_iter
                ]
        self._estimate_err(X_train, Y_train)
        # Estimate the error
