import ngboost
from ngboost.scores import LogScore
from scipy.stats import multivariate_normal
from .model_types import mvn_predictor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from probdrift.mvn_models.mvn_ngboost import DEFAULT_NGBOOST, DEFAULT_BASE


class indep_ngboost(mvn_predictor):
    def __init__(self, params={}):
        super().__init__(params)
        self.model = None
        self.base = None
        if not hasattr(self, "params_base"):
            self.params_base = DEFAULT_BASE
        if not hasattr(self, "params_ngboost"):
            self.params_ngboost = DEFAULT_NGBOOST
        self.models = None

    def fit(
        self,
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        early_stopping_rounds=50,
        base=None,
        retrain=False,
    ):

        if base is None:
            b2 = DecisionTreeRegressor(**self.params_base)
            self.base = b2
        else:
            b2 = base
            self.base = base

        self.models = []
        for i in range(Y_train.shape[1]):
            self.models.append(
                ngboost.NGBoost(
                    Base=b2, **self.params_ngboost, verbose_eval=100, n_estimators=2000
                )
            )

            self.models[i].fit(
                X=X_train,
                Y=Y_train[:, i],
                X_val=X_valid,
                Y_val=Y_valid[:, i],
                early_stopping_rounds=early_stopping_rounds,
            )
            if retrain:
                self.models[i] = ngboost.NGBoost(
                    Score=LogScore,
                    Base=b2,
                    **self.params_ngboost,
                    verbose_eval=100,
                    n_estimators=self.models[i].best_val_loss_itr
                )
                X = np.vstack([X_train, X_valid])
                Y = np.vstack([Y_train, Y_valid])
                self.models[i].fit(X=X, Y=Y[:, i])

    def scipy_distribution(self, X_test, cmat_output=False):
        pred_dists = [
            model.pred_dist(X_test, max_iter=model.best_val_loss_itr)
            for model in self.models
        ]
        print(pred_dists[0].loc.reshape(-1, 1).shape)
        means = np.concatenate([dist.loc.reshape(-1, 1) for dist in pred_dists], axis=1)
        vars = np.concatenate([dist.var.reshape(-1, 1) for dist in pred_dists], axis=1)
        cmat = [np.diag(vars[i, :]) for i in range(vars.shape[0])]
        if cmat_output:
            out = [means, np.stack(cmat)]
        else:
            out = [
                multivariate_normal(means[i, :], cov=cmat[i])
                for i in range(vars.shape[0])
            ]
        return out
