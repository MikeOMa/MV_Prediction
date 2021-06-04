import ngboost
from ngboost.distns import MultivariateNormal
from ngboost.scores import LogScore
from .model_types import mvn_predictor
import numpy as np
from sklearn.tree import DecisionTreeRegressor

DEFAULT_BASE = {
    "criterion": "friedman_mse",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "max_depth": 3,
    "splitter": "best",
}

DEFAULT_NGBOOST = {"minibatch_frac": 1.0, "learning_rate": 0.01}

MAX_ITER = 1000


class mvn_ngboost(mvn_predictor):
    def __init__(self, params={}):
        super().__init__(params)
        self.model = None
        self.base = None
        if not hasattr(self, "params_base"):
            self.params_base = DEFAULT_BASE
        if not hasattr(self, "params_ngboost"):
            self.params_ngboost = DEFAULT_NGBOOST

    def fit(self, X_train, Y_train, X_valid, Y_valid, base=None, retrain=False):
        p = Y_train.shape[1]
        dist = MultivariateNormal(p)

        if base is None:
            b2 = DecisionTreeRegressor(**self.params_base)
            self.base = b2
            print(self.params_base)
        else:
            b2 = base
            self.base = base
        n_estimators = MAX_ITER
        self.model = ngboost.NGBoost(
            Dist=dist,
            Score=LogScore,
            Base=b2,
            **self.params_ngboost,
            verbose_eval=10,
            n_estimators=n_estimators
        )

        self.model.fit(
            X=X_train,
            Y=Y_train,
            X_val=X_valid,
            Y_val=Y_valid,
            early_stopping_rounds=100,
        )
        self.model.best_val_loss_itr
        if retrain:
            self.model = ngboost.NGBoost(
                Dist=dist,
                Score=LogScore,
                Base=b2,
                **self.params_ngboost,
                verbose_eval=10,
                n_estimators=self.model.best_val_loss_itr + 1
            )

            X = np.vstack([X_train, X_valid])
            Y = np.vstack([Y_train, Y_valid])
            self.model.fit(X=X, Y=Y)

    def scipy_distribution(self, X_test, cmat_output=False):
        preds = self.model.pred_dist(X_test, max_iter=self.model.best_val_loss_itr)
        if cmat_output:
            out = [preds.mean(), preds.cov]
        else:
            out = preds.scipy_distribution()
        return out
