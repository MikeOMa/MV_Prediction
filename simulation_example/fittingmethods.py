import time

from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal

from probdrift.mvn_models import indep_ngboost, mv_gbm, mvn_neuralnetwork


def fit_ngboost(X, Y, valid_X, valid_Y, **kwargs):
    """
    Wrapper to fit bivariate ngboost
    Args:
        X: Training X data
        Y: Training Y data
        valid_X: Validation Y dataset
        valid_Y: Validation Y dataset
        **kwargs: Passed to NGBRegressor

    Returns:
        a fitted ngboost model
    """
    dist = MultivariateNormal(2)

    ngb = NGBRegressor(
        Dist=dist, Score=dist.scores[0], verbose=True, n_estimators=1000, **kwargs
    )
    start = time.time()
    ngb.fit(X, Y, X_val=valid_X, Y_val=valid_Y, early_stopping_rounds=50)
    end = time.time()

    return ngb,  end - start


def fit_indep_ngboost(X, Y, valid_X, valid_Y, **kwargs):
    """
    Wrapper to fit ngboost with a diagonal covariance
    Args:
        X: Training X data
        Y: Training Y data
        valid_X: Validation Y dataset
        valid_Y: Validation Y dataset
        **kwargs: Passed to NGBRegressor

    Returns:
        a fitted ngboost model
    """
    model = indep_ngboost(params={"params_ngboost": kwargs})
    start = time.time()
    model.fit(X, Y, valid_X, valid_Y, early_stopping_rounds=50)
    end = time.time()
    return model, end - start


def fit_skGB(X, Y, valid_X, valid_Y):
    """
    Wrapper to fit ngboost with a diagonal covariance
    Args:
        X: Training X data
        Y: Training Y data
        valid_X: Validation Y dataset
        valid_Y: Validation Y dataset
    Returns:
        a fitted sklearn model
    """
    model = mv_gbm()
    start = time.time()
    model.fit(X, Y, valid_X, valid_Y, early_stopping_rounds=50)
    end = time.time()
    return model, end - start


def fit_nn(X, Y, valid_X, valid_Y, verbose=0, cp_name="", **kwargs):
    """
    Wrapper to fit neural network predicting a multivariate Gaussian
    Args:
        X: Training X data
        Y: Training Y data
        valid_X: Validation Y dataset
        valid_Y: Validation Y dataset
        ud: Uses the UD decomposition
        cp_name: Used for checkpoint file name. Needs to be unique from any other model fit running in parallel.
        **kwargs: Extra kwargs to mvn_neuralnetwork function, e.g. hidden_layers
    Returns:
        a fitted neural network model
    """
    model = mvn_neuralnetwork(**kwargs)

    start = time.time()
    model.fit(
        X,
        Y,
        valid_X,
        valid_Y,
        epochs=1000,
        batch_size=256,
        cp_name=cp_name,
        verbose=verbose,
    )
    end = time.time()
    return model, end - start
