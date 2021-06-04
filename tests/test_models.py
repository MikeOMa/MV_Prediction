import pytest
from probdrift.mvn_models import models_list
import pandas as pd
from pathlib import Path
import numpy as np
from probdrift import X_VAR, Y_VAR


TEST_DATA_DIR = Path(__file__).resolve().parent / "data/test_dat.csv"
DATA = pd.read_csv(TEST_DATA_DIR)


@pytest.mark.parametrize("model", models_list)
def test_models(model):

    """

    Args:
        model: one of mvn_models
        p: dimension of the output
    """
    mod = model()
    #*100
    Y = DATA[Y_VAR].values*100
    # X = DATA[X_VAR].values
    # NN gets a singular matrix error with the X data unless we scale it.
    # Using a random X instead.
    X = np.random.randn(Y.shape[0],10)
    N_train = int(X.shape[0]*.9)
    mod.fit(X[:N_train], Y[:N_train], X[N_train:], Y[N_train:], retrain=True)
    preds = mod.scipy_distribution(X)
    preds = mod.scipy_distribution(X, cmat_output=True)


@pytest.mark.parametrize("model", models_list)
@pytest.mark.parametrize("p", [3, 4])
def test_p_dim(model, p):
    """
    Generates a sample p dimensional output to fit on.
    Args:
        model: one of mvn_models
        p: dimension of the output
    """
    mod = model()
    N=1000
    Y = np.random.randn(N, p)
    X = np.random.randn(N,10)
    N_train = int(X.shape[0]*.9)
    mod.fit(X[:N_train], Y[:N_train], X[N_train:], Y[N_train:], retrain=True)
    preds = mod.scipy_distribution(X)
    preds = mod.scipy_distribution(X, cmat_output=True)
