from scipy.stats import multivariate_normal
import numpy as np
import pickle


def fill_nan(a):
    col_mean = np.nanmean(a, axis=0)
    # Find indices that you need to replace
    inds = np.where(np.isnan(a))

    # Place column means in the indices. Align the arrays using take
    a[inds] = np.take(col_mean, inds[1])
    return a


class predictor:
    """
    Attributes inherited from all classes
    __init__ fills the class based on a dictionary
    save pickles itself, this is for convience and one does not need to convert
    self.fname to a filename etc.
    """

    def __init__(self, params={}):
        for key in params.keys():
            setattr(self, key, params[key])

    def save(self, fname="model_1"):
        ## Use fname if the fname is defined.
        pickle.dump(self, open(getattr(self, "fname", fname) + ".p", "wb"))


class mvn_predictor(predictor):
    """
    Template, nothing defined as mvn predictors have custom treatments
    """

    def summary(self):
        pass


class mv_predictor(predictor):
    """
    A class to inherit from for univariate models with a .predict method.
    Each dimension is assigned a homogeneous variance
    """

    def predict(self, X):
        """

        Parameters
        ----------
        X: np.ndarraylike Input

        Returns
        -------
        An X.shape[0], p np.ndarray with predictions stored

        """
        p = len(self.models)
        preds = np.zeros((X.shape[0], p))
        for i in range(p):
            preds[:, i] = self.models[i].predict(X)
        return preds

    def _estimate_err(self, X_train, y_train):
        """
        Stores an error estimate for each dimension

        Parameters
        ----------
        X_train: training explanatory variables
        y_train: training target variables (2d)

        Returns
        -------
        the variance of the errors.


        """

        preds = self.predict(X_train)
        errors = y_train - preds
        self.variance = np.var(errors, ddof=1, axis=0)
        return self.variance

    def scipy_distribution(self, X: np.ndarray, cmat_output=False):
        """

        Args:
            X (object):
        """
        N = X.shape[0]
        preds = self.predict(X)
        err = self.variance
        print(err)
        cmats = [np.diag(err) for i in range(N)]
        if cmat_output:
            out = [preds, np.stack(cmats)]
        else:
            out = [
                multivariate_normal(mean=preds[i, :], cov=cmats[i]) for i in range(N)
            ]
        return out
