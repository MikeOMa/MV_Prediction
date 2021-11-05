import pandas as pd
import os
import yaml
import pickle
from probdrift.metrics import METRIC_LIST
from probdrift.validation_strategy import id_splitter
from scipy.stats import multivariate_normal
import numpy as np

from probdrift import X_VAR, Y_VAR

from dispatcher import random_id_subset, data
import sys

def lists_to_mvn(mean, cov):
    N = len(cov)
    return [multivariate_normal(mean[i,:], cov[i,:,:], allow_singular=True) for i in range(N)]

class experiment_result:
    def __init__(self, config_file, y, folds, base_dir = "results19may_500"):
        self.params = yaml.load(open(config_file, "r"))
        self.name = self.params['fname']
        self.base_dir = base_dir
        self._metrics = self._compute_metrics(y, folds)
    def to_row(self):
        row = []
        row.append(self.name)
        row = [i for i in self.metric()]
        return row
    def _compute_metrics(self, y, folds):
        means, covs = read_result(folds,self.base_dir+"/"+self.name+'/')
        dists = [lists_to_mvn(mean, cov) for mean, cov in zip(means, covs)]
        metrics_dict = {key:[] for key in METRIC_LIST.keys()}
        for key,metric in METRIC_LIST.items():
            for dist, fold in zip(dists, folds):
                metrics_dict[key].append(np.mean(metric(dist,y[fold[1],:])))
        return metrics_dict
    def to_dict(self):
        out_dict = {name:np.mean(values)
                    for name, values in self._metrics.items()}
        for _str in ["x", "y"]:
            if "MSE_"+_str in out_dict.keys():
                out_dict['RMSE_'+_str] = np.sqrt(out_dict['MSE_'+_str])
        return out_dict

    def to_pd(self):
        df = pd.DataFrame(self._metrics)
        df['name'] = self.name
        return df

def read_result(folds, foldername):
    """

    Parameters
    ----------
    fname: The filename to search for the result in
    folds: The list of fold;

    Returns
    -------
    covs: list of covariances. covs[i] corresponding to the data in folds[i]
    means: list of means corresponding to the same distributions as covs
    """
    means = []
    covs = []
    for i, idxs in enumerate(folds):
        fname = foldername+"fold_"+str(i)+"scipy_dists.p"
        mean, cov = pickle.load(open(fname, 'rb'))
        means.append(mean)
        covs.append(cov)
    return means, covs


if __name__ == "__main__":
    #del METRIC_LIST['ES']

    num_to_run = int(sys.argv[-1])
    allfiles = os.listdir("config_files_leaf")
    allfiles.sort()
    config_file = allfiles[num_to_run]
    allfiles[num_to_run]
    ids = data['id'].values
    y = data[['u', 'v']].values*100
    X = data[X_VAR]
    N = y.shape[0]
    np.random.seed(500)

    config_file_full_path = "config_files_leaf/"+config_file
    folds = [random_id_subset(ids,0.1) for i in range(10)]
    metric_class = experiment_result(config_file_full_path, y, folds)

    metric_class.to_pd().to_csv("metrics/"+config_file.split('.')[0]+'.csv')