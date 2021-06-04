import pytest
import numpy as np
import scipy as sp
from probdrift.metrics import METRIC_LIST

y = np.random.rand(5, 2)
dists = [
    sp.stats.multivariate_normal(mean=np.zeros(2) + 1, cov=np.identity(2) * (i + 1))
    for i in range(y.shape[0])
]


@pytest.mark.parametrize("metric", METRIC_LIST.values())
def test_metrics(metric):
    metric(dists, y)
