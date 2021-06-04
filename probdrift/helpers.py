import os
import numpy as np


def get_num_threads():
    return getattr(os.environ, "num_threads", 4)


def id_aug(timevec, cut_off=6):
    times = timevec.values
    diff_times = np.diff(timevec)
    mask = diff_times > np.timedelta64(cut_off, "h")
    mask = np.insert(mask, 0, False)
    csum = np.cumsum(mask)
    return csum
