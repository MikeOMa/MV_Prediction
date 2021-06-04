import pandas as pd
import numpy as np


class lon_lat_cutter:
    def __init__(self, spatial_res):
        self.lon_grid = np.arange(-180, 180.1, spatial_res)
        self.lat_grid = np.arange(-90, 90.1, spatial_res)

    def __call__(self, df):
        df["lon_cut"] = pd.cut(df["lon"], self.lon_grid)
        df["lat_cut"] = pd.cut(df["lat"], self.lat_grid)


def filter_count(df, cut_off=25, spatial_size=1):
    """
    Function used to drop the boxes with less than 25 observations in the dataset

    Parameters
    ----------
    df: Dataframe to filter, with a lon, lat column
    cut_off: the count to take as number of required observations in a bin
    spatial_size: The size of the spatial bin in degrees

    Returns
    -------
    A mask saying which rows can be dropped
    """

    spatial_info = df[["lon", "lat"]].copy()
    cutter = lon_lat_cutter(spatial_size)
    cutter(spatial_info)
    count_by_grid = spatial_info.groupby(["lon_cut", "lat_cut"], sort=False).size()
    count_by_grid.name = "number_in_box"
    spatial_info = spatial_info.join(count_by_grid, on=["lon_cut", "lat_cut"])
    return spatial_info["number_in_box"] > cut_off
