"""
This module has an additional requirement of cartopy, geopandas and shapely.
"""

import numpy as np
from shapely.geometry import Polygon
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple


def joinsdf(splitter, df1: pd.DataFrame, spatial_df):
    """

    Parameters
    ----------
    splitter lon_lat_cutter: which matches spatial_df
    df1 : pd.DataFrame with a lon, lat column
    spatial_df : a pd dataframe with cuts

    Returns
    -------
    df1 with the columns with the matching spatial_df


    """
    # make lon_cut, lat_cut columns
    splitter(df1)
    df1.join(spatial_df, on=["lon_cut", "lat_cut"])
    df1.drop(["lon_cut", "lat_cut"], inplace=True)


def cuts2poly(tuple_tuple: Tuple) -> Polygon:
    """

    Args:
        tuple_tuple: a tuple of pd cuts, designed for .groupby([lon_cut, lat_cut]) operations
    Returns:
         a Shapely Polygon containing the square resulting from the cuts.
    """
    lon, lat = tuple_tuple
    lon1, lon2 = lon.left, lon.right
    lat1, lat2 = lat.left, lat.right
    return Polygon(np.array([(lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2)]))


def spatial_to_gpd(pd_df):
    polys = list(map(cuts2poly, pd_df.index))
    dat = gpd.GeoDataFrame({"geometry": polys}, crs=None)
    for cname in pd_df.columns:
        dat[cname] = pd_df[cname]
    return dat


def plot_column(daf, column, ax, cmap="viridis"):
    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Norm(vmin=daf[column].min(), vmax=daf[column].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    crs = plt.Geodetic()
    for color, rows in daf.groupby("column"):
        color_mapped = sm.cmap(norm(color))
        ax.add_geometries(
            rows["geometry"], crs=crs, facecolor=color_mapped, edgecolor=color_mapped
        )


def make_spatial_map(gdaf, plot_col="count"):
    crs = ccrs.Robinson()
    gdaf.crs = {"init": "epsg:4326"}
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.LogNorm(vmin=gdaf[plot_col].min(), vmax=gdaf[plot_col].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Generate a figure with two axes, one for CartoPy, one for GeoPandas
    fig, axs = plt.subplots(1, 1, subplot_kw={"projection": crs}, figsize=(5.5, 4))

    # Make the CartoPy plot
    crs_proj4 = crs.proj4_init
    df_ae = gdaf.to_crs(crs_proj4)
    for color, rows in df_ae.groupby("count"):
        print("hi")
        color_mapped = sm.cmap(norm(color))
        axs.add_geometries(
            rows["geometry"], crs=crs, facecolor=color_mapped, edgecolor=color_mapped
        )
    axs.coastlines()
    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])  ### x0 y0 width height


def lonto180(x):
    if x > 180:
        ret = x - 360
    else:
        ret = x
    return ret


def plot_summary(summary_df, ax=None, is_sym=True, cbar_name="seismic"):
    dat = spatial_to_gpd(summary_df)
    plot_col = "count"
    gdaf = dat.dropna()
    gdaf.crs = {"init": "epsg:4326"}
    cmap = plt.get_cmap(cbar_name)
    if is_sym:
        cmap_range = max((-gdaf[plot_col].min(), gdaf[plot_col].max()))
        norm = mpl.colors.Normalize(vmin=-cmap_range, vmax=cmap_range)
    else:
        norm = mpl.colors.Normalize(
            vmin=gdaf[plot_col].min(), vmax=gdaf[plot_col].max()
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # Generate a figure with two axes, one for CartoPy, one for GeoPandas
    if ax is None:
        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": crs}, figsize=(5.5, 4))
    else:
        fig = ax.figure
        crs = ax.projection

    # Make the CartoPy plot
    crs_proj4 = crs.proj4_init
    df_ae = gdaf.to_crs(crs_proj4)
    for color, rows in df_ae.groupby("count"):
        color_mapped = sm.cmap(norm(color))
        # print(norm(color))
        ax.add_geometries(
            rows["geometry"], crs=crs, facecolor=color_mapped, edgecolor=color_mapped
        )
    ###Add coastline
    ax.coastlines()
    ### Add colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cbar = plt.colorbar(sm, cax=ax_cb)
    return ax
