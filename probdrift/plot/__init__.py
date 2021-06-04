"""
Some functions for plotting results
Cartopy required for this module
"""

import probdrift.spatialfns.gpd_functions as gpd_fn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def cuts_to_geometry(pd_df):
    polys = list(map(gpd_fn.cuts2poly, pd_df.index))
    return polys


def plot_column(daf, column, ax, cmap="viridis", norm=None):
    if "geometry" not in daf.columns:
        daf["geometry"] = cuts_to_geometry(daf)
    cmap = plt.get_cmap(cmap)
    if norm is None:
        norm = mpl.colors.Normalize(vmin=daf[column].min(), vmax=daf[column].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.05, axes_class=plt.Axes)
    f = ax.get_figure()
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    f.add_axes(cax)
    crs = ccrs.PlateCarree()
    for color, rows in daf.groupby(column):
        color_mapped = sm.cmap(norm(color))
        ax.add_geometries(
            rows["geometry"], crs=crs, facecolor=color_mapped, edgecolor=color_mapped
        )
    return cax
