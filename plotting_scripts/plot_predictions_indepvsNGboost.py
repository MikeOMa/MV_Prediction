from core_functions import (
    data,
    read_models,
    models_list,
    indep_models_list,
    folds,
    min_max_scaler,
)
from probdrift import X_VAR, Y_VAR, mpl_config
from probdrift.MVN_helpers import elipse_points, in_ci
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker as mticker

cproj = ccrs.TransverseMercator(
    central_longitude=-78, central_latitude=28, approx=False
)
pyprojection = pyproj.Proj(**cproj.proj4_params)

projection = pyproj.Proj(
    proj="tmerc",
    lat_0=28,
    lon_0=-78,
    k=1.00000,
    x_0=0,
    y_0=0,
    towgs84=[-90.7, -106.1, -119.2, 4.09, 0.218, -1.05, 1.37],
    units="km",
)

counts = data[folds[0][1]].groupby("id")["u"].count().sort_values()
drift_id = 54386
# for drift_id, ax in zip(ids_to_plot, axs.flatten()):
df_to_plot = data.query(f"id=={drift_id}").iloc[:13].reset_index(drop=True)
gap = 2


def arrow_vel_proj(projection, df_row, scipy_dist, ax, scale=1):
    u = [df_row["u"] * scale, df_row["v"] * scale]
    pos = projection.transform(df_row["lon"], df_row["lat"])
    ax.arrow(
        *pos,
        *u,
        label="Truth",
        color="black",
        head_width=0.15,
        length_includes_head=True,
    )
    u_pred = scipy_dist.mean * 1e-2 * scale
    ax.arrow(
        *pos,
        *u_pred,
        label="Mean Prediction",
        color="red",
        head_width=0.15,
        length_includes_head=True,
        transform=ax.get_transform(),
    )
    # plot_elipse_proj(projection, df_row, scipy_dist, ax, scale=scale)
    error_unscaled = df_row[["u", "v"]].to_numpy() - scipy_dist.mean * 1e-2
    print(in_ci(error_unscaled, scipy_dist.cov * (1e-2 ** 2), 0.70))


def plot_elipse_proj(projection, df_row, scipy_dist, ax, scale=1):
    points = elipse_points(scipy_dist.cov * (1e-2 ** 2) * (scale ** 2), 0.70)
    x, y = projection.transform(df_row["lon"], df_row["lat"])
    ax.plot(
        x + scipy_dist.mean[0] * scale * 1e-2 + points[:, 0],
        y + scipy_dist.mean[1] * scale * 1e-2 + points[:, 1],
        color="red",
        linestyle="--",
        label="70\% PR",
    )


import string


def add_letters(axs):
    for n, ax in enumerate(axs.flatten()):
        ax.text(
            -0.15,
            0.95,
            "\\textbf{" + string.ascii_uppercase[n] + ")}",
            transform=ax.transAxes,
            size=13,
            weight="bold",
        )


def add_xy_grid_lines(ax):
    x_ticks = list(range(-200, 401, 200))
    y_ticks = list(range(-200, 801, 200))
    print(x_ticks)
    print(y_ticks)
    x_ticks_in_ll = [
        pyprojection.transform(x_tick * 1000, min(y_ticks) * 1000, direction="INVERSE")[
            0
        ]
        for x_tick in x_ticks
    ]
    y_ticks_in_ll = [
        pyprojection.transform(min(x_ticks) * 1000, y_tick * 1000, direction="INVERSE")[
            1
        ]
        for y_tick in y_ticks
    ]
    gl = add_gridlines(ax)
    gl.xlocator = mticker.FixedLocator(x_ticks_in_ll)
    gl.ylocator = mticker.FixedLocator(y_ticks_in_ll)
    gl.xlines = True
    gl.ylines = True
    gl.xformatter = mticker.FuncFormatter(
        lambda x, u: nice_km_format(pyprojection.transform(x, min(y_ticks_in_ll))[0])
    )
    gl.yformatter = mticker.FuncFormatter(
        lambda x, u: nice_km_format(pyprojection.transform(min(x_ticks_in_ll), x)[1])
    )
    return gl


def add_gridlines(ax):
    import matplotlib.ticker as mticker

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    gl.xlocator = mticker.FixedLocator([-85, -70, -55, -40])
    gl.ylocator = mticker.FixedLocator(range(0, 66, 15))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return gl
def nice_km_format(coord_in_meters):
    return str(int(np.round(coord_in_meters/10000)*10))+"km"

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))


def plot_traj_preds_northing(df_to_plot, models_list, ax):
    idxs = list(range(0, df_to_plot.shape[0], gap))

    df_to_ellipse = df_to_plot.iloc[idxs, :]
    ax.plot(
        df_to_plot["lon"],
        df_to_plot["lat"],
        "b-o",
        alpha=0.1,
        transform=ccrs.PlateCarree(),
    )
    ax.coastlines()
    pred_distns = models_list[0].scipy_distribution(
        min_max_scaler.transform(df_to_ellipse[X_VAR])
    )
    # units are in m/s so convert to km/day 60(sec)*60(min
    scale = 60 * 60 * 24  # m/day. Underlying units are in days
    ax.set_aspect("equal")
    for (i, df_row), scipy_dist in zip(df_to_ellipse.iterrows(), pred_distns):
        u = [df_row["u"] * scale, df_row["v"] * scale]
        pos = pyprojection.transform(df_row["lon"], df_row["lat"])
        ax.arrow(
            *pos,
            *u,
            label="Truth",
            color="black",
            head_width=1e4,
            length_includes_head=True,
        )
        u_pred = scipy_dist.mean * 1e-2 * scale
        ax.arrow(
            *pos,
            *u_pred,
            label="Mean Prediction",
            color="red",
            head_width=1e4,
            length_includes_head=True,
        )
        plot_elipse_proj(pyprojection, df_row, scipy_dist, ax, scale=scale)
        error_unscaled = df_row[["u", "v"]].to_numpy() - scipy_dist.mean * 1e-2
        # print(in_ci(error_unscaled, scipy_dist.cov*(1e-2**2), 0.70))
        arrow_vel_proj(pyprojection, df_row, scipy_dist, ax=ax, scale=100)

    xy_for_scale_arrow = pyprojection.transform(-78, 28)
    ax.arrow(
        *xy_for_scale_arrow,
        scale,
        scale,
        color="C2",
        label="(1,1)m/s scaled to km/day",
        head_width=1e4,
    )


fig, axs = plt.subplots(1, 2, figsize=(10, 6), subplot_kw={"projection": cproj})
plot_traj_preds_northing(df_to_plot, models_list, axs[0])
plot_traj_preds_northing(df_to_plot, indep_models_list, axs[1])

fig.suptitle(f"Drifter ID {drift_id} Trajectory")
axs[0].set_title("Predictions from Multivariate NGB")
axs[1].set_title("Predictions from Indep NGB")
fig.subplots_adjust(bottom=-0.2)
add_letters(axs)

legend_without_duplicate_labels(axs[1])
gl = add_xy_grid_lines(axs[0])
gl = add_xy_grid_lines(axs[1])
# gl.xformatter = mticker.FixedFormatter([str(x_tick)+"km" for x_tick in x_ticks])
# gl.bottom_labels

fig.savefig(f"../Images/Multivariate_vs_Independent_{gap}_km.pdf", bbox_inches="tight")
