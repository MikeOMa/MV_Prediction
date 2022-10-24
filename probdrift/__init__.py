X_VAR = ["Tx", "Ty", "Wx", "Wy", "u_av", "v_av", "lon", "lat", "t"]
Y_VAR = ["u", "v"]
mpl_config = {
    "text.usetex": True,
    "font.size": 12,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "Times New Roman",
}

FEATURE_NAME_MAP = {
    "Tx": "Longitudinal Wind Stress",
    "Ty": "Latitudinal Wind Stress",
    "u_av": "Longitudinal Geostrophic Velocity Anomaly",
    "v_av": "Latitudinal Geostrophic Velocity Anomaly",
    "Wx": "Longitudinal Wind Speed",
    "Wy": "Latitudinal Wind Speed",
    "lon": "Longitude",
    "lat": "Latitude",
    "t": "Days Since 1 $-$ Jan (capped at 366)",
    "u": "Longitudinal Drifter velocity (Y1)",
    "v": "Latitudinal Drifter velocity (Y2)",
}
ms_units = "$m~s^{-1}$"
UNITS_MAP = {
    "Tx": "Pa",
    "Ty": "Pa",
    "u_av": ms_units,
    "v_av": ms_units,
    "Wx": ms_units,
    "Wy": ms_units,
    "lon": "Degrees",
    "lat": "Degrees",
    "t": "days",
    "u": ms_units,
    "v": ms_units,
}
