import pandas as pd
from probdrift import Y_VAR, X_VAR

col_subset = X_VAR + Y_VAR + ["id"]

dat = pd.read_csv("atlantic_oc.csv")
dat.sort_values("t", inplace=True)
subset = dat[col_subset]
to_save = subset.iloc[:1000, :]
to_save.to_csv("test_dat.csv", index=False)
