import json
import numpy as np
import yaml

from probdrift.mvn_models.mvn_ngboost import DEFAULT_BASE
from probdrift.mvn_models.mvn_ngboost import DEFAULT_NGBOOST
from probdrift.mvn_models.mv_lgbm import DEFAULT_LGBM
from probdrift.mvn_models.mv_gbm import DEFAULT_GBM
import itertools as it

CONFIG_DIR = "config_files_leaf/"
min_leaf_data = [1, 15,32]
#subsample_list = [0.7,1]
max_depths = [15]
max_leaves = [8, 15, 32, 64]
LGBM_BASE_TRIALS = {"num_leaves": max_leaves, "min_data_in_leaf":min_leaf_data, "max_depth":max_depths}
LGBM_TRIALS = {**LGBM_BASE_TRIALS}

BASE_TRIALS = {"max_leaf_nodes":max_leaves, "min_samples_leaf": min_leaf_data, "max_depth":max_depths}
NGBOOST_TRIALS = {**BASE_TRIALS}

NN_trials = {"hidden_layers": [[20], [50], [100], [20, 20], [50,20], [100,20]], "learning_rate": [0.01, 0.001]}

def LGBM_config(new_dict, idx):
    out_dict = {}
    params_dict = DEFAULT_LGBM.copy()
    params_dict.update(new_dict)
    out_dict["modelname"] = "LGBM"
    params_dict['learning_rate'] = 0.1
    out_dict["params"] = params_dict
    out_dict["fname"] = num_to_len3(idx) + out_dict["modelname"]
    return out_dict

def GBM_config(new_dict, idx):
    out_dict = {}

    out_dict["modelname"] = "GBM"
    params = DEFAULT_GBM.copy()

    params["learning_rate"] = 0.1  # new_dict["minibatch_frac"]
    if "max_depth" in new_dict:
        params['max_depth'] = new_dict["max_depth"]
    params['min_samples_leaf'] = new_dict["min_samples_leaf"]
    params['max_leaf_nodes'] = new_dict["max_leaf_nodes"]
    out_dict["params"] = params
    out_dict["fname"] = num_to_len3(idx) + out_dict["modelname"]
    return out_dict

def num_to_len3(x):
    x = str(x)
    return "0" * (3 - len(x)) + str(x)


def NGBoost_config(new_dict, idx):
    out_dict = {}

    out_dict["modelname"] = "mvn_ngboost"
    params_ngboost = DEFAULT_NGBOOST.copy()

    params_ngboost["learning_rate"] = 0.1
    out_dict["params_ngboost"] = params_ngboost
    params_base = DEFAULT_BASE.copy()
    if "max_depth" in new_dict:
        params_base['max_depth'] = new_dict["max_depth"]
    params_base['min_samples_leaf'] = new_dict["min_samples_leaf"]
    params_base['max_leaf_nodes'] = new_dict["max_leaf_nodes"]
    out_dict["params_base"] = params_base
    out_dict["fname"] = num_to_len3(idx) + out_dict["modelname"]
    return out_dict


def get_combination_dicts(options_dict):
    allNames = sorted(options_dict.keys())
    combinations = it.product(*(options_dict[Name] for Name in allNames))
    ret = []
    for k in combinations:
        ret.append({name: val for name, val in zip(allNames, k)})
    return ret


if __name__ == "__main__":
    # LGBM_Configs
    count = 0
    grid_search_lgbm = get_combination_dicts(LGBM_TRIALS)
    all_configs = []
    for gs in grid_search_lgbm:
        d = LGBM_config(gs, count)
        all_configs.append(d)
        count += 1

    grid_search_ngboost = get_combination_dicts(NGBOOST_TRIALS)
    for gs in grid_search_ngboost:
        d = NGBoost_config(gs, count)
        count+=1
        all_configs.append(d)
        # Ordinary gradeint boosting
        # Just set natural gradient to False
        og_boost = NGBoost_config(gs, count)
        og_boost["params_ngboost"]["natural_gradient"] = False
        og_boost["fname"] += "_og"
        all_configs.append(og_boost)
        count += 1

        indep_boost = NGBoost_config(gs, count)
        indep_boost["modelname"] = "indep_ngboost"
        indep_boost["fname"] =  num_to_len3(count)+indep_boost["modelname"]
        all_configs.append(indep_boost)
        count += 1


    for gs in grid_search_ngboost:
        d = GBM_config(gs, count)
        count+=1
        all_configs.append(d)


    for gs in get_combination_dicts(NN_trials):
        nn_config = {"modelname": "nn", "fname": num_to_len3(count) + "nn", **gs}
        count+=1
        all_configs.append(nn_config)

    for d in all_configs:
        yaml.dump(d, stream=open(CONFIG_DIR + d["fname"] + ".yaml", "w"))
