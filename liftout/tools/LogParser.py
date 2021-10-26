#!/usr/bin/env python3

import pandas as pd


def parse_log_file(fname):

    supported_feature_types = ["image_centre", "lamella_centre", "needle_tip", "lamella_edge", "landing_post"]
    ml_dict = dict.fromkeys(supported_feature_types, {})

    for feature_type in ml_dict.keys():
        ml_dict[feature_type] = {"True": 0, "False": 0}

    gamma_dict = {"gamma": [], "diff": []}

    # TODO: add a better logging identifier rather than doing this wierd parsing...
    with open(fname) as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            msg = line.split("—")[-1].strip()  # should just be the message # TODO: need to check the delimeter character...
            res = msg.split(":")[0].strip()
            
            if res == "ml_detection":
                key = msg.split(":")[1].strip()
                val = msg.split(":")[-1].strip()

                if key in ml_dict.keys():
                    ml_dict[key][val] +=1
            if res == "gamma_correction":
                gamma_dict["diff"].append(float(msg.split(":")[2].strip()))
                gamma_dict["gamma"].append(float(msg.split(":")[4].strip()))

    return ml_dict, gamma_dict # TODO: need a better way

def plot_ml_data(ml_dict):

    df = pd.DataFrame(ml_dict)

    # plotting
    return df.T.plot.bar(title="Machine Learning Evaluation")
    #### TODO: change df structure?
    # # feature_type #  success ###  ### count
    # would make it easier to analyse... 

def plot_gamma_data(gamma_dict):
    # pprint(gamma_dict)
    df_gamma = pd.DataFrame(gamma_dict)
    # print(df_gamma)

    return df_gamma["gamma"].plot.hist(bins=5, alpha=0.5, title="Gamma Correction Distribution")