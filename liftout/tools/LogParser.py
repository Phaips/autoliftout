#!/usr/bin/env python3

import pandas as pd


def parse_log_file(fname):

    supported_feature_types = ["image_centre", "lamella_centre", "needle_tip", "lamella_edge", "landing_post"]
    ml_dict = dict.fromkeys(supported_feature_types, {})

    for feature_type in ml_dict.keys():
        ml_dict[feature_type] = {"True": 0, "False": 0}

    gamma_dict = {"gamma": [], "diff": []}


    from liftout.gui.main import AutoLiftoutStatus

    stages = [stage.name for stage in AutoLiftoutStatus]
    stage_dict = dict.fromkeys(stages)
    for stage in stage_dict.keys():
        stage_dict[stage] = {"STARTED": None, "FINISHED": None}

    # TODO: add a better logging identifier rather than doing this weird parsing...
    with open(fname) as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            msg = line.split("—")[-1].strip()  # should just be the message # TODO: need to check the delimeter character...
            res = msg.split(":")[0].strip()

            # print(line)
            if res == "ml_detection":
                key = msg.split(":")[1].strip()
                val = msg.split(":")[-1].strip()

                if key in ml_dict.keys():
                    ml_dict[key][val] += 1

            if res == "gamma_correction":
                data = msg.split(",")
                # print(data)
                gamma_dict["diff"].append(float(data[0].split(":")[2].strip()))
                gamma_dict["gamma"].append(float(data[1].split(":")[1].strip()))

            if msg.__contains__("STARTED") or msg.__contains__("FINISHED"):
                # NOTE: gonna break if more than 1 lamella is completed in a run...
                if "LOAD COORDINATES" in msg:
                    continue
                stage = msg.split(" ")[0].strip()
                status = msg.split(" ")[-1].strip()
                time = line.split("—")[0].split(",")[0]  # don't care about ms
                import datetime
                dt_object = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
                stage_dict[stage][status] = dt_object
            if msg.__contains__("Perform"):
                dat = msg.split(":")
                stage_dict[dat[0].split()[1].strip()]["PERFORM"] = dat[1]

    return ml_dict, gamma_dict, stage_dict # TODO: need a better way


def plot_ml_data(ml_dict):

    df = pd.DataFrame(ml_dict)

    # plotting
    return df.T.plot.bar(title="Machine Learning Evaluation"), df
    #### TODO: change df structure?
    # # feature_type #  success ###  ### count
    # would make it easier to analyse... 

def plot_gamma_data(gamma_dict):

    df_gamma = pd.DataFrame(gamma_dict)

    return df_gamma["gamma"].plot.hist(bins=30, alpha=0.5, title="Gamma Correction Distribution"), df_gamma

def plot_state_dict(state_dict):
    state_duration_dict = {}
    for state in state_dict.keys():
        #     print(stage_dict[state])
        if state_dict[state]["FINISHED"] and state_dict[state]["STARTED"]:
            state_duration = state_dict[state]["FINISHED"] - state_dict[state]["STARTED"]
            print(f"{state}: {state_duration}")
            state_duration_dict[state] = state_duration.total_seconds()

    # import pandas as pd

    print(state_duration_dict)
    df = pd.DataFrame([state_duration_dict])
    return df.plot.bar(title="State Duration"), df