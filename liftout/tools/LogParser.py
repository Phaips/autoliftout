#!/usr/bin/env python3

import pandas as pd


def parse_log_file(fname):

    supported_feature_types = ["image_centre", "lamella_centre", "needle_tip", "lamella_edge", "landing_post"]
    ml_dict = dict.fromkeys(supported_feature_types, {})

    for feature_type in ml_dict.keys():
        ml_dict[feature_type] = {"True": 0, "False": 0}

    gamma_dict = {"gamma": [], "diff": []}

    try:
        from liftout.gui.main import AutoLiftoutStage
    except ModuleNotFoundError:
        from enum import Enum
        class AutoLiftoutStage(Enum):
            Initialisation = -1
            Setup = 0
            Milling = 1
            Liftout = 2
            Landing = 3
            Reset = 4
            Thinning = 5
            Finished = 6
    
    stages = [state.name for state in AutoLiftoutStage] + ["SINGLE_LIFTOUT"]
    state_dict = dict.fromkeys(stages)
    for state in state_dict.keys():
        state_dict[state] = {"STARTED": None, "FINISHED": None}


    # TODO: add a better logging identifier rather than doing this weird parsing...
    with open(fname, encoding="cp1252") as f:
        # Note: need to check the encoding as this is required for em dash (long dash) # TODO: change this delimiter so this isnt required.
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if line == "":
                continue
            msg = line.split("—")[-1].strip()  # should just be the message # TODO: need to check the delimeter character...
            func = line.split("—")[-2].strip()


            # ml parsing
            if "validate_detection" in func:

                key = msg.split(":")[1].strip()
                val = msg.split(":")[-1].strip()

                if key in ml_dict.keys():
                    ml_dict[key][val] += 1


            # gama correction parsing
            if "gamma_correction" in func:
                # print(func)
                data = msg.split(",")
                # print(data)
                gamma_dict["diff"].append(float(data[0].split(":")[2].strip()))
                gamma_dict["gamma"].append(float(data[1].split(":")[1].strip()))



            # state parsing
            if msg.__contains__("STARTED") or msg.__contains__("FINISHED"):
                # NOTE: gonna break if more than 1 lamella is completed in a run...
                if "LOAD COORDINATES" in msg:
                    continue
                state = msg.split(" ")[0].strip()
                status = msg.split(" ")[-1].strip()
                time = line.split("—")[0].split(",")[0]  # don't care about ms
                import datetime
                dt_object = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
                if "|" not in msg: # temp solution until we account for sub-states
                    # print(state, status, msg)
                    state_dict[state][status] = dt_object
            if msg.__contains__("Perform"):
                dat = msg.split(":")
                state_dict[dat[0].split()[1].strip()]["PERFORM"] = dat[1]

            # user interaction
            if "on_gui_click" in func:
                # print("CLICK MESSAGE")
                # print(line)
                # print(msg)
                pass

    return ml_dict, gamma_dict, state_dict # TODO: need a better way


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
        #     print(state_dict[state])
        if state_dict[state]["FINISHED"] and state_dict[state]["STARTED"]:
            state_duration = state_dict[state]["FINISHED"] - state_dict[state]["STARTED"]
            # print(f"{state}: {state_duration}")
            state_duration_dict[state] = state_duration.total_seconds()

    # import pandas as pd

    # print(state_duration_dict)
    df = pd.DataFrame([state_duration_dict])
    return df.plot.bar(title="State Duration"), df