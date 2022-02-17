#!/usr/bin/env python3
import os.path

import pandas as pd
from liftout.fibsem.sampleposition import AutoLiftoutStage, SamplePosition
from collections import defaultdict
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_log_file(log_dir):
    fname = os.path.join(log_dir, "logfile.log")

    # ml detection accuracy
    supported_feature_types = ["image_centre", "lamella_centre", "needle_tip", "lamella_edge", "landing_post"]
    ml_dict = dict.fromkeys(supported_feature_types, [])
    for feature_type in ml_dict.keys():
        ml_dict[feature_type] = [0, 0]

    # gamma correction applied to images
    gamma_dict = {"gamma": [], "diff": []}


    # stage durations
    samples = []
    sample = SamplePosition(log_dir, None)
    yaml_file = sample.setup_yaml_file()
    for sample_no in yaml_file["sample"].keys():
        sample = SamplePosition(log_dir, sample_no)
        sample.load_data_from_file()
        samples.append(sample)

    sample_dict = {}
    stages = [stage.name for stage in AutoLiftoutStage]
    for sp in samples:
        sample_dict[str(sp.sample_id)] = dict.fromkeys(stages)
        for stage in sample_dict[str(sp.sample_id)].keys():
            sample_dict[str(sp.sample_id)][stage] = [0, 0]

    # number of clicks # TODO: per sample
    click_dict = dict.fromkeys(stages, 0)   # count

    with open(fname, encoding="cp1252") as f:
        # Note: need to check the encoding as this is required for em dash (long dash) # TODO: change this delimiter so this isnt required.
        lines = f.read().splitlines()
        for i, line in enumerate(lines):

            if line == "":
                continue
            msg = line.split("—")[-1].strip()  # should just be the message # TODO: need to check the delimeter character...
            func = line.split("—")[-2].strip()

            # print(func, msg)

            # # ml parsing
            if "validate_detection" in func:

                sample_id = msg.split("|")[0].strip()
                stage = msg.split("|")[1].strip()
                key = msg.split("|")[-2].strip()
                val = msg.split("|")[-1].strip()

                # print(sample_id, stage, key, val)
                if val == "True":
                    ml_dict[key][0] += 1  # correct
                ml_dict[key][1] += 1  # total

            # gama correction parsing
            if "gamma_correction" in func:
                diff, gam = msg.split("|")[-2:]
                gamma_dict["diff"].append(float(diff))
                gamma_dict["gamma"].append(float(gam))

            # state parsing
            if "stage_update" in func and "|" in msg:

                # print(func, msg)
                ts = line.split("—")[0].split(",")[0]
                dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")

                sp_id = msg.split("|")[0].strip()
                stage = msg.split("|")[-2].strip()
                word = msg.split("|")[-1].strip()

                if word == "STARTED":
                    sample_dict[sp_id][stage][0] = dt
                if word == "FINISHED":
                    sample_dict[sp_id][stage][1] = dt

            if "INIT" in msg:
                # apply the common stages across all samples

                ts = line.split("—")[0].split(",")[0]
                dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")

                stage = msg.split("|")[-2].strip()
                word = msg.split("|")[-1].strip()

                for sp_id in sample_dict:
                    if word == "STARTED":
                        sample_dict[sp_id][stage][0] = dt
                    if word == "FINISHED":
                        sample_dict[sp_id][stage][1] = dt

            # # user interaction
            if "on_gui_click" in func:
                stage = msg.split("|")[0].split(".")[1].strip()
                click = msg.split("|")[1].strip()
                click_dict[stage] += 1

    statistics = {
        "click": dict(click_dict),
        "ml": dict(ml_dict),
        "gamma": dict(gamma_dict),
        "stage": dict(sample_dict)
    }
    return statistics



# TODO: move to utils
def generate_report_data(statistics: dict, log_dir):

    print("Generating Run Statistics and Plots")

    report_dir = os.path.join(log_dir, "report")
    os.makedirs(report_dir, exist_ok=True)


    # gamma
    df_gamma = pd.DataFrame(statistics["gamma"])

    fig = plt.figure()
    plt.hist(df_gamma["gamma"], bins=15, alpha=0.5)
    plt.title("Gamma Correction Distribution")
    plt.xlabel("Gamma Correction")
    plt.ylabel("Count")
    plt.savefig(os.path.join(report_dir, "gamma_statistics.png"))
    plt.show()

    # ml_statistics
    df = pd.DataFrame(statistics["ml"])
    df = df.rename(index={0: "true", 1: "total"})
    df_ml = df.T
    df_ml["false"] = df_ml["total"] - df_ml["true"]
    df_ml["percentage"] = df_ml["true"] / df_ml["total"]

    print(df_ml)
    # plotting
    fig = plt.figure()
    ax = df_ml[["true", "false"]].plot.bar(title="Machine Learning Evaluation (Count)")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(report_dir,"ml_statistics.png"))

    fig = plt.figure()
    ax = df_ml[["percentage"]].plot.bar(title="Machine Learning Evaluation (Accuracy)")
    ax.set_ylabel("Accuracy (%)")
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(report_dir, "ml_accuracy.png"))

    # https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm

    # clicks

    df_click = pd.DataFrame([statistics["click"]])
    ax = df_click.plot.bar(title="Click Evaluation")
    ax.set_ylabel("Number of User Clicks")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best")
    plt.savefig(os.path.join(report_dir, "clicks.png"))

    # stage duration history
    for sp_id in statistics["stage"]:
        print(sp_id)
        stage_dict = statistics["stage"][sp_id]
        stage_duration_dict = dict.fromkeys(list(stage_dict.keys()))
        for stage in stage_dict.keys():
            if stage == "Finished":
                break
            if stage_dict[stage][1] and stage_dict[stage][1]:

                stage_duration = stage_dict[stage][1] - stage_dict[stage][0]
                stage_duration_dict[stage] = stage_duration.total_seconds() / 60
        df_stage = pd.DataFrame([stage_duration_dict])
        ax = df_stage.plot.bar(title=f"Stage Duration ({sp_id[-6:]})")
        ax.set_ylabel("Duration (mins)")
        plt.savefig(os.path.join(report_dir, f"{sp_id}_duration.png"))

    report_statistics = {
        "click": df_click,
        "ml": df_ml,
        # "stage": df_stage
        "gamma": df_gamma
    }

    return report_statistics
