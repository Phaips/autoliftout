
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from liftout.structures import Sample, Lamella, AutoLiftoutState, load_experiment


@dataclass
class AutoLiftoutStatistics:
    gamma: pd.DataFrame
    click: pd.DataFrame
    move: pd.DataFrame
    ml: pd.DataFrame
    sample: pd.DataFrame
    history: pd.DataFrame
    image: pd.DataFrame = None
    name: str = "name"


def calculate_statistics_dataframe(path: Path) -> AutoLiftoutStatistics:

    fname = os.path.join(path, "logfile.log")
    gamma_info = []
    click_info = []
    move_info = []
    ml_info = []

    current_lamella = None 
    current_stage = "Setup"
    current_step = None

    with open(fname, encoding="cp1252") as f:
        # Note: need to check the encoding as this is required for em dash (long dash) # TODO: change this delimiter so this isnt required.
        lines = f.read().splitlines()
        for i, line in enumerate(lines):

            if line == "":
                continue
            try:
                msg = line.split("—")[
                    -1
                ].strip()  # should just be the message # TODO: need to check the delimeter character...
                func = line.split("—")[-2].strip()

                if "log_status_message" in func:
                    current_lamella = msg.split("|")[1].strip()
                    current_stage = msg.split("|")[2].strip().split(".")[-1].strip()
                    current_step = msg.split("|")[3].strip()

                if "gamma" in func:
                    beam_type, diff, gamma = msg.split("|")[-3:]
                    beam_type = beam_type.strip()
                    if beam_type in ["Electron", "Ion", "Photon"]:
                        gamma_d = {
                            "beam_type": beam_type,
                            "diff": float(diff),
                            "gamma": float(gamma),
                            "lamella": current_lamella,
                            "stage": current_stage,
                            "step": current_step,
                        }
                        gamma_info.append(deepcopy(gamma_d))
                                                
                if "double_click" in func:
                    split_msg = msg.split("|")
                    click_type = split_msg[0].split(":")[-1].strip()
                    beam_type = split_msg[-1].split(".")[-1]
                    pos = split_msg[-2].strip().split(" ")
                    pos_x = pos[1].replace(",", "")
                    pos_y = pos[2]
                    click_d = {
                        "beam_type": beam_type,
                        "type": click_type,
                        "x": float(pos_x),
                        "y": float(pos_y),
                        "lamella": current_lamella,
                        "stage": current_stage,
                        "step": current_step,
                    }
                    click_info.append(deepcopy(click_d))

                if "on_click" in func:
                    split_msg = msg.split("|")
                    if "DectectedFeature" == split_msg[0].split(" ")[0].strip():
                        # click_type = split_msg[0].split(":")[-1].strip()
                        click_type = "DetectedFeature"
                        if "BeamType" in split_msg[-1]:
                            beam_type = split_msg[-1].split(".")[-1]
                        else:
                            beam_type = "None"
                        # beam_type = "ELECTRON" # TODO: need to fix this
                        print(split_msg)
                        pos = split_msg[-1].strip().split(", ")
                        pos_x = pos[0]
                        pos_y = pos[1]
                        click_d = {
                            "beam_type": beam_type,
                            "type": click_type,
                            "x": float(pos_x),
                            "y": float(pos_y),
                            "lamella": current_lamella,
                            "stage": current_stage,
                            "step": current_step,
                        }
                        click_info.append(deepcopy(click_d))


                # ml
                if "Feature" == msg.split("|")[0].strip():
                    # print(msg)
                    feature_type = msg.split("|")[1].split(".")[-1].strip()
                    correct = msg.split("|")[2].strip()
                    ml_d = {
                        "feature": feature_type,
                        "correct": correct,
                        "lamella": current_lamella,
                        "stage": current_stage,
                        "step": current_step,
                    }
                    ml_info.append(deepcopy(ml_d))

                if "move_stage" in func:
                    if "move_stage_relative" in func:
                        beam_type = "ION"
                        mode = "Stable"
                        split_msg = [char.split("=") for char in msg.split(" ")[-3:]]
                        x, y, z = [m[1].replace(",", "") for m in split_msg]
                        z = z.replace(")", "")

                    if "move_stage_eucentric" in func:
                        # TODO: add beam here
                        beam_type = "ION"
                        mode = "Eucentric"
                        z = msg.split(" ")[-1].split("=")[-1].replace(")", "")
                        x, y = 0, 0

                    move_d = {
                        "beam_type": beam_type,
                        "mode": mode,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "lamella": current_lamella,
                        "stage": current_stage,
                        "step": current_step,
                    }
                    move_info.append(deepcopy(move_d))
            except Exception as e:
                pass #print(f"EXCEPTION: {msg} {e}")

            # gamma
            # clicks
            # crosscorrelation
            # ml
            # history
            # sample

    # sample
    sample = load_experiment(path)
    df_sample = sample.__to_dataframe__()
    df_history = create_history_dataframe(sample)


    return AutoLiftoutStatistics(
        gamma=pd.DataFrame.from_dict(gamma_info),
        click=pd.DataFrame.from_dict(click_info),
        move=pd.DataFrame.from_dict(move_info),
        ml=pd.DataFrame.from_dict(ml_info),
        sample=df_sample,
        history=df_history,
        name=sample.name,
    )


def create_history_dataframe(sample: Sample) -> pd.DataFrame:
    history = []
    lam: Lamella
    hist: AutoLiftoutState
    for lam in sample.positions.values():

        petname = lam._petname

        for hist in lam.history:
            start, end = hist.start_timestamp, hist.end_timestamp
            stage_name = hist.stage.name

            hist_d = {
                "petname": petname,
                "stage": stage_name,
                "start": start,
                "end": end,
            }
            history.append(deepcopy(hist_d))

    df_stage_history = pd.DataFrame.from_dict(history)
    df_stage_history["duration"] = df_stage_history["end"] - df_stage_history["start"]

    return df_stage_history


def calculate_aggregated_statistics(paths: list[Path]):
    # TODO: add more over time
    agg_stats = []

    for path in paths:

        stats = calculate_statistics_dataframe(path)

        stats_d = {
            "name": stats.name,
            "lamella": len(stats.sample),
            "clicks": len(stats.click),
            "gamma": stats.gamma["gamma"].mean(),
        }

        agg_stats.append(deepcopy(stats_d))

    return pd.DataFrame.from_dict(agg_stats)
