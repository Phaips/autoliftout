#!/usr/bin/env python3
import os.path

from liftout.fibsem.sampleposition import AutoLiftoutStage, SamplePosition
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
    gamma_dict = {"gamma": {"Electron": [], "Ion": [], "Photon": [], "all": []},
                  "diff": []}

    # cross correlation success
    cc_dict = {"True": 0, "Total": 0}

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
                beam_type, diff, gam = msg.split("|")[-3:]
                beam_type = beam_type.strip()
                if beam_type in ["Electron", "Ion", "Photon"]:
                    # print(beam_type, diff, gam)
                    gamma_dict["gamma"][beam_type].append(float(gam))
                gamma_dict["diff"].append(float(diff))
                gamma_dict["gamma"]["all"].append(float(gam))

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

            if "CROSSCORRELATION" in msg:

                mode = msg.split("|")[-2].strip()
                suc = msg.split("|")[-1].strip()
                if suc == "True":
                    cc_dict["True"] += 1
                cc_dict["Total"] += 1

    statistics = {
        "click": dict(click_dict),
        "ml": dict(ml_dict),
        "gamma": dict(gamma_dict),
        "stage": dict(sample_dict),
        "crosscorrelation": dict(cc_dict)
    }
    return statistics

# TODO: move to utils
def generate_report_data(statistics: dict, log_dir, show=False):

    print("Generating Run Statistics and Plots")

    report_dir = os.path.join(log_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    # gamma
    df_gamma = pd.DataFrame(statistics["gamma"]["gamma"]["all"])

    fig = plt.figure()
    plt.hist(df_gamma[0], bins=15, alpha=0.5)
    plt.title("Gamma Correction Distribution (Overall)")
    plt.xlabel("Gamma Correction")
    plt.ylabel("Count")
    plt.savefig(os.path.join(report_dir, "gamma_statistics.png"))
    plt.show()

    gam_electron = statistics["gamma"]["gamma"]["Electron"]
    gam_ion = statistics["gamma"]["gamma"]["Ion"]

    plt.hist(gam_electron, alpha=0.5, label="Electron")
    plt.hist(gam_ion, alpha=0.5, label="Ion")
    plt.legend(loc="best")
    plt.title("Gamma Correction Evaluation (Beams)")
    plt.xlabel("Gamma Corection")
    plt.ylabel("Count")
    plt.savefig(os.path.join(report_dir, "gamma_beams.png"))
    plt.show()

    # ml_statistics
    df = pd.DataFrame(statistics["ml"])
    df = df.rename(index={0: "true", 1: "total"})
    df_ml = df.T
    df_ml["false"] = df_ml["total"] - df_ml["true"]
    df_ml["percentage"] = df_ml["true"] / df_ml["total"] * 100

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

    # crosscorrelation
    df_cc = pd.DataFrame([statistics["crosscorrelation"]])
    df_cc["False"] = df_cc["Total"] - df_cc["True"]
    df_cc["percentage"] = df_cc["True"] / df_cc["Total"]

    ax = df_cc[["True", "False"]].plot.bar(title="CrossCorrelation Evaluation")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(report_dir, "cc_accuracy.png"))

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


def generate_exemplar_images(log_dir, statistics: dict) -> dict:
    from autoscript_sdb_microscope_client.structures import AdornedImage
    import matplotlib.pyplot as plt

    from PIL import Image
    exemplar_filenames = ["ref_lamella_low_res_eb", "ref_trench_high_res_ib", "jcut_highres_ib",
                            "needle_liftout_landed_highres_ib", "landing_lamella_final_cut_highres_ib", "sharpen_needle_final_ib",
                            "thin_lamella_stage_2_ib", "polish_lamella_final_ib"]

    # save png versions of exemplar images for report
    exemplar_image_dict = {}

    for sp_id in statistics["stage"]:
        exemplar_image_dict[sp_id] = []

        for img_basename in exemplar_filenames:
            
            fname = os.path.join(log_dir, sp_id, f"{img_basename}.tif")
            if os.path.exists(fname):
                fname_png = os.path.join(log_dir, sp_id, f"{img_basename}.png")

                # save png version
                adorned_img = AdornedImage.load(fname)
                img = Image.fromarray(adorned_img.data)
                img.save(fname_png)

                # add to dictionary        
                exemplar_image_dict[sp_id].append(fname_png)
    
    return exemplar_image_dict




def generate_report(log_dir, statistics: dict, exemplar_image_dict: dict) -> str: 
    report_dir = os.path.join(log_dir, "report")
    page_title_text='AutoLiftout Report'
    title_text = f"AutoLiftout Report"
    text = f"AutoLiftout report for run: {log_dir}"
    gamma_text = "Gamma Correction Statistics"
    ml_stats_text = "Machine Learning Statistics"
    stage_duration_text = "Stage Duration Statistics"
    clicks_text = "User Clicks"
    crosscorrelation_text = "CrossCorrelation Statistics"

    # stage duration images
    duration_image_dict = {}
    for sp_id in statistics["stage"]:
        duration_image_dict[sp_id] = os.path.join(report_dir, f"{sp_id}_duration.png")



    stage_duration_html = f"""
    <div id="stage_duration_section">
        <h2> Sample Position Data </h2>
    """
    for sp_id in statistics["stage"]:

        # only add full batches?
        if len(exemplar_image_dict[sp_id]) ==8:

            sp_id_html = f"""<h2>Sample ID ({sp_id})</h2>
                <div id="stage_duration_{sp_id}">
                    <img src='{duration_image_dict[sp_id]}' width="700">
                    </div>
                <div class="gallery" id="img_{sp_id}">
                    <img src='{exemplar_image_dict[sp_id][0]}' width="150">
                    <img src='{exemplar_image_dict[sp_id][1]}' width="150">
                    <img src='{exemplar_image_dict[sp_id][2]}' width="150">
                    <img src='{exemplar_image_dict[sp_id][3]}' width="150">
                    <img src='{exemplar_image_dict[sp_id][4]}' width="150">
                    <img src='{exemplar_image_dict[sp_id][5]}' width="150">
                    <img src='{exemplar_image_dict[sp_id][6]}' width="150">
                    <img src='{exemplar_image_dict[sp_id][7]}' width="150">
                </div> """

            stage_duration_html+= sp_id_html

    stage_duration_html+= f"</div>"


    # metadata
    import json

    with open(os.path.join(log_dir, "metadata.json")) as f:
        metadata = json.load(f)

    metadata_html = f"""<h2 style="text-align:left">Metadata </h2> <p > {metadata} </>"""

    from liftout import tools

    # generate report html
    html = f"""
            <html>
                <head>
                    <title>{page_title_text}</title>
                    <link rel="stylesheet" href='{os.path.join(os.path.dirname(tools.__file__), "style.css")}'>

                </head>
                <body>
                    <h1 style=>{title_text}</h1>
                    <p style="text-align:center">{text}</p>
                    
                    <div id="stats">
                        
                        <h2>{clicks_text}</h2>
                        <div id="click">
                            <img src='{os.path.join(report_dir, "clicks.png")}' width="700">
                        </div>
                        
                        <h2>{gamma_text}</h2>
                        <div id="gamma">
                            <img src='{os.path.join(report_dir, "gamma_statistics.png")}' width="700">
                            <img src='{os.path.join(report_dir, "gamma_beams.png")}' width="700">
                        </div>
            
                        <h2>{ml_stats_text}</h2>
                        <div id="ml">
                            <img src='{os.path.join(report_dir, "ml_statistics.png")}' width="700">
                            <img src='{os.path.join(report_dir, "ml_accuracy.png")}' width="700">
                        </div>

                        <h2>{crosscorrelation_text}</h2>
                        <div id="ml">
                            <img src='{os.path.join(report_dir, "cc_accuracy.png")}' width="700">
                        </div>
                        
                        {stage_duration_html}

                        {metadata_html}
                        
                    </div>
                                    
                </body>
            </html>
            """
    # 3. Write the html string as an HTML file
    with open(os.path.join(log_dir, "html_report.html"), "w") as f:
        f.write(html)

    print("HTML Report Generated")



def generate_html_report(log_dir):
    """Generate a HTML report summarising the run"""

    statistics = parse_log_file(log_dir)

    report_statistics = generate_report_data(statistics, log_dir, show=False)

    exemplar_image_dict = generate_exemplar_images(log_dir, statistics) 

    generate_report(log_dir=log_dir, statistics=statistics, exemplar_image_dict=exemplar_image_dict)



import pandas as pd
def calculate_aggregate_statistics(log_directories):
    """_summary_

    Args:
        log_directories (_type_): _description_

    Returns:
        dict: dictionary containing the aggregated run statistics dataframes
    """

    statistics_agg = {"click": None, "ml": None, "stage": None}
    df_click = None
    df_ml_full = None
    df_stage_full = None




    for log_dir in log_directories:
        
        logfile = os.path.join(log_dir, "logfile.log")
        exp_name = os.path.basename(log_dir)
        ts = exp_name.split("_")[-1]
        print("Experiment Name: ", exp_name)

        statistics = parse_log_file(log_dir)

        # clicks
        tmp_df_click = pd.DataFrame([statistics["click"]])
        tmp_df_click["exp_name"] = exp_name
        
        if df_click is None:
            df_click = tmp_df_click
        else:
            df_click = df_click.append(tmp_df_click)
        
        
        # ml
        df = pd.DataFrame(statistics["ml"])
        df = df.rename(index={0: "true", 1: "total"})
        df_ml = df.T
        df_ml["false"] = df_ml["total"] - df_ml["true"]
        df_ml["percentage"] = df_ml["true"] / df_ml["total"] * 100
        df_ml["exp_name"] = exp_name
        
        
        if df_ml_full is None:
            df_ml_full = df_ml
        else:
            df_ml_full = df_ml_full.append(df_ml)
        
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
    

            ts = exp_name.split("_")[-1]

            df_stage["exp_name"] = exp_name
            df_stage["sp_id"] = sp_id
            df_stage["timestamp"] = ts
            if df_stage_full is None:
                df_stage_full = df_stage
            else:
                df_stage_full = df_stage_full.append(df_stage)
            

    statistics_agg["click"] = df_click
    statistics_agg["ml"] = df_ml_full
    statistics_agg["stage"] = df_stage_full

    return statistics_agg