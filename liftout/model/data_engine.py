#!/usr/bin/env python3

import argparse
import glob
from datetime import datetime

import pandas as pd
from matplotlib.widgets import RectangleSelector

from utils import *
from utils import (detect_and_draw_lamella_and_needle, load_image_from_file,
                   load_model, match_filenames_from_path, show_overlay)


def create_empty_dataset():

    """ create empty correct prediction dataset """

    # load image filenames
    images_path = "data/train/raw/*"
    filenames = sorted(glob.glob(images_path + ".tif"))

    # create dataframe
    df = pd.DataFrame(columns=["fname", "correct"])
    df.fname = filenames
    df.correct = False

    # datetime string (now)
    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    # save df as csv
    csv_fname = f"correct_{dt_string}.csv"
    df.to_csv(csv_fname)
    print(f"Successfully saved {len(df)} rows in {csv_fname}")



def relabel_img(img):

    fig, ax = plt.subplots()
    img_plot = ax.imshow(img, cmap="gray")

    click = [None, None]
    release = [None, None]


    def line_select_callback(eclick, erelease):
        click[:] = eclick.xdata, eclick.ydata
        release[:] = erelease.xdata, erelease.ydata

        x1, y1 = click
        x2, y2 = release

        rect = plt.Rectangle(
            (min(x1, x2), min(y1, y2)),
            np.abs(x1 - x2),
            np.abs(y1 - y2),
            color="red",
            alpha=0.4,
        )
        ax.add_patch(rect)

    rs = RectangleSelector(
        ax,
        line_select_callback,
        drawtype="box",
        useblit=False,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
        rectprops=dict(color="red", alpha=0.4, fill=True),
    )

    plt.show()

    label_json = {"shapes": [{"label": "lamella", "points": [click, release]}]}

    print("Label: ", label_json)

    return label_json






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", action="store", type=int, dest="n", default=1000)
    parser.add_argument("-m", action="store", type=str, dest="model", default="models/fresh_full_n10.pt")
    parser.add_argument("-d", action="store", type=str, dest="data", default="data/train/raw")
    args = parser.parse_args()

    # parse cmd line arguments
    n_images = args.n
    data_path = args.data 
    model_path = args.model


    filenames = match_filenames_from_path(data_path + "/*")
    print("\nStarting Data Engine..")
    print(f"Analysing {len(filenames)} files.")
    print(f"Using {model_path} as model.")
    

    try:

        df_data = pd.read_csv("data.csv", index_col=0)
        print(f"Existing data found...")
            
        correct_filenames = df_data[df_data["correct"]=="y"]["fname"]
        incorrect_filenames = df_data[df_data["correct"]=="n"]["fname"]

        # 
        classified_filenames = list(correct_filenames.values) + list(incorrect_filenames.values)
        unclassified_filenames = [fname for fname in filenames if fname not in classified_filenames] 
        
        # info
        print(f"\n{len(correct_filenames)} / {len(classified_filenames)} correct.")
        print(f"{len(unclassified_filenames)} remaining to be classified.")
    
    except:
        unclassified_filenames = filenames
        df_data = pd.DataFrame()



    # pre-load
    db = dict()
    model = load_model(model_path)

    # loop through each image
    for i, fname in enumerate(unclassified_filenames[:n_images]):
        
        correct = None
        
        print(f"\nFile: {fname}")

        # model inference + display
        img = load_image_from_file(fname)

        img, rgb_mask = model_inference(model, img)

        # detect and draw lamella centre, and needle tip
        (
            lamella_centre_px,
            rgb_mask_lamella,
            needle_tip_px,
            rgb_mask_needle,
            rgb_mask_combined,
        ) = detect_and_draw_lamella_and_needle(rgb_mask)

        # show prediction overlay
        img_overlay = show_overlay(img, rgb_mask_combined)

        # display image
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img, cmap="gray", alpha=1)
        ax.imshow(img_overlay, alpha=0.4)
        plt.show()

        while correct not in ["y", "n", "exit"]:
            
            if correct is not None:
                print(f"{correct} is not a valid input.")

            correct = input("Is the prediction correct (y/n)?: ").lower()
        
        if correct == "exit":
            break


        if correct == "n":
            label_json = relabel_img(img)

        print("Answer is :", correct)
        db[i] = {}
        db[i]["fname"] = fname
        db[i]["correct"] = correct
        db[i]["model"] = model_path

        print(f"Writing {fname} to log file.\n")

    # convert to dataframe
    df = pd.DataFrame.from_dict(db, columns=["fname", "correct", "model"], orient="index")
    csv_filename = "data.csv"
    
    # concat new and existing and save as csv
    df_data = pd.concat([df_data, df], ignore_index=True)
    print(f"Adding {len(df)} filenames to {csv_filename}")
    df_data.to_csv(csv_filename)
    