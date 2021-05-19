#!/usr/bin/env python3

import pandas as pd
import glob

from datetime import datetime

# from evaluation import cached_streamlit_setup
from utils import *

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


from utils import match_filenames_from_path, load_model, detect_and_draw_lamella_and_needle, show_overlay
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", action="store", type=int, dest="n")
    args = parser.parse_args()

    n_images = args.n

    filenames = match_filenames_from_path("data/train/raw/*")
    print("\nStarting Data Engine..")
    print(f"Analysing {len(filenames)} files.")

    try:
        df_data = pd.read_csv("data.csv", index_col=0)
            
        correct_filenames = df_data[df_data["correct"]=="y"]["fname"]
        incorrect_filenames = df_data[df_data["correct"]=="n"]["fname"]

        # 
        classified_filenames = list(correct_filenames.values) + list(incorrect_filenames.values)
        unclassified_filenames = [fname for fname in filenames if fname not in classified_filenames] 
        
        # info
        print(f"\n{len(correct_filenames)} / {len(classified_filenames)} correct.")
        print(f"{len(unclassified_filenames)} remaining to be classified.")
    
    except:
        pass

    # pre-load
    db = dict()
    model = load_model("models/12_04_2021_10_32_23_model.pt")

    # loop through each image
    for i, fname in enumerate(unclassified_filenames[:n_images]):
        
        correct = None
        
        print(f"\nFile: {fname}")

        # model inference + display
        img, rgb_mask = model_inference(model, fname)

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
        plt.imshow(img_overlay)
        plt.show()

        while correct not in ["y", "n"]:
            
            if correct is not None:
                print(f"{correct} is not a valid input.")

            correct = input("Is the prediction correct? (y/n)").lower()

        print("Answer is :", correct)
        db[i] = {}
        db[i]["fname"] = fname
        db[i]["correct"] = correct

        print(f"Writing {fname} to log file.\n")


    # convert to dataframe
    df = pd.DataFrame.from_dict(db, columns=["fname", "correct"], orient="index")
    csv_filename = "data.csv"
    
    # concat new and existing and save as csv
    df_data = pd.concat([df_data, df], ignore_index=True)
    print(f"Adding {len(df)} filenames to {csv_filename}")
    df_data.to_csv(csv_filename)
    


# Streamlit Data engine

# cols_button = st.beta_columns(2)
# yes = cols_button[0].button("Yes")
# no = cols_button[1].button("No")

# if yes:    
#     st.write("Yes: ", filenames[0])
# if no:    
#     st.write("No: ", filenames[0])