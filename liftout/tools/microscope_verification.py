#!/usr/bin/env python3


from liftout.tests.mock_autoscript_sdb_microscope_client import ManipulatorPosition
from liftout.fibsem import movement
from liftout.fibsem import acquire
from liftout.fibsem import utils

from autoscript_sdb_microscope_client.structures import *

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config("Microscope Verification", layout="wide", page_icon=":microscope:")

logging.basicConfig(level=logging.INFO)


def initial_test_setup():
    # setup
    st.write("Running Setup")

    # init microscope
    microscope = utils.initialise_fibsem()

    # read protocol settings
    settings = utils.load_config("../protocol_liftout.yml")

    # move to start position
    movement.move_to_sample_grid(microscope=microscope, settings = settings)
    st.success("Microscope Setup Complete")

    return microscope, settings


# TODO: setup the target coordinates to compare too

def record_stage_position(microscope, df, type, idx):
    """Format as dict, and append to dataframe"""
    stage_position = microscope.specimen.stage.current_position

    position_dict = {
        "timestamp": datetime.now().strftime("%Y%m%d.%H%M%S"),
        "x": stage_position.x,
        "y": stage_position.y,
        "z": stage_position.z,
        "r": stage_position.r,
        "t": stage_position.t,
        "type": type,
        "iter": idx,
    }

    df = df.append(pd.DataFrame.from_records([position_dict]))
    return df


def record_needle_position(microscope, df, type, idx):
    """Format as dict, and append to dataframe"""
    needle_position = microscope.specimen.manipulator.position


    position_dict = {
        "timestamp": datetime.now().strftime("%Y%m%d.%H%M%S"),
        "x": needle_position.x,
        "y": needle_position.y,
        "z": needle_position.z,
        "r": needle_position.r,
        "type": type,
        "iter": idx,
    }

    df = df.append(pd.DataFrame.from_records([position_dict]))
    return df

def rotation_test(microscope, settings, test_name="Rotation"):
    # repeat movements

    df = pd.DataFrame(columns=["timestamp", "x", "y", "z", "r", "t", "type", "img"])

    NUM_REPEATS = 10
    types = ["FLAT_TO_ELECTRON", "FLAT_TO_ION"]

    # get the initial coordinates: the stage shouldnt move only rotate and tilt...

    st.write(f"Running {test_name} test for {NUM_REPEATS} cycles.")
    progress_bar = st.progress(0)

    for i in range(NUM_REPEATS):

        progress_bar.progress((i + 1) / NUM_REPEATS)

        movement.flat_to_beam(microscope=microscope, settings=settings, beam_type=acquire.BeamType.ELECTRON)
        stage_position = microscope.stage.current_position

        df = record_stage_position(microscope, df, type=types[0], idx=i)

        movement.flat_to_beam(microscope=microscope, settings=settings, beam_type=acquire.BeamType.ION)
        stage_position = microscope.stage.current_position

        df = record_stage_position(microscope, df, type=types[1], idx=i)

        time.sleep(1)

    # save csv
    df.to_csv("rotation._data.csv")
    return df

def movement_test(microscope, settings, test_name="movement", NUM_REPEATS=10):

    df = pd.DataFrame(columns=["x", "y", "z", "r", "t", "type", "img"])

    types = ["STAGE_OUT", "SAMPLE_GRID", "LANDING_GRID"]

    # get the initial coordinates: the stage shouldnt move only rotate and tilt...

    st.write(f"Running {test_name} test for {NUM_REPEATS} cycles.")
    progress_bar = st.progress(0)

    for i in range(NUM_REPEATS):

        progress_bar.progress((i + 1) / NUM_REPEATS)

        # move sample stage out
        movement.move_sample_stage_out()
        stage_position = microscope.stage.current_position

        df = record_stage_position(microscope, df, type=types[0], idx=i)

        # move to sample grid
        movement.move_to_sample_grid(microscope=microscope, settings = settings)
        stage_position = microscope.stage.current_position


        df = record_stage_position(microscope, df, type=types[1], idx=i)

        # move to landing grid
        movement.move_to_landing_grid(microscope=microscope, settings=settings)
        stage_position = microscope.stage.current_position

        df = record_stage_position(microscope, df, type=types[2], idx=i)

        time.sleep(1)

    # save data
    df.to_csv("movement_data.csv")

    return df


def needle_movement_test(microscope, settings,
    test_name="Needle Movement", NUM_REPEATS=10
):

    df = pd.DataFrame(columns=["x", "y", "z", "r", "t", "type", "img"])

    types = ["PARKING_ONE", "SAFE_POSITION", "PARKING_TWO", "RETRACT"]

    # get the initial coordinates: the stage shouldnt move only rotate and tilt...

    st.write("Moving stage out for needle testing")
    movement.move_sample_stage_out(microscope=microscope, settings=settings) # setup

    st.write(f"Running {test_name} test for {NUM_REPEATS} cycles.")
    progress_bar = st.progress(0)

    for i in range(NUM_REPEATS):

        progress_bar.progress((i + 1) / NUM_REPEATS)

        # move needle in
        park_position = movement.insert_needle(microscope)
        needle_position = microscope.specimen.manipulator.current_position

        df = record_needle_position(microscope, df, type=types[0], idx=i)

        # move needle to safe location
        safe_position = ManipulatorPosition()
        microscope.specimen.manipulator.absolute_move(safe_position) # TODO: get safe absolute coordinates
        needle_position = microscope.specimen.manipulator.position

        df = record_needle_position(microscope, df, type=types[1], idx=i)

        # move back to park position
        microscope.specimen.manipulator.absolute_move(park_position)
        needle_position = microscope.specimen.manipulator.position

        df = record_needle_position(microscope, df, type=types[2], idx=i)
        
        # retract needle
        microscope.specimen.manipulator.retract()
        needle_position = microscope.specimen.manipulator.current_position

        df = record_needle_position(microscope, df, type=types[3], idx=i)

        time.sleep(1)

    df.to_csv("needle_movement_data.csv")

    return df


def autocontrast_test(microscope, settings, TEST_NAME="AutoContrast"):

    autocontrast_imgs = []
    raw_imgs = []
    hfws = [400e-6, 150e-6, 100e-6, 80e-6]

    image_settings = {'resolution': "1536x1024", 'dwell_time': 1e-6,
                            'hfw': 150-6, 'brightness': None,
                            'contrast': None, 'autocontrast': True,
                            'save': False, 'label': f"rotation_test",
                            # 'beam_type': BeamType.ELECTRON,
                            'save_path': "."}


    st.write(f"Running {TEST_NAME} test on {len(hfws)} horizontal field widths. {hfws}")

    # take reference images with autocontrast
    st.write("Taking AutoContrast Images")
    auto_progress_bar = st.progress(0)

    for i, hfw in enumerate(hfws, 1):
        
        auto_progress_bar.progress(i / len(hfws))
        eb_image, ib_image= acquire.take_reference_images(microscope=microscope, settings=image_settings)
        # eb_image, ib_image = np.array(np.random.randint(0, 255, size=(1024, 1536, 3))), np.array(np.random.randint(0, 255, size=(1024, 1536, 3)))
        
        autocontrast_imgs.append((eb_image, ib_image))
        time.sleep(0.5)

    # take reference images without autocontrast

    st.write("Taking Fixed Images")
    raw_progress_bar = st.progress(0)

    for i, hfw in enumerate(hfws, 1):
        raw_progress_bar.progress(i / len(hfws))
        image_settings["autocontrast"] = False
        image_settings["brightness"] = settings["machine_learning"]["brightness"]
        image_settings["contrast"] = settings["machine_learning"]["contrast"]
        image_settings["hfw"] = hfw

        eb_image, ib_image= acquire.take_reference_images(microscope=microscope, settings=image_settings)
        # eb_image, ib_image = np.array(np.random.randint(0, 255, size=(1024, 1536, 3))), np.array(np.random.randint(0, 255, size=(1024, 1536, 3)))
        
        raw_imgs.append((eb_image, ib_image))
        time.sleep(0.5)


    return autocontrast_imgs, raw_imgs, hfws


def main():

    df = pd.DataFrame()
    auto_imgs = None

    st.header("Microscope Verification")

    # initial setup
    microscope, settings = initial_test_setup()

    st.write(settings)

    buttons_cols = st.columns(4)

    if buttons_cols[0].button("Run Rotation Test"):

        df = rotation_test(microscope, settings)
        st.success("Rotation Test Finished")

    if buttons_cols[1].button("Run Movement Test"):

        df = movement_test(microscope, settings)
        st.success("Movement Test Finished")

    if buttons_cols[2].button("Run Needle Movement Test"):

        df = needle_movement_test(microscope, settings)
        st.success("Needle Movement Test Finished")

    if buttons_cols[3].button("Run AutoContrast Test"):

        auto_imgs, raw_imgs, hfws = autocontrast_test(microscope, settings)
        st.success("AutoContrast Test Finished")

    # display results and charts
    if not df.empty:

        cols = st.columns(2)

        cols[0].write("Rotation Results")
        cols[0].write(df)

        cols[1].write("Summary Statistics ")
        cols[1].write(df.describe())

        plot_cols = st.columns(2)
        fig_2d = px.scatter(df, x="r", y="t", color="type", title="Rotation and Tilt")
        fig_3d = px.scatter_3d(
            df, x="x", y="y", z="z", color="type", title="Position (x, y, z)"
        )

        plot_cols[0].plotly_chart(fig_2d, use_container_width=True)
        plot_cols[1].plotly_chart(fig_3d, use_container_width=True)


    if auto_imgs:

        img_cols = st.columns(2)
        for i in range(len(auto_imgs)):
            
            img_cols[0].subheader(f"AutoContrast at HFW: {hfws[i]:e}")
            img_cols[1].subheader(f"Fixed at HFW: {hfws[i]:e}")
            eb_image, ib_image = auto_imgs[i]
            img_cols[0].image(eb_image, caption=f"AutoContrast Electron Beam Image at {hfws[i]:e} hfw")
            img_cols[0].image(ib_image, caption=f"AutoContrast Ion Beam Image at {hfws[i]:e} hfw")

            eb_image, ib_image = raw_imgs[i]
            img_cols[1].image(eb_image, caption=f"Fixed Electron Beam Image at {hfws[i]:e} hfw")
            img_cols[1].image(ib_image, caption=f"Fixed Ion Beam Image at {hfws[i]:e} hfw")


# TODO: filter results            

if __name__ == "__main__":
    main()
