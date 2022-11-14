import glob
import os
from copy import deepcopy

import liftout
import pandas as pd
import plotly.express as px
import streamlit as st
from autoscript_sdb_microscope_client.structures import AdornedImage
from liftout.structures import (AutoLiftoutState, Lamella, Sample,
                                load_experiment)
from liftout.tools.data import (AutoLiftoutStatistics,
                                calculate_statistics_dataframe,
                                create_history_dataframe)

BASE_PATH = os.path.dirname(liftout.__file__)
LOG_PATH = os.path.join(BASE_PATH, "log")


st.set_page_config(layout="wide")
st.title("AutoLiftout Companion")

#################### EXPERIMENT SECTION ####################

# select experiment
paths = glob.glob(os.path.join(LOG_PATH, "*Dm-*/"))
EXPERIMENT_PATH = st.selectbox(label="Experiment Path ", options=paths)
sample = load_experiment(EXPERIMENT_PATH)

stats = calculate_statistics_dataframe(EXPERIMENT_PATH)

df = stats.sample

# experiment metrics
n_lamella = len(df["petname"])
n_images =  len(glob.glob(os.path.join(EXPERIMENT_PATH, "**/**.tif"), recursive=True))
n_clicks = len(stats.click)
n_ml = len(stats.ml)


# ml accuracy
ml_group = stats.ml.groupby("correct").count()
ml_group = ml_group.reset_index()
n_ml_correct = ml_group[ml_group["correct"] == 'True']["feature"].values[0]
n_ml_incorrect = ml_group[ml_group["correct"] == "False"]["feature"].values[0]
ml_accuracy = n_ml_correct / n_ml

# st write ml_accuracy
# st.write(f"ML Accuracy: {ml_accuracy:.2f}")


df_gamma_mean = stats.gamma.groupby(by="beam_type").mean()
df_gamma_mean.reset_index(inplace=True)
df_gamma_mean = df_gamma_mean.rename({"index": "beam_type"})

eb_gamma = df_gamma_mean[df_gamma_mean["beam_type"] == "Electron"]['gamma'].item()
ib_gamma = df_gamma_mean[df_gamma_mean["beam_type"] == "Ion"]['gamma'].item()


cols = st.columns(5)
cols[0].metric("Lamella", n_lamella)
cols[1].metric("Images", n_images)
cols[2].metric("Clicks", n_clicks)
cols[3].metric("Detections", ml_accuracy)

st.markdown("---")

# statistics
# cols = st.columns(5)
# cols[1].metric("Gamma (Electron)", f"{eb_gamma:.3f}")
# cols[2].metric("Gamma (Ion)", f"{ib_gamma:.3f}")



# plots TODO
fig_gamma = px.histogram(stats.gamma, x="gamma", color="beam_type", nbins=30, title="Gamma Distribution")

cols = st.columns(2)
cols[0].plotly_chart(fig_gamma)
try:
    fig_clicks = px.scatter(stats.click, x="x", y="y", symbol="type", color="beam_type", title="Click Distribution")
    cols[1].plotly_chart(fig_clicks)
except:
    st.write("No Click available.")
# stats.move["size_z"] = abs(stats.move["z"] * 1e6).astype(int)
# fig = px.scatter(stats.move, x="x", y="y", size="size_z", color="beam_type", symbol="mode")
# fig = px.scatter_3d(stats.move, x="x", y="y", z="z", color="beam_type", symbol="mode")



##### Clicking

# TODO: add det to this stat, type of click

st.write(stats.click)
fig_clicks = px.scatter(stats.click, x="x", y="y", symbol="type", color="beam_type", title="Click Distribution")
fig_clicks_stage = px.bar(stats.click, x="type", color="stage", title="Clicks per Stage", barmode="group")

cols = st.columns(2)
cols[0].plotly_chart(fig_clicks)
cols[1].plotly_chart(fig_clicks_stage)




# positions plots
st.subheader("Position Data")
fig_lamella = px.scatter(df, x="lamella.x", y="lamella.y", text="petname", title="Lamella Positions")
fig_landing = px.scatter(df, x="landing.x", y="landing.y", text="petname", title="Landing Positions")

cols = st.columns(2)
cols[0].plotly_chart(fig_lamella)
cols[1].plotly_chart(fig_landing)

try:
    # ml plots
    st.subheader("ML Data")
    fig_ml = px.bar(stats.ml, x="feature", title="Detections", color="stage", barmode="group")
    fig_ml2 = px.bar(stats.ml, x="feature", title="Accuracy", color="correct", barmode="group")

    cols = st.columns(2)
    cols[0].plotly_chart(fig_ml)
    cols[1].plotly_chart(fig_ml2)
except:
    st.write("No ML Data available.")

# history, duration
st.markdown("""---""")
st.subheader("Stage History and Duration")

df_stage_history = stats.history

df_group_count  = df_stage_history.groupby(by="stage").count()
df_group_count.reset_index(inplace=True)
df_group_count = df_group_count.rename({"index": "stage"})
df_group_count["count"] = df_group_count["duration"] # what?

df_group_duration = df_stage_history.groupby(by=["stage"]).mean()
df_group_duration.reset_index(inplace=True)
df_group_duration = df_group_duration.rename(columns={"index": "stage"})

cols = st.columns(2)
fig_stage_count = px.bar(df_group_count, y="count", color="stage", title="Stage Count")
fig_stage_duration = px.bar(df_group_duration, y="duration", color="stage", title="Stage Duration")
cols[0].plotly_chart(fig_stage_count)
cols[1].plotly_chart(fig_stage_duration)


#################### INVIDIUAL LAMELLA SECTION ####################
st.markdown("""---""")
st.subheader("Lamella Data")

# select lamella
petnames = df["petname"].unique()
lamella_name = st.selectbox("Lamella", petnames)
LAMELLA_PATH = os.path.join(EXPERIMENT_PATH, lamella_name)

# select image
image_paths = glob.glob(os.path.join(LAMELLA_PATH, "*.tif*"))
images_filenames = [os.path.basename(path) for path in image_paths]

# info
current_stage = df[df["petname"] == lamella_name]["current_stage"].item()

cols = st.columns(3)
cols[0].metric("Stage", current_stage)
cols[1].metric("Images", len(images_filenames))

# lamella stats
df_lamella = df_stage_history[df_stage_history["petname"] == lamella_name]

df_lamella_duration = df_lamella.groupby(by=["stage"]).mean()
df_lamella_duration.reset_index(inplace=True)
df_lamella_duration = df_lamella_duration.rename(columns={"index": "stage"})
fig_duration = px.bar(df_lamella_duration, y="duration", color="stage", title=f"Stage Duration ({lamella_name})")

cols = st.columns(2)
cols[0].write(df_lamella)
cols[0].plotly_chart(fig_duration)

# images
fname = cols[1].selectbox("Select Image", images_filenames)
img = AdornedImage.load(os.path.join(LAMELLA_PATH, fname))
cols[1].image(img.data, caption=os.path.basename(fname))

