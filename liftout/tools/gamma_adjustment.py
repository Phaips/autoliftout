

import streamlit as st
import glob
import PIL
import matplotlib.pyplot as plt
import numpy as np

import skimage
from skimage import exposure
import os
# timestamps = glob.glob(r"C:Users\Admin\Github\autoliftout\liftout\log\run/*/".replace('\\', "/r"), recursive=True)
# st.write(timestamps)
# timestamp = st.selectbox(timestamps)
# st.write(timestamp)
path = r"C:\Users\Admin\Github\autoliftout\liftout\log\run\20210825.133912\img\*.tif"


filenames = glob.glob(path)

fname = st.selectbox("select image", [f.split("/")[-1] for f in filenames])

img = np.array(PIL.Image.open(fname))

gam = st.slider("gamma", 0.0, 5.0, 2.0)

gamma_corrected = exposure.adjust_gamma(img, gam)

cols = st.columns(2)

cols[0].image(img, caption="raw")
cols[1].image(gamma_corrected, caption="gamma correction")



bins = 30
bin_counts, bin_edges = np.histogram(img, bins)
fig = plt.figure()
plt.hist(img.ravel(), bins, color="blue", label="raw", alpha=0.5)
plt.hist(gamma_corrected.ravel(), bins, color="red", label="gamma_adjusted", alpha=0.5)
plt.legend(loc="best")
st.pyplot(fig)

# bins = 30
# # bin_counts, bin_edges = np.histogram(gamma_corrected, bins)
# fig2 = plt.figure()
# plt.hist(gamma_corrected.ravel(), bins)
# plt.ylim([0, top_y])
# cols[1].pyplot(fig2)


# TODO: fit the histogram for gamma adjustment to a set of 'good' images