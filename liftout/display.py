import logging
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi


def quick_plot(image, median_smoothig=3):
    """Display image with matplotlib.pyplot

    Parameters
    ----------
    image : Adorned image or numpy array
        Input image.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    display_image = image.data
    if median_smoothig is not None:
        display_image = ndi.median_filter(display_image, size=median_smoothig)
    height, width = display_image.shape
    try:
        pixelsize_x = image.metadata.binary_result.pixel_size.x
        pixelsize_y = image.metadata.binary_result.pixel_size.y
    except AttributeError:
        extent_kwargs = [-(width / 2), +(width / 2), -(height / 2), +(height / 2)]
        ax.set_xlabel("Distance from origin (pixels)")
    else:
        extent_kwargs = [
            -(width / 2) * pixelsize_x,
            +(width / 2) * pixelsize_x,
            -(height / 2) * pixelsize_y,
            +(height / 2) * pixelsize_y,
        ]
        ax.set_xlabel(
            "Distance from origin (meters) \n" "1 pixel = {} meters".format(pixelsize_x)
        )
    ax.set_xlim(extent_kwargs[0], extent_kwargs[1])
    ax.set_ylim(extent_kwargs[2], extent_kwargs[3])
    ax.imshow(display_image, cmap="gray", extent=extent_kwargs)
    return fig, ax


def plot(image, median_smoothig=3):
    fig, ax = quick_plot(image, median_smoothig=3)
    fig.show()
    return fig, ax


def select_point(image):
    """Return location of interactive user click on image.

    Parameters
    ----------
    image : AdornedImage or 2D numpy array.

    Returns
    -------
    coords
          Coordinates of last point clicked in the image.
          Coordinates are in x, y format.
          Units are the same as the matplotlib figure axes.
    """
    fig, ax = quick_plot(image)
    coords = []

    def on_click(event):
        print(event.xdata, event.ydata)
        coords.append(event.ydata)
        coords.append(event.xdata)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return np.flip(coords[-2:], axis=0)  # coordintes in x, y format
