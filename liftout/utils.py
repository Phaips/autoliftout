import os
from pathlib import Path

import numpy as np
from fibsem.structures import ImageSettings
from fibsem.utils import configure_logging


def make_logging_directory(path: Path = None, name="run"):
    if path is None:
        from liftout.config import config
        path = os.path.join(config.BASE_PATH, "log")
    directory = os.path.join(path, name)
    os.makedirs(directory, exist_ok=True)
    return directory

def get_last_log_message(path: Path) -> str:
    with open(path) as f:
        lines = f.read().splitlines()
        log_line = lines[-1:][-1]  # last log msg
        log_msg = log_line.split("—")[-1].strip()

    return log_msg


def plot_two_images(img1, img2) -> None:
    import matplotlib.pyplot as plt
    from fibsem.structures import Point

    c = Point(img1.data.shape[1] // 2, img1.data.shape[0] // 2)

    fig, ax = plt.subplots(1, 2, figsize=(30, 30))
    ax[0].imshow(img1.data, cmap="gray")
    ax[0].plot(c.x, c.y, "y+", ms=50, markeredgewidth=2)
    ax[1].imshow(img2.data, cmap="gray")
    ax[1].plot(c.x, c.y, "y+", ms=50, markeredgewidth=2)
    plt.show()


def take_reference_images_and_plot(microscope, image_settings: ImageSettings):
    from pprint import pprint

    from fibsem import acquire

    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    plot_two_images(eb_image, ib_image)

    return eb_image, ib_image


# cross correlate
def crosscorrelate_and_plot(
    ref_image,
    new_image,
    rotate: bool = False,
    lp: int = 128,
    hp: int = 8,
    sigma: int = 6,
    ref_mask: np.ndarray = None,
    xcorr_limit: int = None
):
    import matplotlib.pyplot as plt
    import numpy as np
    from fibsem import alignment
    from fibsem.structures import Point
    from fibsem.imaging import utils as image_utils

    # rotate ref
    if rotate:
        ref_image = image_utils.rotate_image(ref_image)

    dx, dy, xcorr = alignment.shift_from_crosscorrelation(
        ref_image,
        new_image,
        lowpass=lp,
        highpass=hp,
        sigma=sigma,
        use_rect_mask=True,
        ref_mask=ref_mask,
        xcorr_limit=xcorr_limit
    )

    pixelsize = ref_image.metadata.binary_result.pixel_size.x
    dx_p, dy_p = int(dx / pixelsize), int(dy / pixelsize)

    print(f"shift_m: {dx}, {dy}")
    print(f"shift_px: {dx_p}, {dy_p}")

    shift = np.roll(new_image.data, (-dy_p, -dx_p), axis=(0, 1))

    mid = Point(shift.shape[1] // 2, shift.shape[0] // 2)

    if ref_mask is None:
        ref_mask = np.ones_like(ref_image.data)

    fig, ax = plt.subplots(1, 4, figsize=(30, 30))
    ax[0].imshow(ref_image.data * ref_mask , cmap="gray")
    ax[0].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[0].set_title(f"Reference (rotate={rotate})")
    ax[1].imshow(new_image.data, cmap="gray")
    ax[1].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[1].set_title(f"New Image")
    ax[2].imshow(xcorr, cmap="turbo")
    ax[2].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[2].plot(mid.x - dx_p, mid.y - dy_p, "m+", ms=50, markeredgewidth=2)
    ax[2].set_title("XCORR")
    ax[3].imshow(shift, cmap="gray")
    ax[3].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2, label="new_position")
    ax[3].plot(mid.x - dx_p, mid.y - dy_p, "m+", ms=50, markeredgewidth=2, label="old_position")
    ax[3].set_title("New Image Shifted")
    ax[3].legend()
    plt.show()

    return dx, dy, xcorr


def plot_crosscorrelation(ref_image, new_image, dx, dy, xcorr):
    import matplotlib.pyplot as plt
    from fibsem.structures import Point

    pixelsize = ref_image.metadata.binary_result.pixel_size.x
    dx_p, dy_p = int(dx / pixelsize), int(dy / pixelsize)

    print(f"shift_m: {dx}, {dy}")
    print(f"shift_px: {dx_p}, {dy_p}")

    shift = np.roll(new_image.data, (-dy_p, -dx_p), axis=(0, 1))

    mid = Point(shift.shape[1] // 2, shift.shape[0] // 2)

    fig, ax = plt.subplots(1, 4, figsize=(30, 30))
    ax[0].imshow(ref_image.data, cmap="gray")
    ax[0].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[0].set_title(f"Reference)")
    ax[1].imshow(new_image.data, cmap="gray")
    ax[1].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[1].set_title(f"New Image")
    ax[2].imshow(xcorr, cmap="turbo")
    ax[2].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[2].plot(mid.x - dx_p, mid.y - dy_p, "m+", ms=50, markeredgewidth=2)
    ax[2].set_title("XCORR")
    ax[3].imshow(shift, cmap="gray")
    ax[3].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
    ax[3].plot(mid.x - dx_p, mid.y - dy_p, "m+", ms=50, markeredgewidth=2)
    ax[3].set_title("New Image Shifted")
    plt.show()



### VALIDATION

def _validate_model_weights_file(filename):
    import os

    from liftout.model import models

    weights_path = os.path.join(os.path.dirname(models.__file__), filename)
    if not os.path.exists(weights_path):
        raise ValueError(f"Unable to find model weights file {weights_path} specified.")


### SETUP
# TODO: remove in favour of setup_session
def quick_setup():
    """Quick setup for microscope, settings, and image_settings"""

    from fibsem import utils as fibsem_utils
    from liftout.config import config
    settings = fibsem_utils.load_settings_from_config(
        config_path = config.config_path,
        protocol_path= config.protocol_path
    )

    import os

    path = os.path.join(os.getcwd(), "tools/test")
    os.makedirs(path, exist_ok=True)
    configure_logging(path)
    settings.image.save_path = path

    microscope = fibsem_utils.connect_to_microscope(
        ip_address=settings.system.ip_address
    )
    return microscope, settings


def full_setup():
    """Quick setup for microscope, settings,  image_settings, sample and lamella"""
    import os

    from liftout.structures import Lamella, Sample

    microscope, settings = quick_setup()

    # sample
    sample = Sample(path=os.path.dirname(settings.image.save_path), name="test")

    # lamella
    lamella = Lamella(sample.path, 999, _petname="999-test-mule")
    sample.positions[lamella._number] = lamella
    sample.save()

    return microscope, settings, sample, lamella


