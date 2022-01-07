import os
from liftout.fibsem.movement import *
from autoscript_sdb_microscope_client.structures import GrabFrameSettings
import scipy.ndimage as ndi
from scipy import fftpack, misc
from PIL import Image, ImageDraw
from liftout.fibsem import acquire
from liftout.detection import detection
from liftout.model import models

BeamType = acquire.BeamType


def validate_scanning_rotation(microscope):
    """Ensure the scanning rotation is set to zero."""
    rotation = microscope.beams.ion_beam.scanning.rotation.value
    if rotation is None:
        microscope.beams.ion_beam.scanning.rotation.value = 0
        rotation = microscope.beams.ion_beam.scanning.rotation.value
    if not np.isclose(rotation, 0.0):
        raise ValueError(
            "Ion beam scanning rotation must be 0 degrees."
            "\nPlease change your system settings and try again."
            "\nCurrent rotation value is {}".format(rotation)
        )


def correct_stage_drift(
    microscope, image_settings, reference_images, liftout_counter, mode="eb"
):

    ref_eb_lowres, ref_eb_highres, ref_ib_lowres, ref_ib_highres = reference_images

    lowres_count = 1
    if mode == "eb":
        ref_lowres = ref_eb_lowres
        ref_highres = ref_eb_highres
    elif mode == "ib":
        lowres_count = 2
        ref_lowres = rotate_AdornedImage(ref_ib_lowres)
        ref_highres = rotate_AdornedImage(ref_ib_highres)
    elif mode == "land":
        lowres_count = 1
        ref_lowres = ref_ib_lowres
        ref_highres = ref_ib_highres

    stage = microscope.specimen.stage
    # TODO: user input resolution, must match (protocol)
    image_settings["resolution"] = "1536x1024"
    image_settings["dwell_time"] = 1e-6

    pixelsize_x_lowres = ref_lowres.metadata.binary_result.pixel_size.x
    field_width_lowres = pixelsize_x_lowres * ref_lowres.width
    pixelsize_x_highres = ref_highres.metadata.binary_result.pixel_size.x
    field_width_highres = pixelsize_x_highres * ref_highres.width
    microscope.beams.ion_beam.horizontal_field_width.value = field_width_lowres
    microscope.beams.electron_beam.horizontal_field_width.value = field_width_lowres

    # TODO: refactor this code
    # TODO: 'coarse alignment' status
    image_settings["hfw"] = field_width_lowres
    image_settings["save"] = True
    # image_settings['label'] = f'{liftout_counter:02d}_drift_correction_lamella_low_res'  # TODO: add to protocol

    if mode == "land":
        image_settings[
            "label"
        ] = f"{liftout_counter:02d}_{mode}_drift_correction_landing_low_res"  # TODO: add to protocol
        new_eb_lowres, new_ib_lowres = take_reference_images(
            microscope, settings=image_settings
        )
        ret = align_using_reference_images(ref_lowres, new_ib_lowres, stage, mode=mode)
        if ret is False:
            return ret

        # fine alignment
        image_settings["hfw"] = field_width_highres
        image_settings["save"] = True

        image_settings[
            "label"
        ] = f"{liftout_counter:02d}_drift_correction_landing_high_res"  # TODO: add to protocol
        new_eb_highres, new_ib_highres = take_reference_images(
            microscope, settings=image_settings
        )
        ret = align_using_reference_images(
            ref_highres, new_ib_highres, stage, mode=mode
        )
        # TODO: deduplicate this bit ^
    else:
        for i in range(lowres_count):

            image_settings[
                "label"
            ] = f"{liftout_counter:02d}_{mode}_drift_correction_lamella_low_res_{i}"  # TODO: add to protocol
            new_eb_lowres, new_ib_lowres = take_reference_images(
                microscope, settings=image_settings
            )
            ret = align_using_reference_images(ref_lowres, new_eb_lowres, stage)
            if ret is False:
                return ret

        # fine alignment
        image_settings["hfw"] = field_width_highres
        image_settings["save"] = True

        image_settings[
            "label"
        ] = f"{liftout_counter:02d}_drift_correction_lamella_high_res"  # TODO: add to protocol
        new_eb_highres, new_ib_highres = take_reference_images(
            microscope, settings=image_settings
        )
        ret = align_using_reference_images(ref_highres, new_eb_highres, stage)

    logging.info(f"calibration: image cross correlation finished {ret}")
    return ret


def align_using_reference_images(ref_image, new_image, stage, mode=None):

    # TODO: Read in from protocol
    # TODO: there are three different types of cross-corellation, E-E, E-I, I-I
    if mode == "land":
        beam_type = BeamType.ION
    if mode is None:
        beam_type = BeamType.ELECTRON
    lp_ratio = 3
    hp_ratio = 64
    sigma_factor = 10
    sigma_ratio = 1536

    # These are the old cross-correlation values
    # elif mode is not "land":
    #     lp_ratio = 12
    #     hp_ratio = 256
    #     sigma_factor = 2
    #     sigma_ratio = 1536
    #     beam_type = BeamType.ELECTRON
    logging.info(
        f"calibration: align using {beam_type.name} reference image in mode {mode}."
    )
    print("Hello")
    # TODO: possibly hard-code these numbers at fixed resolutions?
    lowpass_pixels = int(max(new_image.data.shape) * 0.66)   # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int(
        max(new_image.data.shape) / hp_ratio
    )  # =6 @ 1536x1024, good for e-beam images
    sigma = 6
    #         int(
    #     sigma_factor * max(new_image.data.shape) / sigma_ratio
    # )  # =2 @ 1536x1024, good for e-beam images
    # TODO: ask Sergey about maths/check if we can use pixel_to_real_value on dx
    dx_ei_meters, dy_ei_meters = shift_from_crosscorrelation_AdornedImages(
        new_image,
        ref_image,
        lowpass=lowpass_pixels,
        highpass=highpass_pixels,
        sigma=sigma,
    )

    # check if the cross correlation is trying to move more than a safety threshold.
    pixelsize_x = ref_image.metadata.binary_result.pixel_size.x
    hfw = pixelsize_x * ref_image.width
    vfw = pixelsize_x * ref_image.height
    X_THRESHOLD = 0.25 * hfw
    Y_THRESHOLD = 0.25 * vfw
    if abs(dx_ei_meters) > X_THRESHOLD or abs(dy_ei_meters) > Y_THRESHOLD:
        logging.warning("calibration: calculated cross correlation movement too large.")
        logging.warning("calibration: cancelling automatic cross correlation.")
        return False
    else:

        x_move = x_corrected_stage_movement(-dx_ei_meters)
        yz_move = y_corrected_stage_movement(
            dy_ei_meters, stage.current_position.t, beam_type=beam_type
        )  # check electron/ion movement
        stage.relative_move(x_move)
        stage.relative_move(yz_move)
        return True


# TODO: figure out a better name
def identify_shift_using_machine_learning(
    microscope, image_settings, settings, shift_type
):

    eb_image, ib_image = take_reference_images(microscope, image_settings)
    weights_file = settings["machine_learning"]["weights"]
    weights_path = os.path.join(os.path.dirname(models.__file__), weights_file)
    detector = detection.Detector(weights_path)

    if image_settings["beam_type"] == BeamType.ION:
        image = ib_image
    else:
        image = eb_image
    (
        image_w_overlay,
        downscaled_image,
        feature_1_px,
        feature_1_type,
        feature_2_px,
        feature_2_type,
    ) = detector.locate_shift_between_features(image, shift_type=shift_type, show=False)
    return (
        image,
        np.array(image_w_overlay),
        np.array(downscaled_image),
        feature_1_px,
        feature_1_type,
        feature_2_px,
        feature_2_type,
    )


def shift_from_correlation_electronBeam_and_ionBeam(
    eb_image, ib_image, lowpass=128, highpass=6, sigma=2
):
    ib_image_rotated = rotate_AdornedImage(ib_image)
    x_shift, y_shift = shift_from_crosscorrelation_AdornedImages(
        eb_image, ib_image_rotated, lowpass=lowpass, highpass=highpass, sigma=sigma
    )
    return x_shift, y_shift


def rotate_AdornedImage(image):
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        GrabFrameSettings,
    )

    data = np.rot90(np.rot90(np.copy(image.data)))
    reference = AdornedImage(data=data)
    reference.metadata = image.metadata
    return reference


def shift_from_crosscorrelation_AdornedImages(
    img1: AdornedImage, img2: AdornedImage, lowpass: int =128, highpass: int =6, sigma: int =6, use_rect_mask: bool =False
):
    pixelsize_x_1 = img1.metadata.binary_result.pixel_size.x
    pixelsize_y_1 = img1.metadata.binary_result.pixel_size.y
    pixelsize_x_2 = img2.metadata.binary_result.pixel_size.x
    pixelsize_y_2 = img2.metadata.binary_result.pixel_size.y
    # normalise both images
    img1_data_norm = (img1.data - np.mean(img1.data)) / np.std(img1.data)
    img2_data_norm = (img2.data - np.mean(img2.data)) / np.std(img2.data)
    # cross-correlate normalised images
    if use_rect_mask:
        rect_mask = _mask_rectangular(img2_data_norm.shape)
        img1_data_norm = rect_mask * img1_data_norm
        img2_data_norm = rect_mask * img2_data_norm
    xcorr = crosscorrelation(
        img1_data_norm, img2_data_norm, bp="yes", lp=lowpass, hp=highpass, sigma=sigma
    )
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    logging.info(f"maxX: {maxX}, {maxY}")
    cen = np.asarray(xcorr.shape) / 2
    logging.info(f"centre = {cen}")
    err = np.array(cen - [maxX, maxY], int)
    logging.info(f"Shift between 1 and 2 is = {err}")
    logging.info(f"img2 is X-shifted by  {err[1]}; Y-shifted by {err[0]}")
    x_shift = err[1] * pixelsize_x_2
    y_shift = err[0] * pixelsize_y_2
    logging.info(f"X-shift =  {x_shift:.2e} meters")
    logging.info(f"Y-shift =  {y_shift:.2e} meters")
    return x_shift, y_shift  # metres


def crosscorrelation(img1, img2, bp="no", *args, **kwargs):
    if img1.shape != img2.shape:
        logging.error(
            "### ERROR in xcorr2: img1 and img2 do not have the same size ###"
        )
        return -1
    if img1.dtype != "float64":
        img1 = np.array(img1, float)
    if img2.dtype != "float64":
        img2 = np.array(img2, float)

    if bp == "yes":
        lpv = kwargs.get("lp", None)
        hpv = kwargs.get("hp", None)
        sigmav = kwargs.get("sigma", None)
        if lpv == "None" or hpv == "None" or sigmav == "None":
            logging.error("ERROR in xcorr2: check bandpass parameters")
            return -1
        bandpass = bandpass_mask(
            size=(img1.shape[1], img1.shape[0]), lp=lpv, hp=hpv, sigma=sigmav
        )
        img1ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img1)))
        s = img1.shape[0] * img1.shape[1]
        tmp = img1ft * np.conj(img1ft)
        img1ft = s * img1ft / np.sqrt(tmp.sum())
        img2ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img2)))
        img2ft[0, 0] = 0
        tmp = img2ft * np.conj(img2ft)
        img2ft = s * img2ft / np.sqrt(tmp.sum())
        xcorr = np.real(fftpack.ifft2(img1ft * np.conj(img2ft)))
    elif bp == "no":
        img1ft = fftpack.fft2(img1)
        img2ft = np.conj(fftpack.fft2(img2))
        img1ft[0, 0] = 0
        xcorr = np.abs(fftpack.fftshift(fftpack.ifft2(img1ft * img2ft)))
    else:
        logging.error(
            "ERROR in xcorr2: bandpass value ( bp= " + str(bp) + " ) not recognized"
        )
        return -1
    return xcorr


def bandpass_mask(size=(128, 128), lp=32, hp=2, sigma=3):
    x = size[0]
    y = size[1]
    lowpass = circ_mask(size=(x, y), radius=lp, sigma=0)
    hpass_tmp = circ_mask(size=(x, y), radius=hp, sigma=0)
    highpass = -1 * (hpass_tmp - 1)
    tmp = lowpass * highpass
    if sigma > 0:
        bandpass = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        bandpass = tmp
    return bandpass


def circ_mask(size=(128, 128), radius=32, sigma=3):
    x = size[0]
    y = size[1]
    img = Image.new("I", size)
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        (x / 2 - radius, y / 2 - radius, x / 2 + radius, y / 2 + radius),
        fill="white",
        outline="white",
    )
    tmp = np.array(img, float) / 255
    if sigma > 0:
        mask = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        mask = tmp
    return mask


# FROM AUTOLAMELLA
def _mask_rectangular(image_shape, sigma=5.0, *, start=None, extent=None):
    """Make a rectangular mask with soft edges for image normalization.

    Parameters
    ----------
    image_shape : tuple
        Shape of the original image array
    sigma : float, optional
        Sigma value (in pixels) for gaussian blur function, by default 5.
    start : tuple, optional
        Origin point of the rectangle, e.g., ([plane,] row, column).
        Default start is 5% of the total image width and height.
    extent : int, optional
        The extent (size) of the drawn rectangle.
        E.g., ([num_planes,] num_rows, num_cols).
        Default is for the rectangle to cover 95% of the image width & height.

    Returns
    -------
    ndarray
        Rectangular mask with soft edges in array matching input image_shape.
    """
    import skimage

    if extent is None:
        # leave at least a 5% gap on each edge
        start = np.round(np.array(image_shape) * 0.05)
        extent = np.round(np.array(image_shape) * 0.90)
    rr, cc = skimage.draw.rectangle(start, extent=extent, shape=image_shape)
    mask = np.zeros(image_shape)
    mask[rr.astype(int), cc.astype(int)] = 1.0
    mask = ndi.gaussian_filter(mask, sigma=sigma)
    return mask




def test_thin_lamella(microscope, settings, image_settings, ref_image=None):

    # TODO: refactor this function
    # rotate and tilt thinning angle

    # user clicks on lamella position

    image_settings["save"] = False
    image_settings["hfw"] = 30e-6
    image_settings["beam_type"] = BeamType.ION
    image_settings["gamma"]["correction"] = False
    image_settings["save"] = True
    image_settings["label"] = "crosscorrelation_1"

    # initial reference image
    ref_image = acquire.new_image(microscope, image_settings)

    # adjust beamshift by known amount
    # microscope.beams.ion_beam.beam_shift.value += (1e-6, 2e-6)

    # align using cross correlation
    img1 = ref_image
    image_settings["label"] = "crosscorrelation_2"
    img2 = acquire.new_image(microscope, settings=image_settings)
    dx, dy = shift_from_crosscorrelation_AdornedImages(
        img1, img2, lowpass=256, highpass=24, sigma=10, use_rect_mask=True
    )

    # adjust beamshift
    microscope.beams.ion_beam.beam_shift.value += (-dx, dy)

    # retake image
    _ = acquire.new_image(microscope, image_settings)

    # load protocol settings
    protocol_stages = []

    for stage_settings in settings["thin_lamella"]["protocol_stages"]:
        tmp_settings = settings["thin_lamella"].copy()
        tmp_settings.update(stage_settings)

        protocol_stages.append(tmp_settings)


    #####################

    # TODO: need to align for each imaging current...
    # # high_current
    # image_settings["label"] = f"crosscorrelation_ref_{protocol_stages[0]['milling_current']}"
    # microscope.beams.ion_beam.beam_current.value = protocol_stages[0]["milling_current"]
    # img_ref_high_current = acquire.new_image(microscope, image_settings)
    #
    # # med_current
    # image_settings["label"] = f"crosscorrelation_ref_{protocol_stages[1]['milling_current']}"
    # microscope.beams.ion_beam.beam_current.value = protocol_stages[1]["milling_current"]
    # img_ref_med_current = acquire.new_image(microscope, image_settings)
    #
    # # low_current
    # image_settings["label"] = f"crosscorrelation_ref_{protocol_stages[2]['milling_current']}"
    # microscope.beams.ion_beam.beam_current.value = protocol_stages[2]["milling_current"]
    # img_ref_low_current = acquire.new_image(microscope, image_settings)
    #
    # img_refs = [img_ref_high_current, img_ref_med_current, img_ref_low_current]

    # maybe a dictionary with the beam_current as key would be better?
    # TODO: make this a loop?
    # TODO: revert to previous imaging current?
    # TODO: will this mean we have to change params for each cross-correlation?
    # TODO: update label here?


    # TODO: mask out the lamella area when cross-correlating


    # for stage_number, stage_settings in protocol_stages:
    #
    #     img1 = img_refs[stage_number]


    ########################

    # mill
    from liftout.fibsem import milling

    ## MILL THIN LAMELLA
    # load protocol stages

    # MILL THIN LAMELLA
    for stage_number, stage_settings in enumerate(protocol_stages):

        # align using cross correlation
        img1 = ref_image
        image_settings["label"] = "crosscorrelation_2"
        img2 = acquire.new_image(microscope, settings=image_settings)
        dx, dy = shift_from_crosscorrelation_AdornedImages(
            img1, img2, lowpass=256, highpass=24, sigma=10, use_rect_mask=True
        )

        # adjust beamshift
        microscope.beams.ion_beam.beam_shift.value += (-dx, dy)
        _ = acquire.new_image(microscope, image_settings)
        logging.info(
            "milling: protocol stage {} of {}".format(
                stage_number + 1, len(protocol_stages)
            )
        )

        milling.setup_milling(microscope, settings, stage_settings)

        # Create and mill patterns

        # upper region
        """Create cleaning cross section milling pattern above lamella position."""
        microscope.imaging.set_active_view(2)  # the ion beam view
        lamella_center_x = 0
        lamella_center_y = 0
        milling_depth = stage_settings["milling_depth"]
        center_y = (
            lamella_center_y
            + (0.5 * stage_settings["lamella_height"])
            + (
                stage_settings["total_cut_height"]
                * stage_settings["percentage_from_lamella_surface"]
            )
            + (
                0.5
                * stage_settings["total_cut_height"]
                * stage_settings["percentage_roi_height"]
            )
        )
        height = float(
            stage_settings["total_cut_height"] * stage_settings["percentage_roi_height"]
        )
        upper_milling_roi = microscope.patterning.create_cleaning_cross_section(
            lamella_center_x,
            center_y,
            stage_settings["lamella_width"],
            height,
            milling_depth,
        )
        upper_milling_roi.scan_direction = "TopToBottom"

        # lower region
        """Create cleaning cross section milling pattern below lamella position."""
        microscope.imaging.set_active_view(2)  # the ion beam view
        lamella_center_x = 0
        lamella_center_y = 0
        milling_depth = stage_settings["milling_depth"]
        center_y = (
            lamella_center_y
            - (0.5 * stage_settings["lamella_height"])
            - (
                stage_settings["total_cut_height"]
                * stage_settings["percentage_from_lamella_surface"]
            )
            - (
                0.5
                * stage_settings["total_cut_height"]
                * stage_settings["percentage_roi_height"]
            )
        )
        height = float(
            stage_settings["total_cut_height"] * stage_settings["percentage_roi_height"]
        )
        lower_milling_roi = microscope.patterning.create_cleaning_cross_section(
            lamella_center_x,
            center_y,
            stage_settings["lamella_width"],
            height,
            milling_depth,
        )
        lower_milling_roi.scan_direction = "BottomToTop"

        # TODO: visualise the milling patterns?

        from autoscript_core.common import ApplicationServerException

        logging.info(f"milling: milling thin lamella pattern...")
        microscope.beams.ion_beam.horizontal_field_width.value = stage_settings[
            "hfw"
        ]
        microscope.imaging.set_active_view(2)  # the ion beam view
        _ = acquire.new_image(microscope, settings=image_settings)

        try:
            microscope.patterning.run()
        except ApplicationServerException:
            logging.error("ApplicationServerException: could not mill!")
        microscope.patterning.clear_patterns()

    # reset milling state and return to imaging current
    logging.info("milling: returning to the ion beam imaging current now.")
    microscope.patterning.clear_patterns()
    microscope.beams.ion_beam.beam_current.value = settings["imaging"]["imaging_current"]
    microscope.patterning.mode = "Serial"
    logging.info("milling: ion beam milling complete.")

    logging.info("Thin Lamella Finished.")

    # for stage in stages:
    #   align
    #   mill stage
    #   setup milling
    #   mill upper region
    #   mill lower region

    return


def auto_focus_and_link(microscope):
    import skimage

    from skimage.morphology import disk
    from skimage.filters.rank import gradient

    PLOT = False

    image_settings = {}
    image_settings["resolution"] = "768x512"  # "1536x1024"
    image_settings["dwell_time"] = 0.3e-6
    image_settings["hfw"] = 50e-6
    image_settings["autocontrast"] = True
    image_settings["beam_type"] = BeamType.ELECTRON
    image_settings["gamma"] = {
        "correction": True,
        "min_gamma": 0.15,
        "max_gamma": 1.8,
        "scale_factor": 0.01,
        "threshold": 46,
    }
    image_settings["save"] = False
    image_settings["label"] = ""
    image_settings["save_path"] = None
    print(image_settings)

    microscope.beams.electron_beam.working_distance.value = 4e-3
    current_working_distance = microscope.beams.electron_beam.working_distance.value

    print("Initial: ", current_working_distance)

    working_distances = [
        current_working_distance - 0.5e-3,
        current_working_distance - 0.25e-3,
        current_working_distance,
        current_working_distance + 0.25e-3,
        current_working_distance + 0.5e-3
    ]

    # loop through working distances and calculate the sharpness (acutance)
    # highest acutance is best focus
    sharpeness_metric = []
    for i, wd in enumerate(working_distances):
        microscope.beams.electron_beam.working_distance.value = wd
        img = acquire.new_image(microscope, image_settings)

        print(f"Img {i}: {img.metadata.optics.working_distance:.5f}")

        # sharpness (Acutance: https://en.wikipedia.org/wiki/Acutance
        out = gradient(skimage.filters.median(np.copy(img.data)), disk(5))

        if PLOT:
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(img.data, cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(out, cmap="gray")
            plt.show()

        sharpness = np.mean(out)
        sharpeness_metric.append(sharpness)

    # select working distance with max acutance
    idx = np.argmax(sharpeness_metric)

    print(*zip(working_distances, sharpeness_metric))
    print(idx, working_distances[idx], sharpeness_metric[idx])

    # reset working distance
    microscope.beams.electron_beam.working_distance.value = working_distances[idx]

    # run fine auto focus and link
    microscope.imaging.set_active_view(1)  # set to Ebeam
    microscope.auto_functions.run_auto_focus()
    microscope.specimen.stage.link()
