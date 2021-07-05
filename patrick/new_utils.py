import numpy as np
import PIL
from PIL import Image
import detection
import matplotlib.pyplot as plt


def take_electron_and_ion_reference_images(
    microscope,
    hor_field_width=50e-6,
    image_settings=None,
    __autocontrast=True,
    eb_brightness=None,
    ib_brightness=None,
    eb_contrast=None,
    ib_contrast=None,
    save=False,
    save_label="_default",
):
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        GrabFrameSettings,
    )

    # image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    #############

    # Take reference images with lower resolution, wider field of view
    microscope.beams.electron_beam.horizontal_field_width.value = hor_field_width
    microscope.beams.ion_beam.horizontal_field_width.value = hor_field_width
    microscope.imaging.set_active_view(1)
    if __autocontrast:
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
    eb_reference = new_electron_image(
        microscope, image_settings, eb_brightness, eb_contrast
    )
    microscope.imaging.set_active_view(2)
    if __autocontrast:
        autocontrast(microscope, beam_type=BeamType.ION)
    ib_reference = new_ion_image(microscope, image_settings, ib_brightness, ib_contrast)

    # save images
    if save:
        storage.SaveImage(eb_image, id=save_label + "_eb")
        storage.SaveImage(ib_image, id=save_label + "_ib")

    return eb_reference, ib_reference

    # REFACTORED ^^

# needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6,
#                 image_settings=image_settings, save=True, save_label="test_image")


######################################## SHARPEN_NEEDLE ###############################################################################################


def sharpen_needle(microscope):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        GrabFrameSettings,
    )
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # def Rotate(x, y, angle):
    #     angle = np.deg2rad(angle)
    #     x_rot = x * math.cos(angle) - y * math.sin(angle)
    #     y_rot = x * math.sin(angle) + y * math.cos(angle)
    #     return x_rot, y_rot

    # move sample stage out
    move_sample_stage_out(microscope)

    # insert needle and move to position
    park_position = insert_needle(microscope)
    stage_tilt = stage.current_position.t
    print("Stage tilt is ", np.rad2deg(stage.current_position.t), " deg...")
    z_move_in = z_corrected_needle_movement(-180e-6, stage_tilt)
    needle.relative_move(z_move_in)

    # if 0: # focus ion beam image : does not work
    #     microscope.imaging.set_active_view(2)  # the ion beam view
    #     original_hfw = microscope.beams.ion_beam.horizontal_field_width.value
    #     microscope.beams.ion_beam.horizontal_field_width.value = 0.000400
    #     print('Automatically refocusing ion  beam.')
    #     microscope.auto_functions.run_auto_focus()
    #     microscope.beams.ion_beam.horizontal_field_width.value = original_hfw

    # needle reference images
    resolution = storage.settings["imaging"]["resolution"]
    dwell_time = storage.settings["imaging"]["dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
    hfw_lowres = storage.settings["imaging"]["horizontal_field_width"]
    needle_eb, needle_ib = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hfw_lowres, image_settings=image_settings, save=True, save_label="A_sharpen_needle_initial"
    )

    x_0, y_0 = needletip_shift_from_img_centre(needle_ib, show=True)

    # new version
    # weights_file = r"\\ad.monash.edu\home\User007\prcle2\Documents\demarco\autoliftout\patrick\models\fresh_full_n10.pt"
    # detector = detection.Detector(weights_file)
    # x_distance, y_distance = detector.calculate_shift_between_features(needle_ib, shift_type="needle_tip_to_image_centre", show=True)





    # move needle to the centre
    x_move = x_corrected_needle_movement(-x_0)
    needle.relative_move(x_move)
    stage_tilt = stage.current_position.t
    print("Stage tilt is ", np.rad2deg(stage.current_position.t), " deg...")
    z_distance = y_0 / np.sin(np.deg2rad(52))
    z_move = z_corrected_needle_movement(z_distance, stage_tilt)
    print("z_move = ", z_move)
    needle.relative_move(z_move)

    # needle images after centering
    resolution = storage.settings["imaging"]["resolution"]
    dwell_time = storage.settings["imaging"]["dwell_time"]
    image_settings = GrabFrameSettings(resolution=resolution, dwell_time=dwell_time)
    hfw_lowres = storage.settings["imaging"]["horizontal_field_width"]
    needle_eb, needle_ib = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hfw_lowres, image_settings=image_settings, save=True, save_label="A_sharpen_needle_centre"
    )
    # storage.SaveImage(needle_eb, id="A_sharpen_needle_eb_centre")
    # storage.SaveImage(needle_ib, id="A_sharpen_needle_ib_centre")

    x_0, y_0 = needletip_shift_from_img_centre(needle_ib, show=True)

    # new version
    # x_distance, y_distance = detector.calculate_shift_between_features(needle_ib, shift_type="needle_tip_to_image_centre", show=True)


    # create sharpening patterns
    cut_coord_bottom, cut_coord_top = calculate_sharpen_needle_pattern(x_0=x_0, y_0=y_0)

    setup_ion_milling(microscope, ion_beam_field_of_view=hfw)

    sharpen_patterns = create_sharpen_needle_patterns(
        microscope, cut_coord_bottom, cut_coord_top
    )

    # run needle sharpening
    confirm_and_run_milling(microscope, milling_current, confirm=True)

    needle_eb, needle_ib = take_electron_and_ion_reference_images(
        microscope, hor_field_width=hfw_lowres, image_settings=image_settings, save=True, save_label="A_sharpen_needle_sharp"
    )
    # storage.step_counter += 1
    # storage.SaveImage(needle_eb, id="A_sharpen_needle_eb_sharp")
    # storage.SaveImage(needle_ib, id="A_sharpen_needle_ib_sharp")
    storage.step_counter += 1

    # retract needle
    retract_needle(microscope, park_position)


def calculate_sharpen_needle_pattern(x_0, y_0):
    height = storage.settings["sharpen"]["height"]
    width = storage.settings["sharpen"]["width"]
    depth = storage.settings["sharpen"]["depth"]
    bias = storage.settings["sharpen"]["bias"]
    hfw = storage.settings["sharpen"]["hfw"]
    tip_angle = storage.settings["sharpen"]["tip_angle"]  # 2NA of the needle   2*alpha
    needle_angle = storage.settings["sharpen"][
        "needle_angle"
    ]  # needle tilt on the screen 45 deg +/-
    milling_current = storage.settings["sharpen"]["sharpen_milling_current"]

    alpha = tip_angle / 2  # half of NA of the needletip
    beta = np.rad2deg(
        np.arctan(width / height)
    )  # box's width and length, beta is the diagonal angle
    D = np.sqrt(width ** 2 + height ** 2) / 2  # half of box diagonal
    rotation_1 = -(needle_angle + alpha)
    rotation_2 = -(needle_angle - alpha) - 180

    ############################################################################
    # dx_1 = D * math.cos( np.deg2rad(needle_angle + alpha) )
    # dy_1 = D * math.sin( np.deg2rad(needle_angle + alpha) )
    # x_1 = x_0 - dx_1 # centre of the bottom box
    # y_1 = y_0 - dy_1 # centre of the bottom box

    # dx_2 = D * math.cos( np.deg2rad(needle_angle - alpha - beta) )
    # dy_2 = D * math.sin( np.deg2rad(needle_angle - alpha - beta) )
    # x_2 = x_0 - dx_2 # centre of the top box
    # y_2 = y_0 - dy_2 # centre of the top box

    # x_1_origin = x_1 - x_0
    # y_1_origin = y_1 - y_0 # shift the x1,y1 to the origin
    # x_2_origin_rot, y_2_origin_rot = Rotate( x_1_origin, y_1_origin, 360-(2*alpha+2*beta) ) # rotate to get the x2,y2 point
    # x_2_rot = x_2_origin_rot + x_0 # shift to the old centre at x0,y0
    # y_2_rot = y_2_origin_rot + y_0

    ############################################################################
    dx_1 = (width / 2) * math.cos(np.deg2rad(needle_angle + alpha))
    dy_1 = (width / 2) * math.sin(np.deg2rad(needle_angle + alpha))
    ddx_1 = (height / 2) * math.sin(np.deg2rad(needle_angle + alpha))
    ddy_1 = (height / 2) * math.cos(np.deg2rad(needle_angle + alpha))
    x_1 = x_0 - dx_1 + ddx_1  # centre of the bottom box
    y_1 = y_0 - dy_1 - ddy_1  # centre of the bottom box

    dx_2 = D * math.cos(np.deg2rad(needle_angle - alpha))
    dy_2 = D * math.sin(np.deg2rad(needle_angle - alpha))
    ddx_2 = (height / 2) * math.sin(np.deg2rad(needle_angle - alpha))
    ddy_2 = (height / 2) * math.cos(np.deg2rad(needle_angle - alpha))
    x_2 = x_0 - dx_2 - ddx_2  # centre of the top box
    y_2 = y_0 - dy_2 + ddy_2  # centre of the top box

    print("needletip xshift offcentre: ", x_0, "; needletip yshift offcentre: ", y_0)
    print("width: ", width)
    print("height: ", height)
    print("depth: ", depth)
    print("needle_angle: ", needle_angle)
    print("tip_angle: ", tip_angle)
    print("rotation1 :", rotation_1)
    print("rotation2 :", rotation_2)
    print("=================================================")
    print("centre of bottom box: x1 = ", x_1, "; y1 = ", y_1)
    print("centre of top box:    x2 = ", x_2, "; y2 = ", y_2)
    print("=================================================")

    # pattern = microscope.patterning.create_rectangle(x_3, y_3, width+2*bias, height+2*bias, depth)
    # pattern.rotation = np.deg2rad(rotation_1)
    # pattern = microscope.patterning.create_rectangle(x_4, y_4, width+2*bias, height+2*bias, depth)
    # pattern.rotation = np.deg2rad(rotation_2)

    # bottom cut pattern
    cut_coord_bottom = {
        "center_x": x_1,
        "center_y": y_1,
        "width": width,
        "height": height - bias,
        "depth": depth,
        "rotation": rotation_1,
        "hfw": hfw,
    }

    # top cut pattern
    cut_coord_top = {
        "center_x": x_2,
        "center_y": y_2,
        "width": width,
        "height": height - bias,
        "depth": depth,
        "rotation": rotation_2,
        "hfw": hfw,
    }

    return cut_coord_bottom, cut_coord_top


def create_sharpen_needle_patterns(microscope, cut_coord_bottom, cut_coord_top):
    sharpen_patterns = []
    for cut_coord in [cut_coord_bottom, cut_coord_top]:
        pattern = _create_sharpen_pattern(
            microscope,
            center_x=cut_coord["center_x"],
            center_y=cut_coord["center_y"],
            width=cut_coord["width"],
            height=cut_coord["height"],
            depth=cut_coord["depth"],
            rotation_degrees=cut_coord["rotation"],
            ion_beam_field_of_view=cut_coord["hfw"],
        )
        sharpen_patterns.append(pattern)

    return sharpen_patterns

# REFACTORED ^^^

###################################################################################################################################################




def calculate_shift_distance_in_metres(img, distance_x, distance_y):
    """Convert the shift distance from proportion of img to metres"""

    pixelsize_x = img.metadata.binary_result.pixel_size.x #5.20833e-008
    field_width   = pixelsize_x  * img.width
    field_height  = pixelsize_x  * img.height
    x_shift = distance_x * field_width
    y_shift = distance_y * field_height
    print('x_shift = ', x_shift/1e-6, 'um; ', 'y_shift = ', y_shift/1e-6, 'um; ')

    return x_shift, y_shift


###########################################  REALIGN EUCENTRIC WITH ML #############################################################################

def realign_eucentric_with_machine_learning(microscope, image_settings, hor_field_width=150e-6, _autocontrast=False):
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        GrabFrameSettings,
        ManipulatorPosition,
        StagePosition
    )

    """ Realign image to lamella centre using ML detection"""
    stage = microscope.specimen.stage

    eb_brightness = storage.settings["machine_learning"]["eb_brightness"]
    eb_contrast = storage.settings["machine_learning"]["eb_contrast"]
    ib_brightness = storage.settings["machine_learning"]["ib_brightness"]
    ib_contrast = storage.settings["machine_learning"]["ib_contrast"]

    storage.step_counter += 1
    save_label = "A_eucentric_calibration_lowres"

    if _autocontrast:
        eb_lowres, ib_lowres = take_electron_and_ion_reference_images(
            microscope,
            hor_field_width=hor_field_width,
            image_settings=image_settings,
            __autocontrast=_autocontrast,
            save=True,
            save_label=save_label,
        )
    else:
        eb_lowres, ib_lowres = take_electron_and_ion_reference_images(
            microscope,
            hor_field_width=hor_field_width,
            image_settings=image_settings,
            eb_brightness=eb_brightness,
            ib_brightness=ib_brightness,
            eb_contrast=eb_contrast,
            ib_contrast=ib_contrast,
            __autocontrast=_autocontrast,
            save=True,
            save_label=save_label,
        )

    # in microscope coordinates (not img coordinate)
    ib_distance_y, ib_distance_x = lamella_shift_from_img_centre(ib_lowres, show=True)


    # test new version
    # weights_file = r"\\ad.monash.edu\home\User007\prcle2\Documents\demarco\autoliftout\patrick\models\fresh_full_n10.pt"
    # detector = detection.Detector(weights_file)
    # x_distance, y_distance = detector.calculate_shift_between_features(ib_lowres, shift_type="lamella_centre_to_image_centre", show=True)
    # dx_meters_ion, dy_meters_ion = calculate_shift_distance_in_metres(ib_lowres, distance_x, distance_y)


    # tested, can align within 20um

    # z correction
    pixelsize_x = ib_lowres.metadata.binary_result.pixel_size.x
    field_width = pixelsize_x * ib_lowres.width
    dy_meters_ion = ib_distance_y * field_width
    tilt_radians = stage.current_position.t

    # if 0:  # simple way, from eucentric correction for landing poles
    #     delta_z = -np.cos(tilt_radians) * dy_meters_ion
    #     # delta_z = -dy_meters_ion / np.cos(np.deg2rad(-38))
    #     stage.relative_move(StagePosition(z=delta_z))
    yz_move_ion = y_corrected_stage_movement(
        dy_meters_ion, stage_tilt=tilt_radians, beam_type=BeamType.ION
    )
    stage.relative_move(yz_move_ion)

    # x correction
    field_height = pixelsize_x * ib_lowres.height
    dx_meters_ion = ib_distance_x * field_height
    stage.relative_move(StagePosition(x=dx_meters_ion))

    print("Realigning x and y in ion beam")
    # print(f"Electron Beam: {eb_distance_y}, {eb_distance_x}")
    print(f"Ion Beam: {ib_distance_y}, {ib_distance_x}")
    print(f"Delta X movement: {dx_meters_ion}")
    print(f"Delta Y movement: {yz_move_ion.y}")
    print(f"Delta Z movement: {yz_move_ion.z}")


    ############################################# PART 2 #############################################

    # electron dy shift
    save_label = "A_eucentric_calibration_lowres_moved_1"

    if _autocontrast:
        eb_lowres, ib_lowres = take_electron_and_ion_reference_images(
            microscope,
            hor_field_width=hor_field_width,
            image_settings=image_settings,
            __autocontrast=_autocontrast,
            save=True,
            save_label=save_label,
        )
    else:
        eb_lowres, ib_lowres = take_electron_and_ion_reference_images(
            microscope,
            hor_field_width=hor_field_width,
            image_settings=image_settings,
            eb_brightness=eb_brightness,
            ib_brightness=ib_brightness,
            eb_contrast=eb_contrast,
            ib_contrast=ib_contrast,
            __autocontrast=_autocontrast,
            save=True,
            save_label=save_label,
        )

    # in microscope coordinates (not img coordinate)
    # renormalise_and_mask_image
    eb_distance_y, eb_distance_x = lamella_shift_from_img_centre(eb_lowres, show=True)

    # new version
    # x_distance, y_distance = detector.calculate_shift_between_features(eb_lowres, shift_type="lamella_centre_to_image_centre", show=True)
    # dx_meters_elec, dy_meters_elec = calculate_shift_distance_in_metres(eb_lowres, distance_x, distance_y)


    # z correction
    pixelsize_x = eb_lowres.metadata.binary_result.pixel_size.x
    field_width = pixelsize_x * eb_lowres.width
    dy_meters_elec = eb_distance_y * field_width
    tilt_radians = stage.current_position.t
    # delta_z = -np.cos(tilt_radians) * dy_meters_elec
    # stage.relative_move(StagePosition(z=delta_z))
    ##### stage.relative_move(StagePosition(y=dy_meters_elec))
    yz_move_elec = y_corrected_stage_movement(
        dy_meters_elec, stage_tilt=tilt_radians, beam_type=BeamType.ELECTRON
    )
    stage.relative_move(yz_move_elec)

    # x correction
    field_height = pixelsize_x * eb_lowres.height
    dx_meters_elec = eb_distance_x * field_height
    stage.relative_move(StagePosition(x=dx_meters_elec))

    print("Realigning x and y in electron beam")
    print(f"Electron Beam: {eb_distance_y}, {eb_distance_x}")
    print(f"Delta Y movement: {yz_move_elec.y}")
    print(f"Delta Z movement: {yz_move_elec.z}")
    print(f"Delta X movement: {dx_meters_elec}")


    ############################################# PART 3 #############################################

    # ion again
    save_label = "A_eucentric_calibration_lowres_moved_2"

    if _autocontrast:
        eb_lowres, ib_lowres = take_electron_and_ion_reference_images(
            microscope,
            hor_field_width=hor_field_width,
            image_settings=image_settings,
            __autocontrast=_autocontrast,
            save=True,
            save_label=save_label,
        )
    else:
        eb_lowres, ib_lowres = take_electron_and_ion_reference_images(
            microscope,
            hor_field_width=hor_field_width,
            image_settings=image_settings,
            eb_brightness=eb_brightness,
            ib_brightness=ib_brightness,
            eb_contrast=eb_contrast,
            ib_contrast=ib_contrast,
            __autocontrast=_autocontrast,
            save=True,
            save_label=save_label,
        )

    # in microscope coordinates (not img coordinate)
    ib_distance_y, ib_distance_x = lamella_shift_from_img_centre(ib_lowres, show=True)

    # new version
    # x_distance, y_distance = detector.calculate_shift_between_features(ib_lowres, shift_type="lamella_centre_to_image_centre", show=True)
    # dx_meters_ion, dy_meters_ion = calculate_shift_distance_in_metres(eb_lowres, distance_x, distance_y)

    # z correction
    pixelsize_x = ib_lowres.metadata.binary_result.pixel_size.x
    field_width = pixelsize_x * ib_lowres.width
    dy_meters_ion = ib_distance_y * field_width
    tilt_radians = stage.current_position.t
    delta_z = -np.cos(tilt_radians) * dy_meters_ion
    # delta_z = -dy_meters_ion / np.cos(np.deg2rad(-38))
    # stage.relative_move(StagePosition(z=delta_z))
    yz_move_ion = y_corrected_stage_movement(
        dy_meters_ion, stage_tilt=tilt_radians, beam_type=BeamType.ION
    )
    stage.relative_move(yz_move_ion)

    # x correction
    field_height = pixelsize_x * ib_lowres.height
    dx_meters_ion = ib_distance_x * field_height
    stage.relative_move(StagePosition(x=dx_meters_ion))

    print("Realigning x and y in ion beam")
    print(f"Ion Beam: {ib_distance_y}, {ib_distance_x}")
    print(f"Delta Y movement: {yz_move_ion.y}")
    print(f"Delta Z movement: {yz_move_ion.z}")
    print(f"Delta X movement: {dx_meters_ion}")

    ############################################# PART 4 #############################################

    # take final position images
    save_label = "A_eucentric_calibration_lowres_final"

    if _autocontrast:
        eb_lowres, ib_lowres = take_electron_and_ion_reference_images(
            microscope,
            hor_field_width=hor_field_width,
            image_settings=image_settings,
            __autocontrast=_autocontrast,
            save=True,
            save_label=save_label,
        )
    else:
        eb_lowres, ib_lowres = take_electron_and_ion_reference_images(
            microscope,
            hor_field_width=hor_field_width,
            image_settings=image_settings,
            eb_brightness=eb_brightness,
            ib_brightness=ib_brightness,
            eb_contrast=eb_contrast,
            ib_contrast=ib_contrast,
            __autocontrast=_autocontrast,
            save=True,
            save_label=save_label,
        )

    storage.step_counter += 1


###################################################################################################################################################

# Detection, Drawing helper functions

def select_point_new(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    coords = []

    def on_click(event):
        print(event.xdata, event.ydata)
        coords.append(event.ydata)
        coords.append(event.xdata)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    return tuple(coords[-2:])

def validate_detection(img, img_base, detection_coord, det_type):
    correct = input(f"Is the detection for {det_type} correct? (y/n)")
    #TODO: change this to user_input
    if correct == "n":

        detection_coord_initial = detection_coord # save initial coord

        print(f"Please click the {det_type} position")
        detection_coord = select_point_new(img)

        # TODO: need to resolve behaviour when user exits plot without selecting?
        # if detection_coord is None:
        #     detection_coord = detection_coord # use initial coord


        # save image for training here
        print("Saving image for labelling")
        #storage.step_counter +=1
        #storage.SaveImage(img_base, id="label_")


    print(f"{det_type}: {detection_coord}")
    return detection_coord


def load_image_from_live(img):

    """ Load a live image from the microscope as np.array """

    return np.asarray(img.data)


def load_image_from_file(fname):

    """ Load a .tif image from disk as np.array """

    img = np.asarray(Image.open(fname))

    return img
