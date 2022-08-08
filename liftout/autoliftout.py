import logging
import time
from datetime import datetime

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         StagePosition)

from liftout.detection.detection import DetectionType
from liftout.fibsem import acquire, calibration, movement
from liftout.fibsem import utils as fibsem_utils
from liftout.fibsem.acquire import BeamType, ImageSettings
from liftout.fibsem.milling import MillingPattern
from liftout.fibsem.sample import (AutoLiftoutStage, Lamella, ReferenceImages,
                                   Sample)
from liftout.gui import windows

# autoliftout workflow functions

# functional mill trench
def mill_lamella_trench(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
):

    # bookkeeping
    image_settings.save_path = lamella.path

    # move to lamella position
    movement.safe_absolute_stage_movement(microscope, lamella.lamella_coordinates)

    # move flat to the ion beam, stage tilt 25 (total image tilt 52)
    movement.move_to_trenching_angle(microscope, settings)

    # Take an ion beam image at the *milling current*
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]

    # TODO: correct stage drift?

    ##### 

    # confirm position
    # windows.ask_user_movement(
    #     microscope, settings, image_settings, msg_type="centre_ib"
    # )

    # update the lamella coordinates, and save
    # lamella.lamella_coordinates = calibration.get_raw_stage_position(microscope)

    ###### 

    # mill_trenches
    windows.open_milling_window(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern=MillingPattern.Trench,
        x=0,
        y=0,
    )

    # reference images of milled trenches
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_med_res"]
    image_settings.save = True
    image_settings.label = f"ref_trench_low_res"
    acquire.take_reference_images(microscope, image_settings=image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label = f"ref_trench_high_res"
    acquire.take_reference_images(microscope, image_settings=image_settings)

    return lamella


# functional jcut
def mill_lamella_jcut(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
) -> Lamella:

    # bookkeeping
    image_settings.save_path = lamella.path

    # reference images of milled trenches
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_med_res"]
    image_settings.save = True
    image_settings.label = f"ref_trench_low_res"
    acquire.take_reference_images(microscope, image_settings=image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label = f"ref_trench_high_res"
    acquire.take_reference_images(microscope, image_settings=image_settings)

    # load the reference images
    reference_images = ReferenceImages(
        low_res_eb=lamella.load_reference_image("ref_trench_low_res_eb"),
        high_res_eb=lamella.load_reference_image("ref_trench_high_res_eb"),
        low_res_ib=lamella.load_reference_image("ref_trench_low_res_ib"),
        high_res_ib=lamella.load_reference_image("ref_trench_high_res_ib"),
    )

    # move flat to electron beam
    movement.flat_to_beam(microscope, settings, beam_type=BeamType.ELECTRON)

    # correct drift using reference images..
    calibration.correct_stage_drift(
        microscope,
        settings,
        image_settings,
        reference_images,
        alignment=(BeamType.ELECTRON, BeamType.ION),
        rotate=True,
        parent_ui=True,
    )

    # move to jcut angle
    stage_settings = MoveSettings(rotate_compucentric=True, tilt_compucentric=True)
    microscope.specimen.stage.relative_move(
        StagePosition(t=np.deg2rad(settings["jcut"]["jcut_angle"])), stage_settings
    )

    # correct drift using reference images..
    calibration.correct_stage_drift(
        microscope,
        settings,
        image_settings,
        reference_images,
        alignment=(BeamType.ION, BeamType.ION),
        rotate=False,
        parent_ui=True,
    )

    ## MILL_JCUT
    # now we are at the angle for jcut, perform jcut
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    windows.open_milling_window(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern=MillingPattern.JCut,
        x=0,
        y=0,
    )

    # take reference images of the jcut
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_med_res"]
    image_settings.save = True
    image_settings.label = f"ref_jcut_tilt_low_res"
    acquire.take_reference_images(microscope, image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_ultra_res"]
    image_settings.label = f"ref_jcut_tilt_high_res"
    acquire.take_reference_images(microscope, image_settings)

    # move to flat eb
    movement.flat_to_beam(microscope, settings, BeamType.ELECTRON)

    # realign
    reference_images = ReferenceImages(
        low_res_eb=lamella.load_reference_image("ref_jcut_tilt_low_res_eb"),
        high_res_eb=lamella.load_reference_image("ref_jcut_tilt_high_res_eb"),
        low_res_ib=lamella.load_reference_image("ref_jcut_tilt_low_res_ib"),
        high_res_ib=lamella.load_reference_image("ref_jcut_tilt_high_res_ib"),
    )

    calibration.correct_stage_drift(
        microscope, settings, image_settings, reference_images, 
        alignment=(BeamType.ELECTRON, BeamType.ELECTRON),
        rotate=True, parent_ui=True
    )

    # take reference images of the jcut
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_med_res"]
    image_settings.save = True
    image_settings.label = f"ref_jcut_low_res"
    acquire.take_reference_images(microscope, image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_ultra_res"]
    image_settings.label = f"ref_jcut_high_res"
    acquire.take_reference_images(microscope, image_settings)

    return lamella


def liftout_lamella(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
) -> Lamella:

    # bookkeeping
    image_settings.save_path = lamella.path

    # convenience
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # get ready to do liftout by moving to liftout angle (flat to eb)
    movement.move_to_liftout_angle(microscope, settings)

    # check eucentric height
    windows.ask_user_movement(
        microscope, settings, image_settings, msg_type="eucentric",
    )  # liftout angle is flat to SEM

    # OR 'untilt' the image... with perspective transform
    reference_images = ReferenceImages(
        low_res_eb=lamella.load_reference_image("ref_jcut_low_res_eb"),
        high_res_eb=lamella.load_reference_image("ref_jcut_high_res_eb"),
        low_res_ib=lamella.load_reference_image("ref_jcut_low_res_ib"),
        high_res_ib=lamella.load_reference_image("ref_jcut_high_res_ib"),
    )

    calibration.correct_stage_drift(
        microscope, settings, image_settings, reference_images, 
        alignment=(BeamType.ELECTRON, BeamType.ELECTRON),
        rotate=True, parent_ui=True
    )

    # reference images for needle location
    image_settings.save = True
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.label = f"ref_needle_liftout"
    acquire.take_reference_images(microscope, image_settings)



    # land needle on lamella
    lamella = land_needle_on_milled_lamella_v2(
        microscope, settings, image_settings, lamella
    )

    # sputter platinum
    fibsem_utils.sputter_platinum(microscope, settings, whole_grid=False)
    logging.info(
        f"{lamella.current_state.stage.name}: lamella to needle welding complete."
    )

    image_settings.save = True
    image_settings.hfw = settings["platinum"]["weld"]["hfw"]
    image_settings.label = f"needle_landed_Pt_sputter"
    acquire.take_reference_images(microscope, image_settings)

    # jcut sever pattern
    windows.open_milling_window(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern=MillingPattern.Sever,
        x=0,
        y=0,
    )

    # take reference images
    image_settings.save = True
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label = f"jcut_sever"
    acquire.take_reference_images(microscope, image_settings)

    # Raise needle 30um from trench
    logging.info(
        f"{lamella.current_state.stage.name}: start removing needle from trench"
    )
    for i in range(3):
        z_move_out_from_trench = movement.z_corrected_needle_movement(
            10e-6, stage.current_position.t
        )
        needle.relative_move(z_move_out_from_trench)
        image_settings.label = f"liftout_trench_{i}"
        acquire.take_reference_images(microscope, image_settings)
        logging.info(
            f"{lamella.current_state.stage.name}: removing needle from trench at {z_move_out_from_trench} ({i + 1}/3)"
        )
        time.sleep(1)

    # reference images after liftout complete
    image_settings.label = f"ref_liftout"
    acquire.take_reference_images(microscope, image_settings)

    # move needle to park position
    movement.retract_needle(microscope)

    return lamella


def land_needle_on_milled_lamella(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
) -> Lamella:

    # bookkeeping
    image_settings.save_path = lamella.path

    # validate needle insertion conditions
    validate_needle_insertion(microscope, settings)

    # insert the needle
    park_position = movement.move_needle_to_liftout_position(microscope)
    logging.info(
        f"{lamella.current_state.stage.name}: needle inserted to park positon: {park_position}"
    )

    # reference images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.save = True
    image_settings.label = f"needle_liftout_start_position_lowres"

    det = calibration.validate_detection(
        microscope,
        settings,
        image_settings,
        lamella=lamella,
        shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
        beam_type=BeamType.ELECTRON,
    )

    movement.move_needle_relative_with_corrected_movement(
        microscope=microscope,
        settings=settings,
        dx=det.distance_metres.x,
        dy=det.distance_metres.y,
        beam_type=BeamType.ELECTRON,
    )

    ### Z-HALF MOVE (ION)
    # calculate shift between lamella centre and needle tip in the ion view
    image_settings.label = f"needle_liftout_post_xy_movement_lowres"
    det = calibration.validate_detection(
        microscope,
        settings,
        image_settings,
        lamella=lamella,
        shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
        beam_type=BeamType.ION,
    )

    # calculate shift in xyz coordinates
    movement.move_needle_relative_with_corrected_movement(
        microscope=microscope,
        settings=settings,
        dx=det.distance_metres.x,
        dy=-det.distance_metres.y / 2,
        beam_type=BeamType.ION,
    )

    # repeat the final movement until user confirms landing.
    response = False
    while response is False:

        ### Z-MOVE FINAL (ION)
        image_settings.hfw = settings["calibration"]["reference_images"][
            "hfw_super_res"
        ]
        image_settings.label = f"needle_liftout_post_z_half_movement_highres"
        det = calibration.validate_detection(
            microscope,
            settings,
            image_settings,
            lamella=lamella,
            shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
            beam_type=BeamType.ION,
        )

        movement.move_needle_relative_with_corrected_movement(
            microscope=microscope,
            settings=settings,
            dx=det.distance_metres.x,
            dy=-det.distance_metres.y,
            beam_type=BeamType.ELECTRON,
        )

        image_settings.save = False
        acquire.take_reference_images(microscope, image_settings)

        response = windows.ask_user_interaction(
            microscope,
            msg="Has the needle landed on the lamella? \nPress Yes to continue, or No to redo the final movement",
            beam_type=BeamType.ION,
        )

    # take final reference images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.save = True
    image_settings.label = f"needle_liftout_landed_lowres"
    acquire.take_reference_images(microscope, image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label = f"needle_liftout_landed_highres"
    acquire.take_reference_images(microscope, image_settings)

    return lamella


## TODO: test
def land_needle_on_milled_lamella_v2(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
) -> Lamella:


    # bookkeeping
    image_settings.save_path = lamella.path

    # validate needle insertion conditions
    validate_needle_insertion(microscope, settings)

    # insert the needle
    park_position = movement.move_needle_to_liftout_position(microscope)
    logging.info(
        f"{lamella.current_state.stage.name}: needle inserted to park positon: {park_position}"
    )

    # reference images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.save = True
    image_settings.label = f"needle_liftout_start_position"

    det = calibration.validate_detection(
        microscope,
        settings,
        image_settings,
        lamella=lamella,
        shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
        beam_type=BeamType.ELECTRON,
    )

    movement.move_needle_relative_with_corrected_movement(
        microscope=microscope,
        settings=settings,
        dx=det.distance_metres.x,
        dy=det.distance_metres.y,
        beam_type=BeamType.ELECTRON,
    )

    ### Z-HALF MOVE (ION)
    # calculate shift between lamella centre and needle tip in the ion view
    image_settings.label = f"needle_liftout_post_xy_movement_lowres"
    det = calibration.validate_detection(
        microscope,
        settings,
        image_settings,
        lamella=lamella,
        shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
        beam_type=BeamType.ION,
    )

    # calculate shift in xyz coordinates
    movement.move_needle_relative_with_corrected_movement(
        microscope=microscope,
        settings=settings,
        dx=det.distance_metres.x,
        dy=-det.distance_metres.y / 2,
        beam_type=BeamType.ION,
    )


     # take image
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings) 
    
    # measure brightness
    BRIGHTNESS_FACTOR  = 1.2
    CROP_SIZE = 250
    previous_brightness = calibration.measure_brightness(ib_image, crop_size=CROP_SIZE)

    image_settings.hfw = settings["calibration"]["reference_images"][
        "hfw_super_res"
    ]
    image_settings.label = f"needle_liftout_post_z_half_movement_highres"

    while True:

        # move needle down 1um
        movement.move_needle_relative_with_corrected_movement(
            microscope, settings, dx=0, dy=-1.e-6, beam_type=BeamType.ION
        )

        # take image
        # TODO: use reduced area?
        ib_image = acquire.new_image(microscope, image_settings) # maybe use the reduced area option?
        brightness = calibration.measure_brightness(ib_image, crop_size=CROP_SIZE)

        if brightness > previous_brightness * BRIGHTNESS_FACTOR:
            # needle has landed...

            # check with user?
            response = windows.ask_user_interaction(
                microscope,
                msg="Has the needle landed on the lamella? \nPress Yes to continue, or No to redo the final movement",
                beam_type=BeamType.ION,
            )
    
            if response:
                break
            
        previous_brightness = brightness

    # take final reference images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.save = True
    image_settings.label = f"needle_liftout_landed_lowres"
    acquire.take_reference_images(microscope, image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label = f"needle_liftout_landed_highres"
    acquire.take_reference_images(microscope, image_settings)

    return lamella


def land_lamella(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
) -> Lamella:

    # bookkeeping
    image_settings.save_path = lamella.path

    # conveienence
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # move to landing coordinate
    movement.safe_absolute_stage_movement(
        microscope=microscope, stage_position=lamella.landing_coordinates,
    )
    movement.auto_link_stage(
        microscope, hfw=settings["calibration"]["reference_images"]["hfw_med_res"],
    )

    # confirm eucentricity
    windows.ask_user_movement(
        microscope, settings, image_settings, msg_type="eucentric", flat_to_sem=False,
    )

    # after eucentricity... we should be at 4mm,
    # so we should set wd to 4mm and link

    logging.info(
        f"{lamella.current_state.stage.name}: initial landing calibration complete."
    )

    ############################## LAND_LAMELLA ##############################
    validate_needle_insertion(microscope, settings)
    movement.move_needle_to_landing_position(microscope)

    #### Y-MOVE (ELECTRON)
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.beam_type = BeamType.ELECTRON
    image_settings.save = True
    image_settings.label = f"landing_needle_start_position"

    acquire.take_reference_images(microscope, image_settings)


    det = calibration.validate_detection(
        microscope,
        settings,
        image_settings,
        lamella=lamella,
        shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
        beam_type=BeamType.ELECTRON,
    )

    # only move y
    movement.move_needle_relative_with_corrected_movement(
        microscope=microscope,
        settings=settings,
        dx=0,
        dy=det.distance_metres.y,
        beam_type=BeamType.ELECTRON,
    )

    image_settings.label = f"landing_needle_after_y_move"
    acquire.take_reference_images(microscope, image_settings)

    #### Z-MOVE (ION)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.beam_type = BeamType.ION
    image_settings.save = True
    image_settings.label = f"landing_needle_after_z_move"

    det = calibration.validate_detection(
        microscope,
        settings,
        image_settings,
        lamella=lamella,
        shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
        beam_type=BeamType.ION,
    )

    # up is down
    movement.move_needle_relative_with_corrected_movement(
        microscope=microscope,
        settings=settings,
        dx=0,
        dy=-det.distance_metres.y,
        beam_type=BeamType.ION,
    )

    acquire.take_reference_images(microscope, image_settings)

    #### X-HALF-MOVE (ION)
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.beam_type = BeamType.ELECTRON
    image_settings.save = True
    image_settings.label = f"landing_needle_land_sample_lowres_after_z_move"

    det = calibration.validate_detection(
        microscope,
        settings,
        image_settings,
        lamella=lamella,
        shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
        beam_type=BeamType.ION,
    )

    # half move
    movement.move_needle_relative_with_corrected_movement(
        microscope=microscope,
        settings=settings,
        dx=det.distance_metres.x / 2,
        dy=0,
        beam_type=BeamType.ION,
    )
    acquire.take_reference_images(microscope, image_settings)

    # repeat final movement until user confirms landing
    response = False
    while response is False:
        #### X-MOVE
        image_settings.hfw = settings["calibration"]["reference_images"][
            "hfw_super_res"
        ]
        image_settings.beam_type = BeamType.ION
        image_settings.save = True
        image_settings.label = f"landing_needle_land_sample_lowres_after_z_move"

        det = calibration.validate_detection(
            microscope,
            settings,
            image_settings,
            lamella=lamella,
            shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
            beam_type=BeamType.ION,
        )

        movement.move_needle_relative_with_corrected_movement(
            microscope=microscope,
            settings=settings,
            dx=det.distance_metres.x / 2,
            dy=0,
            beam_type=BeamType.ION,
        )

        # final reference images
        image_settings.hfw = (
            settings["calibration"]["reference_images"]["hfw_super_res"],
        )
        image_settings.beam_type = BeamType.ELECTRON
        image_settings.save = True
        image_settings.label = f"landing_lamella_final_weld_highres"
        acquire.take_reference_images(
            microscope=microscope, image_settings=image_settings
        )

        response = windows.ask_user_interaction(
            microscope,
            msg="Has the lamella landed on the post? \nPress Yes to continue, or No to redo the final movement",
            beam_type=BeamType.ION,
        )

    #################################################################################################

    ############################## WELD TO LANDING POST #############################################

    windows.open_milling_window(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern=MillingPattern.Weld,
        x=0,
        y=0,
    )

    # final reference images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.save = True
    image_settings.label = f"landing_lamella_final_weld_highres"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    #################################################################################################

    ###################################### CUT_OFF_NEEDLE ######################################

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.beam_type = BeamType.ION
    image_settings.save = True
    image_settings.label = "landing_lamella_pre_cut_off"

    # TODO: can eliminate this if the lamella lands in the centre... just manually calc it
    det = calibration.validate_detection(
        microscope,
        settings,
        image_settings,
        lamella=lamella,
        shift_type=(DetectionType.NeedleTip, DetectionType.ImageCentre),
        beam_type=BeamType.ION,
    )

    # cut off needle
    windows.open_milling_window(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern=MillingPattern.Cut,
        x=det.distance_metres.x,
        y=det.distance_metres.y,
    )

    ################################### REMOVE_NEEDLE ##########################################

    # reference images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.beam_type = BeamType.ION
    image_settings.save = True
    image_settings.label = "landing_lamella_final_cut_lowres"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label = "landing_lamella_final_cut_highres"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    logging.info(
        f"{lamella.current_state.stage.name}: removing needle from landing post"
    )
    # move needle out of trench slowly at first
    for i in range(3):
        z_move_out_from_post = movement.z_corrected_needle_movement(
            10e-6, stage.current_position.t
        )
        needle.relative_move(z_move_out_from_post)
        logging.info(
            f"{lamella.current_state.stage.name}: moving needle out: {z_move_out_from_post} ({i + 1}/3)"
        )
        time.sleep(1)

    # move needle to park position
    movement.retract_needle(microscope)
    logging.info(f"{lamella.current_state.stage.name}: needle retracted.")

    # reference images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.save = True
    image_settings.label = "ref_landing_lamella_low_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label = "ref_landing_lamella_high_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    return lamella


def reset_needle(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
) -> Lamella:

    # bookkeeping
    image_settings.save_path = lamella.path

    # convienence
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # move sample stage out
    movement.move_sample_stage_out(microscope)
    logging.info(f"{lamella.current_state.stage.name}: moved sample stage out")

    ###################################### SHARPEN_NEEDLE ######################################

    # move needle in
    park_position = movement.insert_needle(microscope)
    z_move_in = movement.z_corrected_needle_movement(-180e-6, stage.current_position.t)
    needle.relative_move(z_move_in)
    logging.info(f"{lamella.current_state.stage.name}: insert needle for reset")

    # needle images
    image_settings.save = True
    image_settings.label = f"sharpen_needle_initial"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    det = calibration.validate_detection(
        microscope,
        settings,
        image_settings,
        lamella=lamella,
        shift_type=(DetectionType.NeedleTip, DetectionType.ImageCentre),
        beam_type=BeamType.ION,
    )

    movement.move_needle_relative_with_corrected_movement(
        microscope=microscope,
        settings=settings,
        dx=det.distance_metres.x,
        dy=-det.distance_metres.y,
        beam_type=BeamType.ION,
    )

    # x_move = movement.x_corrected_needle_movement(det.distance_metres.x)
    # needle.relative_move(x_move)
    # z_distance = -det.distance_metres.y / np.sin(
    #     np.deg2rad(settings["system"]["stage_tilt_flat_to_ion"])
    # )
    # z_move = movement.z_corrected_needle_movement(z_distance, stage.current_position.t)
    # needle.relative_move(z_move)


    image_settings.label = f"sharpen_needle_centre"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    # create sharpening patterns
    windows.open_milling_window(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern=MillingPattern.Sharpen,
        x=0,
        y=0,
    )

    #################################################################################################

    # take reference images
    image_settings.label = f"sharpen_needle_final"
    image_settings.save = True
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    # retract needle
    movement.retract_needle(microscope)

    # reset stage position
    stage_settings = MoveSettings(rotate_compucentric=True)
    stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)
    # self.stage.absolute_move(StagePosition(r=lamella_coordinates.r))
    stage.absolute_move(StagePosition(x=0.0, y=0.0))

    # TODO: replace and test
    # stage_position = StagePosition(x=0, y=0, t=0)
    # movement.safe_absolute_stage_movement(self.microscope, stage_position)

    return lamella


def thin_lamella(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
) -> Lamella:

    # bookkeeping
    image_settings.save_path = lamella.path

    # move to the initial landing coordinates
    movement.safe_absolute_stage_movement(
        microscope=microscope, stage_position=lamella.landing_coordinates,
    )

    # ensure_eucentricity # TODO: Maybe remove, not required?
    windows.ask_user_movement(
        microscope, settings, image_settings, msg_type="eucentric", flat_to_sem=False,
    )

    # rotate_and_tilt_to_thinning_angle
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    movement.move_to_thinning_angle(microscope=microscope, settings=settings)


    # load the reference images
    reference_images = ReferenceImages(
        low_res_eb=lamella.load_reference_image("ref_landing_lamella_low_res_eb"),
        high_res_eb=lamella.load_reference_image("ref_landing_lamella_high_res_eb"),
        low_res_ib=lamella.load_reference_image("ref_landing_lamella_low_res_ib"),
        high_res_ib=lamella.load_reference_image("ref_landing_lamella_high_res_ib"),
    )

    # TODO:
    calibration.correct_stage_drift(
        microscope, settings, image_settings,
        reference_images=reference_images, 
        alignment=(BeamType.ION, BeamType.ION),
        rotate=True, parent_ui=True
    )

    # ensure_eucentricity at thinning angle
    windows.ask_user_movement(
        microscope, settings, image_settings, msg_type="eucentric"
    )

    # lamella images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_med_res"]
    image_settings.save = True
    image_settings.label = f"thin_lamella_high_res"
    acquire.take_reference_images(microscope, image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["high_super_res"]
    image_settings.save = False
    windows.ask_user_movement(
        microscope, settings, image_settings, msg_type="alignment"
    )

    # take reference images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.save = True
    image_settings.label = f"thin_drift_correction_highres"
    acquire.take_reference_images(microscope, image_settings)

    # thin_lamella (align and mill)
    image_settings.resolution = settings["protocol"]["thin_lamella"]["resolution"]
    image_settings.dwell_time = settings["protocol"]["thin_lamella"]["dwell_time"]
    image_settings.hfw = settings["thin_lamella"]["hfw"]

    # mill thin_lamella
    windows.open_milling_window(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern=MillingPattern.Thin,
        x=0, y=0,
    )


    # take reference images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.save = True
    image_settings.label = f"ref_thin_lamella_low_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.save = True
    image_settings.label = f"ref_thin_lamella_high_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_ultra_res"]
    image_settings.save = True
    image_settings.label = f"ref_thin_lamella_super_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    return


def polish_lamella(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
) -> Lamella:

    # bookkeeping
    image_settings.save_path = lamella.path

    # restore state from thinning stage TODO: align?
    reference_images = ReferenceImages(
        low_res_eb=lamella.load_reference_image("ref_thin_lamella_low_res_eb"),
        high_res_eb=lamella.load_reference_image("ref_thin_lamella_high_res_eb"),
        low_res_ib=lamella.load_reference_image("ref_thin_lamella_low_res_ib"),
        high_res_ib=lamella.load_reference_image("ref_thin_lamella_high_res_ib"),
    )

    # TODO: Test
    calibration.correct_stage_drift(
        microscope, settings, image_settings,
        reference_images=reference_images, 
        alignment=(BeamType.ION, BeamType.ION),
        rotate=True, parent_ui=True
    )

    # realign lamella to image centre
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.save = True
    image_settings.label = f"polish_drift_correction_medres"
    acquire.take_reference_images(microscope, image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.save = False

    # confirm
    windows.ask_user_movement(
        microscope, settings, image_settings, msg_type="alignment"
    )

    # # take reference images
    # image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    # image_settings.save = True
    # image_settings.label = f"polish_drift_correction_highres"
    acquire.take_reference_images(microscope, image_settings)

    # polish (align and mill)
    image_settings.resolution = settings["protocol"]["polish_lamella"]["resolution"]
    image_settings.dwell_time = settings["protocol"]["polish_lamella"]["dwell_time"]
    image_settings.hfw = settings["protocol"]["polish_lamella"]["hfw"]

    windows.open_milling_window(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern=MillingPattern.Polish,
        x=0, y=0,
    )

    # take reference images (ultra, super, high)
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_ultra_res"]
    image_settings.save = True
    image_settings.label = f"ref_polish_lamella_ultra_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label = f"ref_polish_lamella_super_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.label = f"ref_polish_lamella_high_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    logging.info(
        f"{lamella.current_state.stage.name}: polish lamella {lamella._petname} complete."
    )

    return lamella


def run_autoliftout_workflow(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    sample: Sample,
    parent_ui=None,
) -> Sample:

    HIGH_THROUGHPUT = settings["system"]["high_throughput"]
    CONFIRM_WORKFLOW_ADVANCE = True

    # autoliftout_workflow
    autoliftout_stages = {
        AutoLiftoutStage.Setup: run_setup_autoliftout,
        AutoLiftoutStage.MillTrench: mill_lamella_trench,
        AutoLiftoutStage.MillJCut: mill_lamella_jcut,
        AutoLiftoutStage.Liftout: liftout_lamella,
        AutoLiftoutStage.Landing: land_lamella,
        AutoLiftoutStage.Reset: reset_needle,
        AutoLiftoutStage.Thinning: thin_lamella,
        AutoLiftoutStage.Polishing: polish_lamella,
    }

    logging.info(f"AutoLiftout Workflow started for {len(sample.postions)} lamellae.")

    # high throughput workflow
    if HIGH_THROUGHPUT:
        for terminal_stage in [
            AutoLiftoutStage.MillTrench,
            AutoLiftoutStage.MillJCut,
        ]:
            for lamella in sample.positions.values():

                while (
                    lamella.current_state.microscope_state.last_completed_stage.value
                    < terminal_stage.value
                ):

                    next_stage = AutoLiftoutStage(
                        lamella.current_state.microscope_state.last_completed_stage.value
                        + 1
                    )

                    # update image settings (save in correct directory)
                    image_settings.save_path = lamella.path

                    # reset to the previous state
                    lamella = start_of_stage_update(
                        microscope, lamella, next_stage=next_stage, parent_ui=parent_ui
                    )

                    # run the next workflow stage
                    lamella = autoliftout_stages[next_stage](
                        microscope=microscope,
                        settings=settings,
                        image_settings=image_settings,
                        lamella=lamella,
                    )

                    # advance workflow
                    sample = end_of_stage_update(microscope, sample, lamella, parent_ui)

    # standard workflow
    for lamella in sample.positions.values():

        while (
            lamella.current_state.microscope_state.last_completed_stage.value
            < AutoLiftoutStage.Reset.value
        ):

            next_stage = AutoLiftoutStage(
                lamella.current_state.microscope_state.last_completed_stage.value + 1
            )
            if CONFIRM_WORKFLOW_ADVANCE:
                msg = f"""Continue Lamella {(lamella._petname)}) from {next_stage.name}?"""
                response = windows.ask_user_interaction(
                    microscope, msg=msg, beam_type=BeamType.ION,
                )
            else:
                response = True

            # update image settings (save in correct directory)
            image_settings.save_path = lamella.path

            if response:

                # reset to the previous state
                lamella = start_of_stage_update(
                    microscope, lamella, next_stage=next_stage, parent_ui=parent_ui
                )

                # run the next workflow stage
                lamella = autoliftout_stages[next_stage](
                    microscope=microscope,
                    settings=settings,
                    image_settings=image_settings,
                    lamella=lamella,
                )

                # advance workflow
                sample = end_of_stage_update(
                    microscope, sample, lamella, parent_ui=parent_ui
                )
            else:
                break  # go to the next sample

    return sample


def end_of_stage_update(
    microscope: SdbMicroscopeClient, sample: Sample, lamella: Lamella, parent_ui=None
) -> Sample:
    """Save the current microscope state configuration to disk, and log that the stage has been completed."""

    # save state information
    lamella.current_state.microscope_state = calibration.get_current_microscope_state(
        microscope=microscope, stage=lamella.current_state.stage,
    )
    lamella.current_state.end_timestamp = datetime.timestamp(datetime.now())

    # write history
    lamella.history.append(lamella.current_state)

    # update and save sample
    sample = update_sample_lamella_data(sample, lamella)

    # update ui
    if parent_ui:
        parent_ui.update_scroll_ui()

    logging.info(f"{lamella._petname} | {lamella.current_state.stage} | FINISHED")

    return sample


def start_of_stage_update(
    microscope: SdbMicroscopeClient,
    lamella: Lamella,
    next_stage: AutoLiftoutStage,
    parent_ui=None,
) -> Lamella:
    """Check the last completed stage and reload the microscope state if required. Log that the stage has started. """
    last_completed_stage = lamella.current_state.microscope_state.last_completed_stage

    # restore to the last state
    if last_completed_stage.value == next_stage.value - 1:

        logging.info(
            f"{lamella._petname} restarting from end of stage: {last_completed_stage.name}"
        )
        calibration.set_microscope_state(
            microscope, lamella.current_state.microscope_state
        )

    # set current state information
    lamella.current_state.stage = next_stage
    lamella.current_state.start_timestamp = datetime.timestamp(datetime.now())
    logging.info(f"{lamella._petname} | {lamella.current_state.stage} | STARTED")

    # update ui
    if parent_ui:
        parent_ui.update_status(lamella=lamella)

    return lamella


def run_thinning_workflow(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    sample: Sample,
) -> Sample:

    # thinning
    for lamella in sample.positions.values():

        if (
            lamella.current_state.microscope_state.last_completed_stage
            == AutoLiftoutStage.Reset
        ):
            lamella = start_of_stage_update(
                microscope, lamella, next_stage=AutoLiftoutStage.Thinning
            )
            thin_lamella(microscope, settings, image_settings, lamella)
            sample = end_of_stage_update(microscope, sample, lamella)

    # polish
    for lamella in sample.positions.values():

        if (
            lamella.current_state.microscope_state.last_completed_stage
            == AutoLiftoutStage.Thinning
        ):
            lamella = start_of_stage_update(
                microscope, lamella, next_stage=AutoLiftoutStage.Polishing
            )
            thin_lamella(microscope, settings, image_settings, lamella)
            sample = end_of_stage_update(microscope, sample, lamella)

    # finish the experiment
    for lamella in sample.positions.values():
        if (
            lamella.current_state.microscope_state.last_completed_stage
            == AutoLiftoutStage.Polishing
        ):
            lamella.current_state.microscope_state.last_completed_stage = (
                AutoLiftoutStage.Finished
            )
            sample = end_of_stage_update(microscope, sample, lamella)

    return sample


def get_current_lamella(
    microscope: SdbMicroscopeClient,
    sample: Sample,
) -> bool:

    if sample.positions:
        select_another_lamella = windows.ask_user_interaction(
            microscope,
            msg=f"Do you want to select another lamella?\n"
            f"{len(sample.positions)} currentlly selected.",
            beam_type=BeamType.ELECTRON,
        )

    else:
        select_another_lamella = True

    return select_another_lamella


def user_select_feature(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    msg: str = "Select the feature.",
):
    """Get the user to centre the beam on the desired feature"""

    # ask user to select feature
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_med_res"]
    image_settings.save = False
    windows.ask_user_movement(
        microscope, settings, image_settings, msg_type="centre_ib", msg=msg,
    )

    return calibration.get_raw_stage_position(microscope)


def select_initial_lamella_positions(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    sample: Sample,
    eucentric_calibration: bool = False,
) -> Lamella:
    """Select the initial sample positions for liftout"""

    # create lamella
    lamella_no = max(sample.positions.keys()) + 1
    lamella = Lamella(sample.path, lamella_no)

    # TODO: replace with auto eucentric calibration
    if eucentric_calibration is False:
        movement.move_to_sample_grid(microscope, settings=settings)
        movement.auto_link_stage(microscope)

        windows.ask_user_movement(
            microscope,
            settings,
            image_settings,
            msg_type="eucentric",
            flat_to_sem=True,
        )
        movement.move_to_trenching_angle(microscope, settings=settings)

    # save lamella coordinates
    lamella.lamella_coordinates = user_select_feature(
        microscope, settings, image_settings, msg="Select a lamella position."
    )

    # save microscope state
    lamella.current_state.microscope_state = calibration.get_current_microscope_state(
        microscope=microscope, stage=lamella.current_state.stage,
    )

    # take reference images
    image_settings = acquire.update_image_settings_v3(
        settings=settings,
        hfw=settings["calibration"]["reference_images"]["hfw_med_res"],
        save=True,
        save_path=lamella.path,
        label=f"ref_lamella_low_res",
    )
    acquire.take_reference_images(microscope, image_settings=image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label = "ref_lamella_high_res"
    acquire.take_reference_images(microscope, image_settings=image_settings)

    return lamella


def select_landing_positions(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    sample: Sample,
):
    """Select landing positions for autoliftout"""

    ####################################
    # # move to landing grid
    movement.move_to_landing_grid(microscope, settings=settings)
    # movement.auto_link_stage(self.microscope, hfw=900e-6)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_low_res"]
    windows.ask_user_movement(
        microscope, settings, image_settings, msg_type="eucentric", flat_to_sem=False
    )
    ####################################

    # select corresponding sample landing positions
    for lamella in sample.positions.values():

        # check if landing position already selected? so it doesnt overwrite
        if lamella.landing_selected is False:
            lamella = select_landing_sample_positions(lamella)

            sample = update_sample_lamella_data(sample, lamella)

    return sample


def update_sample_lamella_data(sample: Sample, lamella: Lamella) -> Sample:

    sample.positions[lamella._number] = lamella
    sample.save()
    return sample


def select_landing_sample_positions(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
) -> Lamella:
    """Select the landing coordinates for a lamella."""
    logging.info(f"Selecting Landing Position: {lamella._petname}")

    # update image path
    image_settings.save_path = lamella.path

    # select landing coordinates
    lamella.landing_coordinates = user_select_feature(
        microscope,
        settings,
        image_settings,
        msg=f"Select the landing coordinate for {lamella._petname}.",
    )

    # mill the landing edge flat
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.beam_type = BeamType.ION
    image_settings.save = False

    windows.open_milling_window(
        microscope=microscope,
        settings=settings,
        image_settings=image_settings,
        milling_pattern=MillingPattern.Flatten,
        x=0,
        y=0,
    )

    # take reference images
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_med_res"]
    image_settings.save = True
    image_settings.label = "ref_landing_low_res"
    acquire.take_reference_images(microscope, image_settings=image_settings)

    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.label = "ref_landing_high_res"
    acquire.take_reference_images(microscope, image_settings)

    lamella.landing_selected = True

    return lamella


def select_lamella_positions(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    sample: Sample,
    parent_ui=None,
):

    select_another = get_current_lamella(
        microscope, sample
    )

    # allow the user to select additional lamella positions
    eucentric_calibration = False
    while select_another:

        lamella = select_initial_lamella_positions(
            microscope, settings, image_settings, sample, eucentric_calibration
        )

        # save lamella data
        sample = update_sample_lamella_data(sample, lamella)

        # select another?
        select_another = get_current_lamella(
            microscope, sample
        )

        # state variable
        eucentric_calibration = True

        # update ui
        if parent_ui:
            parent_ui.update_scroll_ui()

    # select landing positions
    select_landing_positions(microscope, settings, image_settings, sample)

    # finish setup
    finish_setup_autoliftout(microscope, sample, parent_ui)

    return sample


def finish_setup_autoliftout(
    microscope: SdbMicroscopeClient, sample: Sample, parent_ui=None
):
    """Finish the setup stage for autolifout/autolamella"""

    # reset microscope coordinate system
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)

    logging.info(f"Selected {len(sample.positions)} lamella for autoliftout.")
    logging.info(f"Path: {sample.path}")
    logging.info(f"INIT | {AutoLiftoutStage.Setup.name} | FINISHED")

    if parent_ui:
        parent_ui.update_scroll_ui()
        parent_ui.pushButton_autoliftout.setEnabled(True)
        parent_ui.pushButton_thinning.setEnabled(True)


def run_setup_autoliftout(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    sample: Sample,
    parent_ui=None,
) -> Sample:

    logging.info(f"INIT | {AutoLiftoutStage.Setup.name} | STARTED")

    # move to the initial sample grid position
    movement.move_to_sample_grid(microscope, settings)

    # initial image settings
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_low_res"]
    image_settings.beam_type = BeamType.ELECTRON
    image_settings.save = True
    image_settings.save_path = sample.path
    image_settings.label = "grid"
    # NOTE: can't take ion beam image with such a high hfw, will default down to max ion beam hfw
    acquire.new_image(microscope, image_settings)

    # sputter platinum to protect grid and prevent charging...
    fibsem_utils.sputter_platinum_on_whole_sample_grid(
        microscope, settings, image_settings
    )

    # reference images
    image_settings.label = "grid_Pt"
    acquire.take_reference_images(microscope, image_settings)

    # check if focus is good enough
    ret = calibration.validate_focus(microscope, settings, image_settings, link=False)

    if ret is False:
        windows.ask_user_interaction(
            microscope,
            msg="The AutoFocus routine has failed, please correct the focus manually.",
            beam_type=BeamType.ELECTRON,
        )

    # select the lamella and landing positions
    sample = select_lamella_positions(
        microscope, settings, image_settings, sample, parent_ui
    )

    return sample


def validate_needle_insertion(microscope: SdbMicroscopeClient, settings: dict) -> None:
    
    # move needle to liftout start position
    ret = calibration.validate_stage_height_for_needle_insertion(
        microscope, settings
    )

    while ret is False:
        windows.ask_user_interaction(
            microscope,
            msg="""The system has identified the distance between the sample and the pole piece is less than 3.7mm. "
            "The needle will contact the sample, and it is unsafe to insert the needle. "
            "\nPlease manually refocus and link the stage, then press OK to continue. """,
            beam_type=BeamType.ELECTRON,
        )
        
        ret = calibration.validate_stage_height_for_needle_insertion(
            microscope, settings
        )