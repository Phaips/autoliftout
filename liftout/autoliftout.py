import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import (
    CoordinateSystem, ManipulatorCoordinateSystem)
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         Rectangle,
                                                         StagePosition)
from fibsem import acquire, alignment, calibration, movement
from fibsem import utils as fibsem_utils
from fibsem import validation
from fibsem.acquire import BeamType
from fibsem.detection.detection import DetectionFeature, DetectionType
from fibsem.imaging import masks
from fibsem.imaging import utils as image_utils
from fibsem.structures import (BeamType, MicroscopeSettings, MicroscopeState,
                               Point)
from fibsem.ui import windows as fibsem_ui_windows

from liftout import actions, patterning
from liftout.gui.milling_window import GUIMillingWindow
from liftout.patterning import MillingPattern
from liftout.sample import (AutoLiftoutStage, Lamella, ReferenceImages, Sample,
                            get_reference_images)
from liftout.structures import ReferenceHFW

# autoliftout workflow functions

# functional mill trench
def mill_lamella_trench(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
):

    # bookkeeping
    settings.image.save_path = lamella.path

    # move to lamella position
    calibration.set_microscope_state(microscope, lamella.lamella_state)

    # Take an ion beam image at the *milling current*
    settings.image.hfw = ReferenceHFW.Super.value

    ######

    # mill_trenches
    open_milling_window(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Trench,
        point=Point(0, 0),
    )

    # discharge check
    calibration.auto_discharge_beam(microscope, settings.image)

    acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=settings.image,
        hfws=[ReferenceHFW.Medium.value, ReferenceHFW.Super.value],
        label="ref_trench",
    )

    return lamella


# functional jcut
def mill_lamella_jcut(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    settings.image.save_path = lamella.path

    # reference images of milled trenches
    hfws = [ReferenceHFW.Medium.value, ReferenceHFW.Super.value]
    reference_images = acquire.take_set_of_reference_images(
        microscope, settings.image, hfws=hfws, label="ref_trench"
    )

    # move flat to electron beam
    movement.move_flat_to_beam(microscope, settings, beam_type=BeamType.ELECTRON)

    # correct drift using reference images..
    alignment.correct_stage_drift(
        microscope,
        settings,
        reference_images,
        alignment=(BeamType.ION, BeamType.ELECTRON),
        rotate=True,
        use_ref_mask=True,
    )

    # adjust for relative shift between beams
    RELATIVE_ELECTRON_ION_SHIFT = settings.protocol["initial_position"][
        "relative_beam_shift_x"
    ]
    movement.move_stage_relative_with_corrected_movement(
        microscope,
        settings,
        dx=RELATIVE_ELECTRON_ION_SHIFT,
        dy=0,
        beam_type=BeamType.ELECTRON,
    )

    # confirm
    fibsem_ui_windows.ask_user_movement(
        microscope,
        settings,
        msg_type="eucentric",
        msg="Confirm lamella is centred in Ion Beam",
    )

    # take reference, pre-tilting
    settings.image.beam_type = BeamType.ION
    ref_ib = acquire.new_image(microscope, settings.image)

    # move to jcut angle
    TILT_DEGREES = settings.protocol["jcut"]["jcut_angle"]
    stage_settings = MoveSettings(rotate_compucentric=True)
    microscope.specimen.stage.relative_move(
        StagePosition(t=np.deg2rad(TILT_DEGREES)), stage_settings
    )

    # mask ref, cosine stretch
    new_ib = acquire.new_image(microscope, settings.image)
    mask = masks.create_lamella_mask(
        ref_ib, settings.protocol["lamella"], scale=2.5, use_trench_height=True
    )
    alignment.align_using_reference_images(
        microscope,
        settings,
        image_utils.cosine_stretch(ref_ib, TILT_DEGREES),
        new_ib,
        ref_mask=mask,
    )

    ## MILL_JCUT
    # now we are at the angle for jcut, perform jcut
    settings.image.hfw = ReferenceHFW.Super.value
    open_milling_window(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.JCut,
        point=Point(0, 0),
    )

    # take reference images of the jcut (tilted)
    hfws = [ReferenceHFW.Medium.value, ReferenceHFW.Super.value]
    reference_images = acquire.take_set_of_reference_images(
        microscope, settings.image, hfws=hfws, label="ref_jcut_tilt"
    )

    # move to flat eb
    movement.move_flat_to_beam(microscope, settings, BeamType.ELECTRON)

    # realign

    # TODO: create helper for this aligned tilt correction
    # mask ref, cosine stretch
    settings.image.hfw = ReferenceHFW.Super.value
    new_ib = acquire.new_image(microscope, settings.image)
    mask = masks.create_lamella_mask(
        ref_ib, settings.protocol["lamella"], scale=2.5, use_trench_height=True
    )
    alignment.align_using_reference_images(
        microscope,
        settings,
        reference_images.high_res_ib,
        image_utils.cosine_stretch(new_ib, TILT_DEGREES),
        ref_mask=mask,
    )

    # take reference images of the jcut
    hfws = [ReferenceHFW.Medium.value, ReferenceHFW.Super.value]
    reference_images = acquire.take_set_of_reference_images(
        microscope, settings.image, hfws=hfws, label="ref_jcut"
    )

    return lamella


def liftout_lamella(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    settings.image.save_path = lamella.path

    # convenience
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # get ready to do liftout by moving to liftout angle (flat to eb)
    actions.move_to_liftout_angle(microscope, settings)

    reference_images = get_reference_images(lamella, "ref_jcut")

    settings.image.beam_type = BeamType.ELECTRON
    settings.image.save = False
    new_image = acquire.new_image(microscope, settings.image)
    alignment.align_using_reference_images(
        microscope, settings, reference_images.high_res_ib, new_image
    )

    # confirm
    fibsem_ui_windows.ask_user_movement(
        microscope,
        settings,
        msg_type="eucentric",
        msg="Confirm lamella is centred in Ion Beam",
    )

    # reference images for needle location
    settings.image.save = True
    settings.image.hfw = ReferenceHFW.High.value
    settings.image.label = f"ref_needle_liftout"
    acquire.take_reference_images(microscope, settings.image)

    # land needle on lamella
    lamella = land_needle_on_milled_lamella(microscope, settings, lamella)

    # sputter platinum
    fibsem_utils.sputter_platinum(
        microscope,
        settings.protocol["platinum"],
        whole_grid=False,
        default_application_file=settings.system.application_file,
    )
    logging.info(
        f"{lamella.current_state.stage.name}: lamella to needle welding complete."
    )

    settings.image.save = True
    settings.image.hfw = settings.protocol["platinum"]["weld"]["hfw"]
    settings.image.label = f"needle_landed_Pt_sputter"
    acquire.take_reference_images(microscope, settings.image)

    # jcut sever pattern
    open_milling_window(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Sever,
        point=Point(
            x=settings.protocol["lamella"]["lamella_width"] / 2, y=0
        ),  # half the lamella width
    )

    # take reference images
    settings.image.save = True
    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.label = f"jcut_sever"
    acquire.take_reference_images(microscope, settings.image)

    # Raise needle 30um from trench
    logging.info(
        f"{lamella.current_state.stage.name}: start removing needle from trench"
    )
    for i in range(3):
        z_move_out_from_trench = movement.z_corrected_needle_movement(
            10e-6, stage.current_position.t
        )
        needle.relative_move(z_move_out_from_trench)
        settings.image.label = f"liftout_trench_{i}"
        acquire.take_reference_images(microscope, settings.image)
        logging.info(
            f"{lamella.current_state.stage.name}: removing needle from trench at {z_move_out_from_trench} ({i + 1}/3)"
        )
        time.sleep(1)

    # reference images after liftout complete
    settings.image.label = f"ref_liftout"
    acquire.take_reference_images(microscope, settings.image)

    # move needle to park position
    movement.retract_needle(microscope)

    return lamella


def land_needle_on_milled_lamella(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    settings.image.save_path = lamella.path

    # validate needle insertion conditions
    validate_needle_insertion(
        microscope, settings.system.stage.needle_stage_height_limit
    )

    # get updated needle insertion position
    insert_position = fibsem_utils.get_updated_needle_insertion_position(lamella.base_path)

    # insert the needle for liftout
    actions.move_needle_to_liftout_position(microscope, insert_position)

    # # # reference images
    settings.image.hfw = ReferenceHFW.High.value
    settings.image.save = True
    settings.image.label = f"needle_liftout_start_position"
    acquire.take_reference_images(microscope, settings.image)

    # take image

    # measure brightness
    BRIGHTNESS_FACTOR = 1.2
    settings.image.beam_type = BeamType.ION
    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.label = f"needle_liftout_land"
    settings.image.save = True
    settings.image.gamma.enabled = False
    reduced_area = Rectangle(0.4, 0.45, 0.2, 0.1)
    ib_image = acquire.new_image(microscope, settings.image, reduced_area)
    previous_brightness = image_utils.measure_brightness(ib_image)

    brightness_history = [previous_brightness]
    MEAN_BRIGHTNESS = np.mean(brightness_history)

    iteration_count = 0
    MAX_ITERATIONS = 10

    while True:

        # move needle down
        microscope.specimen.manipulator.set_default_coordinate_system(
            ManipulatorCoordinateSystem.STAGE
        )

        dx = 1.0e-6
        dy = 0.0e-6
        movement.move_needle_relative_with_corrected_movement(
            microscope, dx=dx, dy=dy, beam_type=BeamType.ION
        )

        # calculate brightness
        settings.image.label = f"bright_{fibsem_utils.current_timestamp()}"
        ib_image = acquire.new_image(
            microscope, settings.image, reduced_area=reduced_area
        )
        brightness = image_utils.measure_brightness(ib_image)

        logging.info(
            f"iter: {iteration_count}: brightness: {brightness}, prevs: {previous_brightness}, MEAN BRIGHTNESS: {MEAN_BRIGHTNESS}"
        )

        if brightness > MEAN_BRIGHTNESS * BRIGHTNESS_FACTOR:
            # needle has landed...
            logging.info("THRESHOLD REACHED STOPPPING")

            # check with user?
            response = fibsem_ui_windows.ask_user_interaction(
                microscope,
                msg="Has the needle landed on the lamella? \nPress Yes to continue, or No to redo the final movement",
                beam_type=BeamType.ION,
            )

            if response:
                break

        previous_brightness = brightness
        brightness_history.append(brightness)
        MEAN_BRIGHTNESS = np.mean(brightness_history)

        iteration_count += 1
        if iteration_count >= MAX_ITERATIONS:
            break

    settings.image.gamma.enabled = True

    acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[ReferenceHFW.High.value, ReferenceHFW.Super.value],
        label="ref_needle_liftout_landed",
    )

    return lamella


def land_lamella(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    settings.image.save_path = lamella.path
    settings.image.save = False

    # conveienence
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # move to landing coordinate
    calibration.set_microscope_state(microscope, lamella.landing_state)

    calibration.auto_link_stage(microscope)

    # confirm eucentricity
    fibsem_ui_windows.ask_user_movement(
        microscope, settings, msg_type="eucentric"
    )

    logging.info(
        f"{lamella.current_state.stage.name}: initial landing calibration complete."
    )

    ############################## LAND_LAMELLA ##############################
    validate_needle_insertion(
        microscope, settings.system.stage.needle_stage_height_limit
    )

    # get updated needle insertion position
    insert_position = fibsem_utils.get_updated_needle_insertion_position(lamella.base_path)

    actions.move_needle_to_landing_position(microscope, insert_position)
    # TODO: move lower than eucentric to make sure landing

    # needle starting position
    settings.image.hfw = ReferenceHFW.High.value
    settings.image.beam_type = BeamType.ELECTRON
    settings.image.save = True
    settings.image.label = f"landing_needle_start_position"
    acquire.take_reference_images(microscope, settings.image)

    # repeat final movement until user confirms landing
    response = False
    while response is False:
        #### X-MOVE
        settings.image.hfw = ReferenceHFW.Super.value
        settings.image.beam_type = BeamType.ION
        settings.image.save = True
        settings.image.label = f"landing_needle"
        ref_eb, ref_ib = acquire.take_reference_images(microscope, settings.image)

        det = fibsem_ui_windows.detect_features(
            microscope=microscope,
            settings=settings,
            lamella=lamella,
            ref_image=ref_ib,
            features=[
                DetectionFeature(DetectionType.LamellaEdge, None),
                DetectionFeature(DetectionType.LandingPost, None),
            ],
            validate=True,
        )

        movement.move_needle_relative_with_corrected_movement(
            microscope=microscope,
            dx=det.distance_metres.x,
            dy=0,
            beam_type=BeamType.ION,
        )

        # final reference images
        settings.image.hfw = ReferenceHFW.Super.value
        settings.image.beam_type = BeamType.ELECTRON
        settings.image.save = True
        settings.image.label = f"landing_lamella_final_weld_highres"
        acquire.take_reference_images(
            microscope=microscope, image_settings=settings.image
        )

        response = fibsem_ui_windows.ask_user_interaction(
            microscope,
            msg="Has the lamella landed on the post? \nPress Yes to continue, or No to redo the final movement",
            beam_type=BeamType.ION,
        )

    #################################################################################################

    ############################## WELD TO LANDING POST #############################################

    open_milling_window(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Weld,
        point=Point(),
    )

    # final reference images
    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.save = True
    settings.image.label = f"landing_lamella_final_weld_high_res"
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    #################################################################################################

    ###################################### CUT_OFF_NEEDLE ######################################

    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.beam_type = BeamType.ION
    settings.image.save = True
    settings.image.label = "landing_lamella_pre_cut"

    # # TODO: can eliminate this if the lamella lands in the centre... just manually calc it
    # # cut off needle
    # open_milling_window(
    #     microscope=microscope,
    #     settings=settings,
    #     milling_pattern=MillingPattern.Cut,
    #     point=Point(),
    # )

    # back out needle from lamella , no cut required?

    for i in range(5):

        # move needle back
        movement.move_needle_relative_with_corrected_movement(
            microscope=microscope,
            dx=-1e-6,
            dy=0,
            beam_type=BeamType.ION,
        )

        # take image 
        acquire.new_image(microscope, settings.image)


    # TODO: if this works, remove references to "cut"
    # TODO: if this works, can probably remove the slow removal below too...

    ################################### REMOVE_NEEDLE ##########################################

    # reference images
    acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=settings.image,
        hfws=[ReferenceHFW.High.value, ReferenceHFW.Super.value],
        label="landing_lamella_post_cut",
    )

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
    acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=settings.image,
        hfws=[ReferenceHFW.High.value, ReferenceHFW.Super.value],
        label="ref_landing_lamella",
    )

    return lamella


def reset_needle(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    settings.image.save_path = lamella.path

    # convienence
    stage = microscope.specimen.stage

    # move sample stage out
    actions.move_sample_stage_out(microscope, settings)

    ###################################### SHARPEN_NEEDLE ######################################

    validation.validate_stage_height_for_needle_insertion(
        microscope, settings.system.stage.needle_stage_height_limit
    )

    # get updated needle insertion position
    insert_position = fibsem_utils.get_updated_needle_insertion_position(lamella.base_path)

    # move needle in
    actions.move_needle_to_reset_position(microscope, insert_position)

    # explicitly set the coordinate system
    microscope.specimen.manipulator.set_default_coordinate_system(
        ManipulatorCoordinateSystem.STAGE
    )

    # needle images
    settings.image.save = True
    settings.image.label = f"sharpen_needle_start_position"
    ref_eb, ref_ib = acquire.take_reference_images(
        microscope=microscope, image_settings=settings.image
    )
    settings.image.beam_type = BeamType.ION

    # TODO: move needle to the centre, because it has been cut off...
    det = fibsem_ui_windows.detect_features(
        microscope=microscope,
        settings=settings,
        ref_image=ref_ib,
        features=[
            DetectionFeature(DetectionType.NeedleTip, None),
            DetectionFeature(DetectionType.ImageCentre, None),
        ],
        validate=True,
    )

    movement.move_needle_relative_with_corrected_movement(
        microscope=microscope,
        dx=det.distance_metres.x,
        dy=-det.distance_metres.y,
        beam_type=BeamType.ION,
    )

    # TODO: validate this movement

    # create sharpening patterns
    open_milling_window(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Sharpen,
        point=Point(),
    )

    #################################################################################################

    # reset the "eucentric position" for the needle, centre needle in both views
    calibration.align_needle_to_eucentric_position(microscope, settings, lamella, validate=True)

    # take reference images
    settings.image.label = f"ref_reset"
    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.save = True
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    # retract needle
    movement.retract_needle(microscope)

    # reset stage position
    move_settings = MoveSettings(rotate_compucentric=True)
    stage.absolute_move(StagePosition(t=np.deg2rad(0)), move_settings)
    stage.absolute_move(StagePosition(x=0.0, y=0.0), move_settings)

    # TODO: test this
    if lamella.landing_selected:
        calibration.set_microscope_state(microscope, lamella.landing_state)

    return lamella



def thin_lamella(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    settings.image.save_path = lamella.path

    # move to the initial landing coordinates
    calibration.set_microscope_state(microscope, lamella.landing_state)

    # ensure_eucentricity # TODO: Maybe remove, not required?
    fibsem_ui_windows.ask_user_movement(
        microscope, settings, msg_type="eucentric")

    # rotate_and_tilt_to_thinning_angle
    settings.image.hfw = ReferenceHFW.High.value
    actions.move_to_thinning_angle(microscope=microscope, protocol=settings.protocol)

    # load the reference images
    reference_images = get_reference_images(lamella, label="ref_landing_lamella")

    # TODO: test
    alignment.correct_stage_drift(
        microscope,
        settings,
        reference_images=reference_images,
        alignment=(BeamType.ION, BeamType.ELECTRON),
        rotate=True,
        use_ref_mask=False,
    )

    # ensure_eucentricity at thinning angle
    # confirm
    fibsem_ui_windows.ask_user_movement(
        microscope,
        settings,
        msg_type="eucentric",
        msg="Confirm lamella is centred in Ion Beam",
    )

    # lamella images
    settings.image.hfw = ReferenceHFW.Medium.value
    settings.image.save = True
    settings.image.label = f"thin_lamella_high_res"
    acquire.take_reference_images(microscope, settings.image)

    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.save = False
    fibsem_ui_windows.ask_user_movement(microscope, settings, msg_type="alignment")

    # take reference images
    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.save = True
    settings.image.label = f"thin_drift_correction_highres"
    acquire.take_reference_images(microscope, settings.image)

    # thin_lamella (align and mill)
    settings.image.resolution = settings.protocol["thin_lamella"]["resolution"]
    settings.image.dwell_time = settings.protocol["thin_lamella"]["dwell_time"]
    settings.image.hfw = settings.protocol["thin_lamella"]["hfw"]

    # mill thin_lamella
    open_milling_window(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Thin,
        point=Point(),
    )

    # # take reference images
    reference_images = acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[ReferenceHFW.High.value, ReferenceHFW.Super.value],
        label="ref_thin_lamella",
    )

    settings.image.hfw = ReferenceHFW.Ultra.value
    settings.image.save = True
    settings.image.label = f"ref_thin_lamella_super_res"
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    return


def polish_lamella(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    settings.image.save_path = lamella.path

    # # restore state from thinning stage
    reference_images = get_reference_images(lamella, "ref_thin_lamella")

    # TODO: Test, probs only needs 1 step
    # alignment.correct_stage_drift(
    #     microscope,
    #     settings,
    #     reference_images=reference_images,
    #     alignment=(BeamType.ION, BeamType.ION),
    #     rotate=Fa;,
    # )

    settings.image = fibsem_utils.match_image_settings(
        reference_images.high_res_ib, settings.image, BeamType.ION
    )

    # settings.image.beam_type = BeamType.ION
    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.save = True
    settings.image.label = "restore_from_thin_lamella"
    new_ib = acquire.new_image(microscope, settings.image)
    alignment.align_using_reference_images(
        microscope,
        settings,
        ref_image=reference_images.high_res_ib,
        new_image=new_ib,
        ref_mask=None,
    )

    # confirm
    fibsem_ui_windows.ask_user_movement(
        microscope,
        settings,
        msg_type="eucentric",
        msg="Confirm lamella is centred in Ion Beam",
    )

    # realign lamella to image centre
    settings.image.hfw = ReferenceHFW.High.value
    settings.image.save = True
    settings.image.label = f"polish_drift_correction_highres"
    acquire.take_reference_images(microscope, settings.image)

    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.save = False

    # confirm
    fibsem_ui_windows.ask_user_movement(microscope, settings, msg_type="alignment")

    # # take reference images
    acquire.take_reference_images(microscope, settings.image)

    # polish (align and mill)
    settings.image.resolution = settings.protocol["polish_lamella"]["resolution"]
    settings.image.dwell_time = settings.protocol["polish_lamella"]["dwell_time"]
    settings.image.hfw = settings.protocol["polish_lamella"]["hfw"]

    open_milling_window(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Polish,
        point=Point(),
    )

    # take reference images (ultra, super, high)

    hfws = [ReferenceHFW.High.value, ReferenceHFW.Super.value]
    reference_images = acquire.take_set_of_reference_images(
        microscope, settings.image, hfws, "ref_thin_lamella"
    )

    settings.image.hfw = ReferenceHFW.Ultra.value
    settings.image.save = True
    settings.image.label = f"ref_polish_lamella_ultra_res"
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    return lamella


def run_autoliftout_workflow(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    sample: Sample,
    parent_ui=None,
) -> Sample:

    HIGH_THROUGHPUT = True
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

    logging.info(f"AutoLiftout Workflow started for {len(sample.positions)} lamellae.")

    # high throughput workflow
    if HIGH_THROUGHPUT:
        for terminal_stage in [
            AutoLiftoutStage.MillTrench,
            AutoLiftoutStage.MillJCut,
        ]:
            lamella: Lamella
            for lamella in sample.positions.values():

                while lamella.current_state.stage.value < terminal_stage.value:

                    next_stage = AutoLiftoutStage(lamella.current_state.stage.value + 1)

                    # update image settings (save in correct directory)
                    settings.image.save_path = lamella.path

                    # reset to the previous state
                    lamella = start_of_stage_update(
                        microscope, lamella, next_stage=next_stage, parent_ui=parent_ui
                    )

                    # run the next workflow stage
                    lamella = autoliftout_stages[next_stage](
                        microscope=microscope, settings=settings, lamella=lamella,
                    )

                    # advance workflow
                    sample = end_of_stage_update(microscope, sample, lamella, parent_ui)

    # standard workflow
    lamella: Lamella
    for lamella in sample.positions.values():

        while lamella.current_state.stage.value < AutoLiftoutStage.Reset.value:

            next_stage = AutoLiftoutStage(lamella.current_state.stage.value + 1)
            if CONFIRM_WORKFLOW_ADVANCE:
                msg = (
                    f"""Continue Lamella {(lamella._petname)} from {next_stage.name}?"""
                )
                response = fibsem_ui_windows.ask_user_interaction(
                    microscope, msg=msg, beam_type=BeamType.ION,
                )
            else:
                response = True

            # update image settings (save in correct directory)
            settings.image.save_path = lamella.path

            if response:

                # reset to the previous state
                lamella = start_of_stage_update(
                    microscope, lamella, next_stage=next_stage, parent_ui=parent_ui
                )

                # run the next workflow stage
                lamella = autoliftout_stages[next_stage](
                    microscope=microscope, settings=settings, lamella=lamella,
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
        microscope=microscope
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
    """Check the last completed stage and reload the microscope state if required. Log that the stage has started."""
    last_completed_stage = lamella.current_state.stage

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
    settings: MicroscopeSettings,
    sample: Sample,
    parent_ui=None,
) -> Sample:

    # thinning
    lamella: Lamella
    for lamella in sample.positions.values():

        if lamella.current_state.stage == AutoLiftoutStage.Reset:
            lamella = start_of_stage_update(
                microscope,
                lamella,
                next_stage=AutoLiftoutStage.Thinning,
                parent_ui=parent_ui,
            )
            thin_lamella(microscope, settings, lamella)
            sample = end_of_stage_update(
                microscope, sample, lamella, parent_ui=parent_ui
            )

    # polish
    for lamella in sample.positions.values():

        if lamella.current_state.stage == AutoLiftoutStage.Thinning:
            lamella = start_of_stage_update(
                microscope,
                lamella,
                next_stage=AutoLiftoutStage.Polishing,
                parent_ui=parent_ui,
            )
            polish_lamella(microscope, settings, lamella)
            sample = end_of_stage_update(
                microscope, sample, lamella, parent_ui=parent_ui
            )

    # finish the experiment
    for lamella in sample.positions.values():
        if lamella.current_state.stage == AutoLiftoutStage.Polishing:
            lamella.current_state.stage = AutoLiftoutStage.Finished
            sample = end_of_stage_update(
                microscope, sample, lamella, parent_ui=parent_ui
            )

    return sample


def get_current_lamella(microscope: SdbMicroscopeClient, sample: Sample,) -> bool:

    if sample.positions:
        select_another_lamella = fibsem_ui_windows.ask_user_interaction(
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
    settings: MicroscopeSettings,
    msg: str = "Select the feature.",
) -> MicroscopeState:
    """Get the user to centre the beam on the desired feature"""

    # ask user to select feature
    settings.image.hfw = ReferenceHFW.High.value
    settings.image.save = False
    fibsem_ui_windows.ask_user_movement(
        microscope, settings, msg_type="eucentric", msg=msg,
    )

    return calibration.get_current_microscope_state(microscope)


def select_initial_lamella_positions(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    sample: Sample,
    eucentric_calibration: bool = False,
) -> Lamella:
    """Select the initial sample positions for liftout"""

    # create lamella
    if sample.positions:
        lamella_no = max(sample.positions.keys()) + 1
    else:
        lamella_no = 1
    lamella = Lamella(sample.path, lamella_no)

    # TODO: replace with auto eucentric calibration
    if eucentric_calibration is False:
        actions.move_to_sample_grid(
            microscope, settings=settings, protocol=settings.protocol
        )
        movement.move_flat_to_beam(microscope, settings, BeamType.ELECTRON)
        calibration.auto_link_stage(microscope)
        fibsem_ui_windows.ask_user_movement(microscope, settings, msg_type="eucentric")
        actions.move_to_trenching_angle(microscope, settings=settings)

    # save lamella coordinates
    lamella.lamella_state = user_select_feature(
        microscope, settings, msg="Select a lamella position."
    )

    # save microscope state
    lamella.current_state.microscope_state = calibration.get_current_microscope_state(
        microscope=microscope,
    )

    settings.image.hfw = ReferenceHFW.Medium.value
    settings.image.save = True
    settings.image.save_path = lamella.path

    acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[ReferenceHFW.Medium.value, ReferenceHFW.Super.value],
        label="ref_lamella",
    )

    return lamella


def select_landing_positions(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, sample: Sample,
):
    """Select landing positions for autoliftout"""

    ####################################
    # # move to landing grid
    actions.move_to_landing_grid(microscope, settings, settings.protocol)

    settings.image.save = False

    settings.image.hfw = ReferenceHFW.Low.value
    fibsem_ui_windows.ask_user_movement(
        microscope, settings, msg_type="eucentric"
    )
    ####################################

    # select corresponding sample landing positions
    lamella: Lamella
    for lamella in sample.positions.values():

        # check if landing position already selected? so it doesnt overwrite
        if lamella.landing_selected is False:
            lamella = select_landing_sample_positions(microscope, settings, lamella)

            sample = update_sample_lamella_data(sample, lamella)

    return sample


def update_sample_lamella_data(sample: Sample, lamella: Lamella) -> Sample:

    sample.positions[lamella._number] = lamella
    sample.save()
    return sample


def select_landing_sample_positions(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:
    """Select the landing coordinates for a lamella."""
    logging.info(f"Selecting Landing Position: {lamella._petname}")

    # update image path
    settings.image.save_path = lamella.path

    # select landing coordinates
    lamella.landing_state = user_select_feature(
        microscope,
        settings,
        msg=f"Select the landing coordinate for {lamella._petname}.",
    )

    # mill the landing edge flat
    settings.image.hfw = ReferenceHFW.High.value
    settings.image.beam_type = BeamType.ION
    settings.image.save = False

    open_milling_window(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Flatten,
        point=Point(),
    )

    # take reference images
    acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[ReferenceHFW.Medium.value, ReferenceHFW.High.value],
        label="ref_landing",
    )

    lamella.landing_selected = True

    return lamella


def select_lamella_positions(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    sample: Sample,
    parent_ui=None,
):

    select_another = get_current_lamella(microscope, sample)

    # allow the user to select additional lamella positions
    eucentric_calibration = False
    while select_another:

        lamella = select_initial_lamella_positions(
            microscope, settings, sample, eucentric_calibration
        )

        # save lamella data
        sample = update_sample_lamella_data(sample, lamella)

        # select another?
        select_another = get_current_lamella(microscope, sample)

        # state variable
        eucentric_calibration = True

        # update ui
        if parent_ui:
            parent_ui.update_scroll_ui()

    # select landing positions
    select_landing_positions(microscope, settings, sample)

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
    settings: MicroscopeSettings,
    sample: Sample,
    parent_ui=None,
) -> Sample:

    logging.info(f"INIT | {AutoLiftoutStage.Setup.name} | STARTED")

    # move to the initial sample grid position
    actions.move_to_sample_grid(microscope, settings, settings.protocol)

    # initial image settings
    settings.image.hfw = ReferenceHFW.Low.value
    settings.image.beam_type = BeamType.ELECTRON
    settings.image.save = True
    settings.image.save_path = sample.path
    settings.image.label = "grid"
    # NOTE: can't take ion beam image with such a high hfw, will default down to max ion beam hfw
    acquire.new_image(microscope, settings.image)

    # sputter platinum to protect grid and prevent charging...
    sputter_platinum_on_whole_sample_grid(
        microscope, settings, settings.protocol
    )

    # reference images
    settings.image.label = "grid_Pt"
    acquire.take_reference_images(microscope, settings.image)

    # check if focus is good enough
    ret = validation.validate_focus(microscope, settings.system.electron, link=False)

    if ret is False:
        fibsem_ui_windows.ask_user_interaction(
            microscope,
            msg="The AutoFocus routine has failed, please correct the focus manually.",
            beam_type=BeamType.ELECTRON,
        )

    # select the lamella and landing positions
    sample = select_lamella_positions(microscope, settings, sample, parent_ui)

    return sample


def validate_needle_insertion(
    microscope: SdbMicroscopeClient, needle_stage_height_limit: float = 3.7e-3
) -> None:

    # move needle to liftout start position
    ret = validation.validate_stage_height_for_needle_insertion(
        microscope, needle_stage_height_limit
    )

    while ret is False:
        fibsem_ui_windows.ask_user_interaction(
            microscope,
            msg=f"""The system has identified the distance between the sample and the pole piece is less than {needle_stage_height_limit * 1000}mm. "
            "The needle will contact the sample, and it is unsafe to insert the needle. "
            "\nPlease manually refocus and link the stage, then press OK to continue. """,
            beam_type=BeamType.ELECTRON,
        )

        ret = validation.validate_stage_height_for_needle_insertion(
            microscope, needle_stage_height_limit
        )

def sputter_platinum_on_whole_sample_grid(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    protocol: dict,
) -> None:
    """Move to the sample grid and sputter platinum over the whole grid"""

    # Whole-grid platinum deposition
    response = fibsem_ui_windows.ask_user_interaction(
        microscope=microscope,
        msg="Do you want to sputter the whole \nsample grid with platinum?",
        beam_type=BeamType.ELECTRON,
    )

    if response:
        actions.move_to_sample_grid(microscope, settings, protocol)
        fibsem_utils.sputter_platinum(
            microscope=microscope,
            protocol=protocol["platinum"],
            whole_grid=True,
            default_application_file=settings.system.application_file,
        )

    return


def open_milling_window(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    milling_pattern: patterning.MillingPattern,
    point: Point = Point(),
    parent=None,
):
    """Open the Milling Window ready for milling

    Args:
        milling_pattern (MillingPattern): The type of milling pattern
        x (float, optional): the initial pattern offset (x-direction). Defaults to 0.0.
        y (float, optional): the initial pattenr offset (y-direction). Defaults to 0.0.
    """
    milling_window = GUIMillingWindow(
        microscope=microscope,
        settings=settings,
        milling_pattern_type=milling_pattern,
        point = point,
        parent=parent,
    )

    milling_window.show()
    milling_window.exec_()
