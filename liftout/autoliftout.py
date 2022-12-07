import logging
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import napari
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
from fibsem.detection import detection
from fibsem.detection.detection import Feature, FeatureType
from fibsem.imaging import masks
from fibsem.imaging import utils as image_utils
from fibsem.patterning import MillingPattern
from fibsem.structures import (BeamType, MicroscopeSettings, MicroscopeState,
                               Point)
from fibsem.ui import windows as fibsem_ui_windows

from liftout import actions
from liftout.structures import (AutoLiftoutMode, AutoLiftoutStage, Lamella,
                                ReferenceHFW, ReferenceImages, Sample)


def log_status_message(lamella: Lamella, step: str):
    logging.debug(
        f"STATUS | {lamella._petname} | {lamella.current_state.stage.name} | {step}"
    )

# autoliftout workflow functions

# functional mill trench
def mill_lamella_trench(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
):

    # bookkeeping
    mode = AutoLiftoutMode[settings.protocol["options"]["auto"]["trench"].capitalize()]
    settings.image.save_path = lamella.path

    # move to lamella position
    calibration.set_microscope_state(microscope, lamella.lamella_state) # TODO: remove, this is redundant we are already here at setup? check

    # Take an ion beam image at the *milling current*
    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.save = False

    ######
    log_status_message(lamella, "MILL_TRENCH")

    # mill_trenches
    fibsem_ui_windows.milling_ui(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Trench,
        point=None,
        auto_continue=bool(mode is AutoLiftoutMode.Auto),
    )

    # charge neutralisation
    settings.image.beam_type = BeamType.ELECTRON
    calibration.auto_charge_neutralisation(microscope, settings.image)

    log_status_message(lamella, "REF_TRENCH")


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
    mode = AutoLiftoutMode[settings.protocol["options"]["auto"]["jcut"].capitalize()]
    settings.image.save_path = lamella.path
    settings.image.save = False

    log_status_message(lamella, "ALIGN_REF_TRENCH")

    # align to ref_trench
    reference_images = lamella.get_reference_images("ref_trench")
    alignment.correct_stage_drift(
        microscope,
        settings,
        reference_images=reference_images,
        alignment=(BeamType.ION, BeamType.ION),
        rotate=False,
        xcorr_limit=(100, 100),
        constrain_vertical=False,
    )

    # reference images of milled trenches
    settings.image.beam_type = BeamType.ELECTRON
    settings.image.hfw = ReferenceHFW.Medium.value
    calibration.auto_charge_neutralisation(microscope, settings.image)
    
    hfws = [ReferenceHFW.Low.value, ReferenceHFW.High.value]
    reference_images = acquire.take_set_of_reference_images(
        microscope, settings.image, hfws=hfws, label="ref_trench_jcut"
    )

    # move flat to electron beam
    movement.move_flat_to_beam(microscope, settings, beam_type=BeamType.ELECTRON)

    log_status_message(lamella, "ALIGN_REF_TRENCH_ROTATE")

    # align to ref_trench
    alignment.eucentric_correct_stage_drift(
        microscope, 
        settings, 
        reference_images, 
        rotate=True, 
        xcorr_limit=(512, 512, 512, 512)
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

    # align eucentric with reference images #   FLAG_TEST
    log_status_message(lamella, "ALIGN_REF_TRENCH_EUCENTRIC")

    # confirm
    if mode is AutoLiftoutMode.Manual:
        fibsem_ui_windows.ask_user_movement(
            microscope,
            settings,
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
    log_status_message(lamella, "ALIGN_REF_JCUT_TILT")

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

    det = fibsem_ui_windows.detect_features_v2(
        microscope=microscope,
        settings=settings,
        features=[
            Feature(FeatureType.LamellaLeftEdge),
            Feature(FeatureType.ImageCentre),
        ],
        validate=bool(mode is AutoLiftoutMode.Manual),
    )

    fibsem_ui_windows.milling_ui(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.JCut,
        point=det.features[0].feature_m,
        auto_continue=bool(mode is AutoLiftoutMode.Auto),
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
    log_status_message(lamella, "ALIGN_REF_JCUT_FLAT")

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
    log_status_message(lamella, "REF_JCUT")

    hfws = [ReferenceHFW.Medium.value, ReferenceHFW.Super.value]
    reference_images = acquire.take_set_of_reference_images(
        microscope, settings.image, hfws=hfws, label="ref_jcut"
    )

    return lamella


def liftout_lamella(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    mode = AutoLiftoutMode[settings.protocol["options"]["auto"]["liftout"].capitalize()]
    settings.image.save_path = lamella.path

    # convenience
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # get ready to do liftout by moving to liftout angle (flat to eb)
    actions.move_to_liftout_angle(microscope, settings)

    log_status_message(lamella, "ALIGN_REF_JCUT")

    reference_images = lamella.get_reference_images("ref_jcut")
    alignment.correct_stage_drift(
        microscope,
        settings,
        reference_images=reference_images,
        alignment=(BeamType.ELECTRON, BeamType.ELECTRON),
        rotate=False,
        xcorr_limit=(100, 100),
    )

    # eucentric alignment
    alignment.correct_stage_drift(
        microscope,
        settings,
        reference_images=reference_images,
        alignment=(BeamType.ION, BeamType.ION),
        rotate=False,
        xcorr_limit=(100, 100),
        constrain_vertical=True
    )

    # confirm
    if mode is AutoLiftoutMode.Manual:
        fibsem_ui_windows.ask_user_movement(
            microscope,
            settings,
            msg="Confirm lamella is centred in Ion Beam",
        )

    # reference images for needle location
    settings.image.save = True
    settings.image.hfw = ReferenceHFW.High.value
    settings.image.label = f"ref_liftout_needle"
    acquire.take_reference_images(microscope, settings.image)

    # land needle on lamella
    lamella = land_needle_on_milled_lamella(microscope, settings, lamella, mode=mode)

    log_status_message(lamella, "NEEDLE_JOIN_LAMELLA")

    # joining options
    if settings.protocol["options"]["liftout_joining_method"].capitalize() == "Weld":

        settings.image.beam_type = BeamType.ION
        det = fibsem_ui_windows.detect_features_v2(
            microscope=microscope,
            settings=settings,
            features=[
                Feature(FeatureType.LamellaLeftEdge),
                Feature(FeatureType.ImageCentre),
            ],
            validate=bool(mode is AutoLiftoutMode.Manual),
        )
        # mill weld
        fibsem_ui_windows.milling_ui(
            microscope=microscope,
            settings=settings,
            milling_pattern=MillingPattern.Weld,
            point=det.features[0].feature_m,
            auto_continue=bool(mode is AutoLiftoutMode.Auto),
        )
    if (
        settings.protocol["options"]["liftout_joining_method"].capitalize()
        == "Platinum"
    ):
        # sputter platinum
        fibsem_utils.sputter_platinum(
            microscope,
            settings.protocol["platinum"],
            whole_grid=False,
            default_application_file=settings.system.application_file,
        )

    logging.info(
        f"{lamella.current_state.stage.name}: lamella to needle joining complete."
    )

    settings.image.save = True
    settings.image.hfw = ReferenceHFW.High.value
    settings.image.label = f"liftout_needle_contact"
    acquire.take_reference_images(microscope, settings.image)

    log_status_message(lamella, "NEEDLE_SEVER_LAMELLA")

    settings.image.beam_type = BeamType.ION
    det = fibsem_ui_windows.detect_features_v2(
        microscope=microscope,
        settings=settings,
        features=[
            Feature(FeatureType.LamellaRightEdge),
            Feature(FeatureType.ImageCentre),
        ],
        validate=bool(mode is AutoLiftoutMode.Manual),
    )

    point = det.features[0].feature_m

    # jcut sever pattern
    fibsem_ui_windows.milling_ui(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Sever,
        point=point,
        auto_continue=bool(mode is AutoLiftoutMode.Auto),
    )

    # take reference images
    settings.image.save = True
    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.label = f"ref_liftout_sever"
    acquire.take_reference_images(microscope, settings.image)

    # Raise needle 30um from trench
    logging.info(
        f"{lamella.current_state.stage.name}: start removing needle from trench"
    )

    # move needle back from trench x
    dx = -1.5e-6
    movement.move_needle_relative_with_corrected_movement(
        microscope, dx=dx, dy=0, beam_type=BeamType.ION
    )

    for i in range(3):
        z_move_out_from_trench = movement.z_corrected_needle_movement(
            10e-6, stage.current_position.t
        ) #TODO: replace this with corrected relative needle move?
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
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    lamella: Lamella,
    mode: AutoLiftoutMode = AutoLiftoutMode.Manual,
) -> Lamella:

    # bookkeeping
    settings.image.save_path = lamella.path

    # validate needle insertion conditions
    validate_needle_insertion(
        microscope, settings.system.stage.needle_stage_height_limit
    )

    log_status_message(lamella, "INSERT_NEEDLE")

    # insert the needle for liftout
    actions.move_needle_to_liftout_position(microscope)

    # align needle to side of lamella
    log_status_message(lamella, "NEEDLE_EB_DETECTION")

    settings.image.beam_type = BeamType.ELECTRON
    det = fibsem_ui_windows.detect_features_v2(
        microscope=microscope,
        settings=settings,
        features=[
            Feature(FeatureType.NeedleTip),
            Feature(FeatureType.LamellaLeftEdge),
        ],
        validate=bool(mode is AutoLiftoutMode.Manual),
    )

    det.distance.x -= 2e-6  # move 1um to the left of the lamella edge
    detection.move_based_on_detection(
        microscope, settings, det, beam_type=settings.image.beam_type
    )

    log_status_message(lamella, "NEEDLE_IB_DETECTION")
     
    settings.image.beam_type = BeamType.ION
    det = fibsem_ui_windows.detect_features_v2(
        microscope=microscope,
        settings=settings,
        features=[
            Feature(FeatureType.NeedleTip),
            Feature(FeatureType.LamellaLeftEdge),
        ],
        validate=bool(mode is AutoLiftoutMode.Manual),
    )
    detection.move_based_on_detection(
        microscope, settings, det, beam_type=settings.image.beam_type, move_x=False
    )

    # reference images
    settings.image.hfw = ReferenceHFW.High.value
    settings.image.save = True
    settings.image.label = f"needle_liftout_start_position"
    acquire.take_reference_images(microscope, settings.image)

    if mode is AutoLiftoutMode.Manual:
        # check with user? # TODO: confusing wording
        response = fibsem_ui_windows.ask_user_interaction(
            msg="""Confirm the needle is in the trench, to the left of the lamella edge. 
            \nIf not, please move manually then press yes to continue."""
        )

    # take image

    # charge neutralisation  to charge lamella
    settings.image.beam_type = BeamType.ION
    calibration.auto_charge_neutralisation(microscope, settings.image, n_iterations=50)

    # measure brightness
    BRIGHTNESS_FACTOR = 1.2
    settings.image.beam_type = BeamType.ION
    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.label = f"needle_liftout_land"
    settings.image.save = True
    settings.image.gamma.enabled = False
    reduced_area = Rectangle(0.2, 0.2, 0.70, 0.70)  # TODO: improve contact detection
    ib_image = acquire.new_image(microscope, settings.image, reduced_area)
    previous_brightness = image_utils.measure_brightness(ib_image)

    brightness_history = [previous_brightness]
    mean_brightness = np.mean(brightness_history)

    iteration_count = 0
    MAX_ITERATIONS = 10

    log_status_message(lamella, "NEEDLE_CONTACT_DETECTION")

    while True:

        # move needle down
        microscope.specimen.manipulator.set_default_coordinate_system(
            ManipulatorCoordinateSystem.STAGE
        )

        dx = 0.5e-6
        dy = 0.0e-6
        movement.move_needle_relative_with_corrected_movement(
            microscope, dx=dx, dy=dy, beam_type=BeamType.ION
        )

        # calculate brightness
        settings.image.label = f"liftout_contact_brightness_{iteration_count}"
        ib_image = acquire.new_image(
            microscope, settings.image, reduced_area=reduced_area
        )
        brightness = image_utils.measure_brightness(ib_image)

        logging.info(
            f"iter: {iteration_count}: brightness: {brightness}, prevs: {previous_brightness}, MEAN BRIGHTNESS: {mean_brightness}"
        )

        above_brightness_threshold = brightness > mean_brightness * BRIGHTNESS_FACTOR

        if above_brightness_threshold or mode is AutoLiftoutMode.Manual:

            # needle has landed...
            if above_brightness_threshold:
                logging.info("THRESHOLD REACHED STOPPPING")

            # check with user? # TODO: confusing wording
            response = fibsem_ui_windows.ask_user_interaction(
                msg="Has the needle landed on the lamella? \nPress Yes to continue, or No to redo the final movement"
            )

            if response:
                break

        previous_brightness = brightness
        brightness_history.append(brightness)
        mean_brightness = np.mean(brightness_history)

        iteration_count += 1
        if iteration_count >= MAX_ITERATIONS:
            break

    # move needle up in z to prevent bottoming out
    dz = 0.5e-6 # positive is away from sample (up)
    movement.move_needle_relative_with_corrected_movement(
            microscope, dx=0, dy=dz, beam_type=BeamType.ION
        )
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
    mode = AutoLiftoutMode[settings.protocol["options"]["auto"]["landing"].capitalize()]
    settings.image.save_path = lamella.path
    settings.image.save = False

    # conveienence
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # move to landing coordinate
    calibration.set_microscope_state(microscope, lamella.landing_state)

    # align to ref
    log_status_message(lamella, "ALIGN_REF_LANDING")
    reference_images = lamella.get_reference_images("ref_landing")
    alignment.correct_stage_drift(
        microscope,
        settings,
        reference_images=reference_images,
        alignment=(BeamType.ION, BeamType.ION),
        rotate=False,
        xcorr_limit=(100, 100),
    )

    # confirm eucentricity
    if mode is AutoLiftoutMode.Manual:
        fibsem_ui_windows.ask_user_movement(microscope, settings)

    ############################## LAND_LAMELLA ##############################
    validate_needle_insertion(
        microscope, settings.system.stage.needle_stage_height_limit
    )


    # landing entry
    log_status_message(lamella, "LAND_LAMELLA_ENTRY")
    landing_entry_procedure(microscope, settings, bool(mode is AutoLiftoutMode.Manual))
    
    # land lamella on post
    log_status_message(lamella, "LAND_LAMELLA_ON_POST")
    response = False
    while response is False:

        # land the lamella on the post
        land_lamella_on_post(microscope, settings, lamella, mode)

        # confirm with user
        if mode is AutoLiftoutMode.Manual:
            response = fibsem_ui_windows.ask_user_interaction(
                msg="Has the lamella landed on the post? \nPress Yes to continue, or No to redo the landing procedure."
            )
        else:
            response = True

    log_status_message(lamella, "LAND_LAMELLA_REMOVE_NEEDLE")
    # move needle out of trench slowly at first (QUERY: is this required?)
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

    # reference images
    acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=settings.image,
        hfws=[ReferenceHFW.Medium.value, ReferenceHFW.Super.value],
        label="ref_landing_lamella",
    )

    return lamella


def land_lamella_on_post(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    lamella: Lamella,
    mode: AutoLiftoutMode,
):
    VALIDATE = mode is AutoLiftoutMode.Manual

    # repeat final movement until user confirms landing
    response = False
    i = 0
    while response is False:

        #### X-MOVE
        settings.image.hfw = ReferenceHFW.High.value if i == 0 else ReferenceHFW.Super.value
        settings.image.beam_type = BeamType.ION
        settings.image.save = True
        settings.image.label = f"landing_needle_pre_move_{i}"

        log_status_message(lamella, "LAND_LAMELLA_IB_DETECTION")

        det = fibsem_ui_windows.detect_features_v2(
            microscope=microscope,
            settings=settings,
            features=[
                Feature(FeatureType.LamellaRightEdge),
                Feature(FeatureType.LandingPost),
            ],
            validate=VALIDATE,
            mask_radius=512
        )

        det.distance.x += 0.5e-6
        detection.move_based_on_detection(
            microscope, settings, det, beam_type=settings.image.beam_type
        )

        # final reference images
        # settings.image.hfw = ReferenceHFW.Super.value
        # settings.image.beam_type = BeamType.ELECTRON
        settings.image.save = True
        settings.image.label = f"landing_lamella_contact_{i}"
        acquire.take_reference_images(
            microscope=microscope, image_settings=settings.image
        )

        if mode is AutoLiftoutMode.Manual:
            response = fibsem_ui_windows.ask_user_interaction(
                msg="Has the lamella made contact with the post? \nPress Yes to continue, or No to redo the final movement",
            )
        else:
            response = True

            # TODO: add a check where if the lamella edge and post are close enough, exit the loop

        # increment count
        i += 1

    #################################################################################################

    ############################## WELD TO LANDING POST #############################################

    log_status_message(lamella, "LAND_LAMELLA_WELD_TO_POST")

    settings.image.beam_type = BeamType.ION
    det = fibsem_ui_windows.detect_features_v2(
        microscope=microscope,
        settings=settings,
        features=[
            Feature(FeatureType.LamellaRightEdge),
            Feature(FeatureType.ImageCentre),
        ],
        validate=bool(mode is AutoLiftoutMode.Manual),
    )

    fibsem_ui_windows.milling_ui(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Weld,
        point=det.features[0].feature_m,
        auto_continue=bool(mode is AutoLiftoutMode.Auto),
    )

    # final reference images
    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.save = True
    settings.image.label = f"landing_lamella_final_weld_high_res"
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    #################################################################################################

    ###################################### REMOVE NEEDLE ######################################

    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.beam_type = BeamType.ION
    settings.image.save = True
    settings.image.label = "landing_lamella_needle_removal"

    
    # charge neutralisation # discharge to unlock lamella
    response = True
    n_iter = 50
    while response:
        settings.image.beam_type = BeamType.ELECTRON
        calibration.auto_charge_neutralisation(microscope, settings.image, n_iterations=n_iter)

        response = fibsem_ui_windows.ask_user_interaction(
            msg=f"Repeat charge procedure? iter: {n_iter} \nPress Yes to repeat, or No to continue.",
        )

    # optional? makes it not repeatable
    log_status_message(lamella, "LAND_LAMELLA_REMOVE_NEEDLE")

    # if settings.protocol["options"]["landing_joining_method"].capitalize() == "Weld":
        # det = fibsem_ui_windows.detect_features_v2(
        #     microscope=microscope,
        #     settings=settings,
        #     features=[
        #         Feature(FeatureType.LamellaLeftEdge),
        #         Feature(FeatureType.ImageCentre),
        #     ],
        #     validate=bool(mode is AutoLiftoutMode.Manual),
        # )

        # fibsem_ui_windows.milling_ui(
        #     microscope=microscope,
        #     settings=settings,
        #     milling_pattern=MillingPattern.Cut,
        #     point=det.features[0].feature_m,
        #     auto_continue=bool(mode is AutoLiftoutMode.Auto),
        # )

    logging.info(f"{lamella.current_state.stage.name}: removing needle from lamella")

    # back out needle from lamella , no cut required?
    for i in range(10):

        # move needle back
        movement.move_needle_relative_with_corrected_movement(
            microscope=microscope, dx=-0.5e-6, dy=0, beam_type=BeamType.ION,
        )

    # reference images
    acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=settings.image,
        hfws=[ReferenceHFW.High.value, ReferenceHFW.Super.value],
        label="landing_lamella_needle_removal",
    )

    return

def landing_entry_procedure(microscope: SdbMicroscopeClient, settings: MicroscopeSettings, VALIDATE: bool = True):
    
    # entry procedure, align vertically to post
    actions.move_needle_to_landing_position(microscope)

    hfws = [ReferenceHFW.Low, ReferenceHFW.Medium, ReferenceHFW.High]
    beam_types = [BeamType.ION, BeamType.ELECTRON, BeamType.ION]

    for i, (hfw, beam_type) in enumerate(zip(hfws, beam_types)):

        # needle starting position
        settings.image.hfw = hfw.value
        settings.image.beam_type =beam_type
        settings.image.save = True
        settings.image.label = f"landing_needle_start_position_{i}"
        acquire.take_reference_images(microscope, settings.image)

        det = fibsem_ui_windows.detect_features_v2(
            microscope=microscope,
            settings=settings,
            features=[
                Feature(FeatureType.LamellaRightEdge),
                Feature(FeatureType.ImageCentre),
            ],
            validate=VALIDATE,
            mask_radius = 512
        )
        det.distance.x -= 30e-6
        detection.move_based_on_detection(
            microscope, settings, det, beam_type=settings.image.beam_type
        )

    settings.image.hfw = ReferenceHFW.Super.value
    settings.image.label = f"landing_needle_ready_position"
    acquire.take_reference_images(microscope, settings.image)



def reset_needle(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    mode = AutoLiftoutMode[settings.protocol["options"]["auto"]["reset"].capitalize()]
    settings.image.save_path = lamella.path

    # optionally skip reset
    if mode is AutoLiftoutMode.Manual:
        response = fibsem_ui_windows.ask_user_interaction(
            msg="Do you want to complete reset? \nPress Yes to continue, or No to skip",
        )

        if response is False:
            return lamella

    # convienence
    stage = microscope.specimen.stage

    # move sample stage out
    actions.move_sample_stage_out(microscope, settings)

    ###################################### SHARPEN_NEEDLE ######################################

    validation.validate_stage_height_for_needle_insertion(
        microscope, settings.system.stage.needle_stage_height_limit
    )

    # move needle in
    actions.move_needle_to_reset_position(microscope)

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
    calibration.align_needle_to_eucentric_position(microscope, settings, validate=False)

    # TODO: validate this movement

    # create sharpening patterns
    fibsem_ui_windows.milling_ui(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Sharpen,
        point=None,
        auto_continue=bool(mode is AutoLiftoutMode.Auto),
    )

    #################################################################################################

    # reset the "eucentric position" for the needle, centre needle in both views
    calibration.align_needle_to_eucentric_position(microscope, settings, validate=True)

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


def setup_polish_lamella(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    mode = AutoLiftoutMode[settings.protocol["options"]["auto"]["thin"].capitalize()]
    settings.image.save_path = lamella.path

    # move to the initial landing coordinates
    calibration.set_microscope_state(microscope, lamella.landing_state)

    log_status_message(lamella, "THIN_LAMELLA_MOVEMENT")

    # rotate_and_tilt_to_thinning_angle
    settings.image.hfw = ReferenceHFW.Medium.value
    actions.move_to_thinning_angle(microscope=microscope, protocol=settings.protocol)

    # load the reference images
    reference_images = lamella.get_reference_images(label="ref_landing_lamella")
 
    log_status_message(lamella, "THIN_LAMELLA_ALIGN")

    alignment.correct_stage_drift(
        microscope,
        settings,
        reference_images=reference_images,
        alignment=(BeamType.ION, BeamType.ION),
        rotate=True,
        xcorr_limit = (512, 512)
    )

    # ensure_eucentricity at thinning angle
    if mode is AutoLiftoutMode.Manual:
        
        # confirm
        fibsem_ui_windows.ask_user_movement(
            microscope,
            settings,
            msg="Confirm lamella right edge is centred in Ion Beam",
        ) 
        # TODO: add tilt correction here...
        # TODO: add beam shift too?

    # lamella images
    reference_images = acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[ReferenceHFW.High.value, ReferenceHFW.Super.value],
        label="ref_thin_lamella_start",
    )

    # thin_lamella (align and mill)
    settings.image.resolution = settings.protocol["thin_lamella"]["resolution"]
    settings.image.dwell_time = settings.protocol["thin_lamella"]["dwell_time"]
    settings.image.hfw = settings.protocol["thin_lamella"]["hfw"]

    log_status_message(lamella, "REF_THIN_LAMELLA_SETUP")

    # # take reference images
    reference_images = acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[ReferenceHFW.High.value, ReferenceHFW.Super.value],
        label="ref_thin_lamella_setup",
    )

    settings.image.hfw = ReferenceHFW.Ultra.value
    settings.image.save = True
    settings.image.label = f"ref_thin_lamella_setup_ultra_res"
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    return


def thin_lamella(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    mode = AutoLiftoutMode[settings.protocol["options"]["auto"]["thin"].capitalize()]
    settings.image.save_path = lamella.path

    # load the reference images
    reference_images = lamella.get_reference_images(label="ref_thin_lamella_setup")
   
    log_status_message(lamella, "THIN_LAMELLA_ALIGN")

    alignment.correct_stage_drift(
        microscope,
        settings,
        reference_images=reference_images,
        alignment=(BeamType.ION, BeamType.ION),
        rotate=True,
        xcorr_limit = (100, 100)
    )

    # ensure_eucentricity at thinning angle
    if mode is AutoLiftoutMode.Manual:
        
        # confirm
        fibsem_ui_windows.ask_user_movement(
            microscope,
            settings,
            msg="Confirm lamella right edge is centred in Ion Beam",
        ) 

    # lamella images
    reference_images = acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[ReferenceHFW.High.value, ReferenceHFW.Super.value],
        label="ref_thin_lamella_restore",
    )

    # thin_lamella (align and mill)
    settings.image.resolution = settings.protocol["thin_lamella"]["resolution"]
    settings.image.dwell_time = settings.protocol["thin_lamella"]["dwell_time"]
    settings.image.hfw = settings.protocol["thin_lamella"]["hfw"]

    log_status_message(lamella, "THIN_LAMELLA_MILL")

    # # QUERY: add a fiducial here?
    # fibsem_ui_windows.milling_ui(
    #     microscope=microscope,
    #     settings=settings,
    #     milling_pattern=MillingPattern.Fiducial,
    #     point=None,
    #     auto_continue=bool(mode is AutoLiftoutMode.Auto),
    # )

    # mill thin_lamella
    fibsem_ui_windows.milling_ui(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Thin,
        point=None,
        auto_continue=bool(mode is AutoLiftoutMode.Auto),
    )

    log_status_message(lamella, "REF_THIN_LAMELLA")

    # # take reference images
    reference_images = acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[ReferenceHFW.High.value, ReferenceHFW.Super.value],
        label="ref_thin_lamella",
    )

    settings.image.hfw = ReferenceHFW.Ultra.value
    settings.image.save = True
    settings.image.label = f"ref_thin_lamella_ultra_res"
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    return


def polish_lamella(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:

    # bookkeeping
    mode = AutoLiftoutMode[settings.protocol["options"]["auto"]["polish"].capitalize()]
    settings.image.save_path = lamella.path

    # # restore state from thinning stage
    reference_images = lamella.get_reference_images(label="ref_thin_lamella")

    # TODO: Test, probs only needs 1 step
    log_status_message(lamella, "POLISH_LAMELLA_ALIGN")

    alignment.correct_stage_drift(
        microscope,
        settings,
        reference_images=reference_images,
        alignment=(BeamType.ION, BeamType.ION),
        rotate=False,
        xcorr_limit = (100, 100)
    )

    # confirm
    if mode is AutoLiftoutMode.Manual:
        fibsem_ui_windows.ask_user_movement(
            microscope,
            settings,
            msg="Confirm lamella right edge is centred in Ion Beam",
        )

    # realign lamella to image centre
    reference_images = acquire.take_set_of_reference_images(
        microscope, settings.image,  [ReferenceHFW.High.value, ReferenceHFW.Super.value], "ref_polish_lamella_start"
    )

    settings.image.hfw = ReferenceHFW.Ultra.value
    settings.image.save = True
    settings.image.label = f"ref_polish_lamella_ultra_res"
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    # polish (align and mill)
    settings.image.resolution = settings.protocol["polish_lamella"]["resolution"]
    settings.image.dwell_time = settings.protocol["polish_lamella"]["dwell_time"]
    settings.image.hfw = settings.protocol["polish_lamella"]["hfw"]

    log_status_message(lamella, "POLISH_LAMELLA_MILL")
    fibsem_ui_windows.milling_ui(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Polish,
        point=None,
        auto_continue=bool(mode is AutoLiftoutMode.Auto),
    )

    log_status_message(lamella, "POLISH_LAMELLA_REF")
    # take reference images (ultra, super, high)
    reference_images = acquire.take_set_of_reference_images(
        microscope, settings.image,  [ReferenceHFW.High.value, ReferenceHFW.Super.value], "ref_polish_lamella"
    )

    settings.image.hfw = ReferenceHFW.Ultra.value
    settings.image.save = True
    settings.image.label = f"ref_polish_lamella_ultra_res"
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    return lamella


def run_autoliftout_workflow(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, sample: Sample,
) -> Sample:

    BATCH_MODE = bool(settings.protocol["options"]["batch_mode"])
    CONFIRM_WORKFLOW_ADVANCE = bool(settings.protocol["options"]["confirm_advance"])

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
    settings.image.save = False
    settings.image.label = f"{fibsem_utils.current_timestamp()}"

    # batch mode workflow
    if BATCH_MODE:
        for terminal_stage in [
            AutoLiftoutStage.MillTrench,
            AutoLiftoutStage.MillJCut,
        ]:

            lamella: Lamella
            for lamella in sample.positions.values():

                if lamella.is_failure:
                    continue  # skip failures

                while lamella.current_state.stage.value < terminal_stage.value:

                    next_stage = AutoLiftoutStage(lamella.current_state.stage.value + 1)

                    # update image settings (save in correct directory)
                    settings.image.save_path = lamella.path

                    # reset to the previous state
                    lamella = start_of_stage_update(
                        microscope, lamella, next_stage=next_stage
                    )

                    # run the next workflow stage
                    lamella = autoliftout_stages[next_stage](
                        microscope=microscope, settings=settings, lamella=lamella,
                    )

                    # advance workflow
                    sample = end_of_stage_update(microscope, sample, lamella)

    
    # autohome stage
    # if terminal_stage is AutoLiftoutStage.MillJCut:
    # calibration.auto_home_and_link_v2(microscope)

    # standard workflow
    lamella: Lamella
    for lamella in sample.positions.values():

        if lamella.is_failure:
            continue  # skip failures

        while lamella.current_state.stage.value < AutoLiftoutStage.Reset.value:

            next_stage = AutoLiftoutStage(lamella.current_state.stage.value + 1)
            if CONFIRM_WORKFLOW_ADVANCE:
                msg = (
                    f"""Continue Lamella {(lamella._petname)} from {next_stage.name}?"""
                )
                response = fibsem_ui_windows.ask_user_interaction(msg=msg)
            else:
                response = True

            # update image settings (save in correct directory)
            settings.image.save_path = lamella.path

            if response:

                # reset to the previous state
                lamella = start_of_stage_update(
                    microscope, lamella, next_stage=next_stage
                )

                # run the next workflow stage
                lamella = autoliftout_stages[next_stage](
                    microscope=microscope, settings=settings, lamella=lamella,
                )

                # advance workflow
                sample = end_of_stage_update(microscope, sample, lamella)
            else:
                break  # go to the next sample

    return sample


def end_of_stage_update(
    microscope: SdbMicroscopeClient, sample: Sample, lamella: Lamella
) -> Sample:
    """Save the current microscope state configuration to disk, and log that the stage has been completed."""

    # save state information
    lamella.current_state.microscope_state = calibration.get_current_microscope_state(
        microscope=microscope
    )
    lamella.current_state.end_timestamp = datetime.timestamp(datetime.now())

    # write history
    lamella.history.append(deepcopy(lamella.current_state))

    # update and save sample
    sample = update_sample_lamella_data(sample, lamella)

    logging.info(f"STATUS | {lamella._petname} | {lamella.current_state.stage} | FINISHED")

    return sample


def start_of_stage_update(
    microscope: SdbMicroscopeClient, lamella: Lamella, next_stage: AutoLiftoutStage,
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
    logging.info(f"STATUS | {lamella._petname} | {lamella.current_state.stage} | STARTED")

    return lamella


def run_thinning_workflow(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, sample: Sample,
) -> Sample:


    autoliftout_polishing_stages = {
        AutoLiftoutStage.SetupPolish: setup_polish_lamella,
        AutoLiftoutStage.Thinning: thin_lamella,
        AutoLiftoutStage.Polishing: polish_lamella,
    }

    # lamella: Lamella
    for next_stage  in [AutoLiftoutStage.SetupPolish, AutoLiftoutStage.Thinning, AutoLiftoutStage.Polishing]:
        for lamella in sample.positions.values():
                
            if lamella.is_failure:
                continue

            if lamella.current_state.stage.value == next_stage.value - 1:
                lamella = start_of_stage_update(
                    microscope, lamella, next_stage=next_stage
                )
                autoliftout_polishing_stages[next_stage](microscope, settings, lamella)
                sample = end_of_stage_update(microscope, sample, lamella)

    # finish the experiment
    for lamella in sample.positions.values():
        if lamella.current_state.stage == AutoLiftoutStage.Polishing:
            logging.info(f"STATUS | {lamella._petname} | {AutoLiftoutStage.Finished} | STARTED")
            lamella.current_state.stage = AutoLiftoutStage.Finished
            sample = end_of_stage_update(microscope, sample, lamella)

    return sample


def get_current_lamella(sample: Sample,) -> bool:

    if sample.positions:
        select_another_lamella = fibsem_ui_windows.ask_user_interaction(
            msg=f"Do you want to select another lamella?\n"
            f"{len(sample.positions)} currentlly selected.",
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
        microscope, settings, msg=msg,
    )

    return calibration.get_current_microscope_state(microscope)


def select_initial_lamella_positions(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    sample: Sample,
) -> Lamella:
    """Select the initial sample positions for liftout"""

    # create lamella
    if sample.positions:
        lamella_no = max(sample.positions.keys()) + 1
    else:
        lamella_no = 1
    lamella = Lamella(sample.path, lamella_no)

    # save lamella coordinates
    lamella.lamella_state = user_select_feature(
        microscope, settings, msg="Select a lamella position."
    )

    # save microscope state
    lamella.current_state.microscope_state = calibration.get_current_microscope_state(
        microscope=microscope,
    )

    settings.image.hfw = ReferenceHFW.Low.value
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
    landing_state = MicroscopeState.__from_dict__(settings.protocol["landing_state"])
    actions.move_to_landing_grid_v2(microscope, landing_state)

    settings.image.save = False
    ####################################

    # select corresponding sample landing positions
    lamella: Lamella
    for lamella in sample.positions.values():

        # check if landing position already selected? so it doesnt overwrite
        if lamella.landing_selected is False:
            settings.image.hfw = ReferenceHFW.Low.value
            lamella = select_landing_sample_positions(microscope, settings, lamella)

            sample = update_sample_lamella_data(sample, lamella)

    return sample


def update_sample_lamella_data(sample: Sample, lamella: Lamella) -> Sample:

    sample.positions[lamella._number] = deepcopy(lamella)
    sample.save()
    return sample


def select_landing_sample_positions(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, lamella: Lamella,
) -> Lamella:
    """Select the landing coordinates for a lamella."""
    logging.info(f"Selecting Landing Position: {lamella._petname}")

    # update image path
    settings.image.save_path = lamella.path
    settings.image.hfw = ReferenceHFW.Low.value

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

    fibsem_ui_windows.milling_ui(
        microscope=microscope,
        settings=settings,
        milling_pattern=MillingPattern.Flatten,
        point=None,
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
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, sample: Sample,
):

    select_another = get_current_lamella(sample)

    lamella_state = MicroscopeState.__from_dict__(settings.protocol["lamella_state"])
    actions.move_to_sample_grid_v2(microscope, lamella_state)

    fibsem_ui_windows.ask_user_movement(microscope, settings)

    # allow the user to select additional lamella positions
    while select_another:

        lamella = select_initial_lamella_positions(
            microscope, settings, sample
        )

        # save lamella data
        sample = update_sample_lamella_data(sample, lamella)

        # select another?
        select_another = get_current_lamella(sample)

    # select landing positions
    select_landing_positions(microscope, settings, sample)

    # finish setup
    finish_setup_autoliftout(microscope, sample)

    return sample


def finish_setup_autoliftout(
    microscope: SdbMicroscopeClient, sample: Sample,
):
    """Finish the setup stage for autolifout/autolamella"""

    # reset microscope coordinate system
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)

    logging.info(f"Selected {len(sample.positions)} lamella for autoliftout.")
    logging.info(f"INIT | {AutoLiftoutStage.Setup.name} | FINISHED")


def run_setup_autoliftout(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, sample: Sample,
) -> Sample:

    logging.info(f"INIT | {AutoLiftoutStage.Setup.name} | STARTED")

    # move to the initial sample grid position
    actions.move_to_sample_grid(microscope, settings, settings.protocol) # TODO: set a state for this

    # initial image settings
    settings.image.hfw = ReferenceHFW.Low.value
    settings.image.beam_type = BeamType.ELECTRON
    settings.image.save = True
    settings.image.save_path = sample.path
    settings.image.label = "grid"
    # NOTE: can't take ion beam image with such a high hfw, will default down to max ion beam hfw
    acquire.new_image(microscope, settings.image)

    # sputter platinum to protect grid and prevent charging...
    sputter_platinum_on_whole_sample_grid(microscope, settings, settings.protocol)

    # reference images
    settings.image.label = "grid_Pt"
    acquire.take_reference_images(microscope, settings.image)

    # check if focus is good enough
    ret = validation.validate_focus(microscope, settings.system.electron, link=False)

    if ret is False:
        fibsem_ui_windows.ask_user_interaction(
            msg="The AutoFocus routine has failed, please correct the focus manually."
        )

    # select the lamella and landing positions
    sample = select_lamella_positions(microscope, settings, sample)

    return sample


###TODO: MOVE TO FIBSEM


def validate_needle_insertion(
    microscope: SdbMicroscopeClient, needle_stage_height_limit: float = 3.7e-3
) -> None:

    # move needle to liftout start position
    ret = validation.validate_stage_height_for_needle_insertion(
        microscope, needle_stage_height_limit
    )

    while ret is False:
        fibsem_ui_windows.ask_user_interaction(
            msg=f"""The system has identified the distance between the sample and the pole piece is less than {needle_stage_height_limit * 1000}mm. "
            "The needle will contact the sample, and it is unsafe to insert the needle. "
            "\nPlease manually refocus and link the stage, then press OK to continue. """,
        )

        ret = validation.validate_stage_height_for_needle_insertion(
            microscope, needle_stage_height_limit
        )


def sputter_platinum_on_whole_sample_grid(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings, protocol: dict,
) -> None:
    """Move to the sample grid and sputter platinum over the whole grid"""

    # Whole-grid platinum deposition
    response = fibsem_ui_windows.ask_user_interaction(
        msg="Do you want to sputter the whole \nsample grid with platinum?",
        image=acquire.last_image(microscope, BeamType.ELECTRON).data,
    )

    if response:
        actions.move_to_sample_grid(microscope, settings, protocol) # TODO: set a state for here??
        fibsem_utils.sputter_platinum(
            microscope=microscope,
            protocol=protocol["platinum"],
            whole_grid=True,
            default_application_file=settings.system.application_file,
        )

    return
