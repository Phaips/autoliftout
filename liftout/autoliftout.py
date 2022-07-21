import datetime
import logging
import time

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
from autoscript_sdb_microscope_client.structures import MoveSettings, StagePosition, ManipulatorPosition

from liftout import utils
from liftout.fibsem import acquire, calibration, movement
from liftout.fibsem.acquire import BeamType, GammaSettings, ImageSettings
from liftout.fibsem.sample import Lamella, Sample, AutoLiftoutStage
from liftout.gui import windows
from liftout.gui.milling_window import MillingPattern
from liftout.fibsem import utils as fibsem_utils
import numpy as np
from liftout.detection.detection import DetectionType

# autoliftout workflow functions

# functional mill trench
def mill_lamella_trench(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella,
):

    # move to lamella position
    movement.safe_absolute_stage_movement(microscope, lamella.lamella_coordinates)

    # move flat to the ion beam, stage tilt 25 (total image tilt 52)
    movement.move_to_trenching_angle(microscope, settings)

    # Take an ion beam image at the *milling current*
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_super_res"]

    # confirm position
    windows.ask_user_movement_v2(
        microscope, settings, image_settings, msg_type="centre_ib"
    )

    # update the lamella coordinates, and save
    lamella.lamella_coordinates = calibration.get_raw_stage_position(microscope)
    # self.current_sample_position.save_data()

    # mill_trenches
    windows.open_milling_window_v2(MillingPattern.Trench)

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
    lamella: Lamella) -> Lamella:


    # load the reference images
    reference_images = []
    for fname in ["ref_trench_low_res_eb", "ref_trench_high_res_eb", "ref_trench_low_res_ib", "ref_trench_high_res_ib"]:

        reference_images.append(lamella.load_reference_image(fname))

    # move flat to electron beam
    movement.flat_to_beam(microscope, settings, beam_type=BeamType.ELECTRON)

    # TODO: move user correction inside correct_stage_drift
    # correct drift using reference images..
    ret = calibration.correct_stage_drift(
        microscope,
        image_settings,
        reference_images,
        mode="ib",
    )

    if ret is False:
        # # cross-correlation has failed, manual correction required
        logging.info(f"{lamella.current_state.stage.name}: cross-correlation manually corrected")
        
        windows.ask_user_movement_v2(
            microscope,
            settings,
            image_settings,
            msg_type="centre_eb",
        )

    
    # then using ML, tilting/correcting in steps so drift isn't too large
    image_settings.hfw = settings["calibration"]["drift_correction_hfw_highres"]
    image_settings.save=True
    image_settings.label=f"drift_correction_ML"

    stage_settings = MoveSettings(rotate_compucentric=True)
    movement.move_relative(
        microscope,
        t=np.deg2rad(settings["jcut"]["jcut_angle"]),
        settings=stage_settings,
    )
    image_settings.hfw=settings["calibration"]["drift_correction_hfw_highres"],
    image_settings.save=True        
    image_settings.label=f"drift_correction_ML"
    calibration.correct_stage_drift_with_ML_v2(microscope, settings, image_settings)

    ## MILL_JCUT
    # now we are at the angle for jcut, perform jcut
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"]
    windows.open_milling_window_v2(MillingPattern.JCut)

    # take reference images of the jcut
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_med_res"]
    image_settings.save=True
    image_settings.label=f"jcut_lowres"
    acquire.take_reference_images(microscope, image_settings)

    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_ultra_res"]
    image_settings.label=f"jcut_highres"
    acquire.take_reference_images(microscope, image_settings)

    return lamella


def liftout_lamella(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella) -> Lamella:

    # convenience
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # get ready to do liftout by moving to liftout angle
    movement.move_to_liftout_angle(microscope, settings)

    # check eucentric height
    windows.ask_user_movement_v2(
        microscope,
        settings,
        image_settings,
        msg_type="eucentric",
        flat_to_sem=True,
    )  # liftout angle is flat to SEM

    # correct stage drift from mill_lamella stage #TODO: shouldnt need to do this, remove...
    calibration.correct_stage_drift_with_ML_v2(microscope, settings, image_settings)

    # move needle to liftout start position
    calibration.validate_stage_height_for_needle_insertion(microscope, settings, image_settings)
    park_position = movement.move_needle_to_liftout_position(microscope)
    logging.info(f"{lamella.current_state.stage.name}: needle inserted to park positon: {park_position}"
    )

    # land needle on lamella
    lamella = land_needle_on_milled_lamella(microscope, settings, image_settings, lamella)

    # sputter platinum
    fibsem_utils.sputter_platinum(microscope, settings, whole_grid=False)
    logging.info(f"{lamella.current_state.stage.name}: lamella to needle welding complete.")

    image_settings.save=True
    image_settings.hfw=settings["platinum"]["weld"]["hfw"]
    image_settings.label=f"needle_landed_Pt_sputter"
    acquire.take_reference_images(microscope, image_settings)

    # jcut sever pattern
    windows.open_milling_window_v2(MillingPattern.Sever)

    # take reference images
    image_settings.save=True
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"],
    image_settings.label=f"jcut_sever"
    acquire.take_reference_images(microscope, image_settings)

    # Raise needle 30um from trench
    logging.info(f"{lamella.current_state.stage.name}: start removing needle from trench")
    for i in range(3):
        z_move_out_from_trench = movement.z_corrected_needle_movement(
            10e-6, stage.current_position.t
        )
        needle.relative_move(z_move_out_from_trench)
        image_settings.label = f"liftout_trench_{i}"
        acquire.take_reference_images(microscope, image_settings)
        logging.info(f"{lamella.current_state.stage.name}: removing needle from trench at {z_move_out_from_trench} ({i + 1}/3)")
        time.sleep(1)

    # reference images after liftout complete
    image_settings.label = f"liftout_of_trench"
    acquire.take_reference_images(microscope, image_settings)

    # move needle to park position
    movement.retract_needle(microscope, park_position)

    return lamella


def land_needle_on_milled_lamella(
    microscope: SdbMicroscopeClient, 
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella) -> Lamella:

    # conveinence
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # reference images
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.save=True
    image_settings.label=f"needle_liftout_start_position_lowres"

    det = calibration.validate_detection_v2(
        microscope,
        settings,
        image_settings,
        shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
        beam_type=BeamType.ELECTRON,
    )

    x_move = movement.x_corrected_needle_movement(
        det.distance_metres.x, stage_tilt=stage.current_position.t
    )
    yz_move = movement.y_corrected_needle_movement(
        det.distance_metres.y, stage_tilt=stage.current_position.t
    )
    needle.relative_move(x_move)
    needle.relative_move(yz_move)
    logging.info(f"{lamella.current_state.stage.name}: needle x-move: {x_move}")
    logging.info(f"{lamella.current_state.stage.name}: needle yz-move: {yz_move}")

    ###

    # TODO: Consolidate all this into tighter functions

    ### Z-HALF MOVE (ION)
    # calculate shift between lamella centre and needle tip in the ion view
    image_settings.label = f"needle_liftout_post_xy_movement_lowres"
    det = calibration.validate_detection_v2(
        microscope,
        settings,
        image_settings,
        shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
        beam_type=BeamType.ION,
    )

    # calculate shift in xyz coordinates
    z_distance = det.distance_metres.y / np.cos(stage.current_position.t)

    # Calculate movement
    zy_move_half = movement.z_corrected_needle_movement(
        -z_distance / 2, stage.current_position.t
    )
    needle.relative_move(zy_move_half)
    logging.info(f"{lamella.current_state.stage.name}: needle z-half-move: {zy_move_half}")

    ###

    # repeat the final movement until user confirms.
    response = False
    while response is False:

        ### Z-MOVE FINAL (ION)
        image_settings.hfw = settings["calibration"]["reference_images"][
            "hfw_super_res"
        ]
        image_settings.label = f"needle_liftout_post_z_half_movement_highres"
        det = calibration.validate_detection_v2(
            microscope,
            settings,
            image_settings,
            shift_type=(DetectionType.NeedleTip, DetectionType.LamellaCentre),
            beam_type=BeamType.ION,
        )

        # calculate shift in xyz coordinates
        z_distance = det.distance_metres.y / np.cos(stage.current_position.t)

        # move in x
        x_move = movement.x_corrected_needle_movement(det.distance_metres.x)
        needle.relative_move(x_move)

        # move in z
        # detection is based on centre of lamella, we want to land near the edge
        # gap = 0.2e-6 #lamella_height / 10
        zy_move_gap = movement.z_corrected_needle_movement(
            -z_distance, stage.current_position.t
        )
        needle.relative_move(zy_move_gap)

        logging.info(f"{lamella.current_state.stage.name}: needle x-move: {x_move}")
        logging.info(f"{lamella.current_state.stage.name}: needle zy-move: {zy_move_gap}")

        image_settings.save = False
        acquire.take_reference_images(microscope, image_settings)

        response = windows.ask_user_interaction_v2(
            microscope,
            settings,
            image_settings,
            msg="Has the needle landed on the lamella? \nPress Yes to continue, or No to redo the final movement",
            beam_type=BeamType.ION,
        )

    # take final reference images
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"],
    image_settings.save=True
    image_settings.label=f"needle_liftout_landed_lowres"
    acquire.take_reference_images(microscope, image_settings)

    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label=f"needle_liftout_landed_highres"
    acquire.take_reference_images(microscope, image_settings)


    return lamella


def land_lamella( 
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella) -> Lamella:

    # conveienence
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator



    # move to landing coordinate
    movement.safe_absolute_stage_movement(
        microscope=microscope,
        stage_position=lamella.landing_coordinates,
    )
    movement.auto_link_stage(
        microscope,
        hfw=settings["calibration"]["reference_images"]["hfw_med_res"],
    )

    # confirm eucentricity
    windows.ask_user_movement_v2(
        microscope,
        settings,
        image_settings,
        msg_type="eucentric",
        flat_to_sem=False,
    )

    # after eucentricity... we should be at 4mm,
    # so we should set wd to 4mm and link

    logging.info(f"{lamella.current_state.stage.name}: initial landing calibration complete.")

    ############################## LAND_LAMELLA ##############################
    calibration.validate_stage_height_for_needle_insertion(microscope, settings, image_settings)
    park_position = movement.move_needle_to_landing_position(microscope)

    #### Y-MOVE (ELECTRON)
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.beam_type=BeamType.ELECTRON
    image_settings.save=True
    image_settings.label=f"landing_needle_land_sample_lowres"

    det = calibration.validate_detection_v2(
        microscope,
        settings,
        image_settings,
        shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
        beam_type=BeamType.ELECTRON,
    )

    y_move = movement.y_corrected_needle_movement(
        det.distance_metres.y, stage.current_position.t
    )
    needle.relative_move(y_move)
    logging.info(f"{lamella.current_state.stage.name}: y-move complete: {y_move}")
    acquire.take_reference_images(microscope, image_settings)

    #### Z-MOVE (ION)
    
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.beam_type=BeamType.ION
    image_settings.save=True
    image_settings.label=f"landing_needle_land_sample_lowres_after_y_move"
    
    det = calibration.validate_detection_v2(
        microscope,
        settings,
        image_settings,
        shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
        beam_type=BeamType.ION,
    )

    # up is down
    z_distance = -det.distance_metres.y / np.sin(
        np.deg2rad(settings["system"]["stage_tilt_flat_to_ion"])
    )
    z_move = movement.z_corrected_needle_movement(z_distance, stage.current_position.t)
    needle.relative_move(z_move)
    logging.info(f"{lamella.current_state.stage.name}: z-move complete: {z_move}")

    acquire.take_reference_images(microscope, image_settings)

    # TODO: change this to use ion view...
    #### X-HALF-MOVE (ELECTRON)
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.beam_type=BeamType.ELECTRON
    image_settings.save=True
    image_settings.label=f"landing_needle_land_sample_lowres_after_z_move"

    det = calibration.validate_detection_v2(
        microscope,
        settings,
        image_settings,
        shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
        beam_type=BeamType.ELECTRON,
    )

    # half move
    x_move = movement.x_corrected_needle_movement(det.distance_metres.x / 2)
    needle.relative_move(x_move)
    logging.info(f"{lamella.current_state.stage.name}: x-half-move complete: {x_move}")
    acquire.take_reference_images(microscope, image_settings)

    # repeat final movement until user confirms landing
    response = False
    while response is False:
        #### X-MOVE
        image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"]
        image_settings.beam_type=BeamType.ELECTRON
        image_settings.save=True
        image_settings.label=f"landing_needle_land_sample_lowres_after_z_move"
        

        det = calibration.validate_detection_v2(
            microscope,
            settings,
            image_settings,
            shift_type=(DetectionType.LamellaEdge, DetectionType.LandingPost),
            beam_type=BeamType.ELECTRON,
        )

        x_move = movement.x_corrected_needle_movement(det.distance_metres.x)
        needle.relative_move(x_move)
        logging.info(f"{lamella.current_state.stage.name}: x-move complete: {x_move}")

        # final reference images
       
        image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"],
        image_settings.beam_type=BeamType.ELECTRON
        image_settings.save=True
        image_settings.label=f"landing_lamella_final_weld_highres",
        acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

        response = windows.ask_user_interaction_v2(
            microscope,
            settings,
            image_settings,
            msg="Has the lamella landed on the post? \nPress Yes to continue, or No to redo the final movement",
            beam_type=BeamType.ION,
        )

    #################################################################################################

    ############################## WELD TO LANDING POST #############################################

    windows.open_milling_window_v2(MillingPattern.Weld)

    # final reference images
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.save=True
    image_settings.label=f"landing_lamella_final_weld_highres"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    #################################################################################################

    ###################################### CUT_OFF_NEEDLE ######################################

    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"],
    image_settings.beam_type=BeamType.ION,
    image_settings.save=True,
    image_settings.label=f"landing_lamella_pre_cut_off",
   
    det = calibration.validate_detection_v2(
        microscope,
        settings,
        image_settings,
        shift_type=(DetectionType.NeedleTip, DetectionType.ImageCentre),
        beam_type=BeamType.ION,
    )

    # cut off needle
    windows.open_milling_window_v2(
        MillingPattern.Cut, x=det.distance_metres.x, y=det.distance_metres.y
    )

    ################################### REMOVE_NEEDLE ##########################################

    # reference images
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"],
    image_settings.beam_type=BeamType.ION,
    image_settings.save=True,
    image_settings.label=f"landing_lamella_final_cut_lowres",
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"],
    image_settings.label=f"landing_lamella_final_cut_highres",
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    logging.info(f"{lamella.current_state.stage.name}: removing needle from landing post")
    # move needle out of trench slowly at first
    for i in range(3):
        z_move_out_from_post = movement.z_corrected_needle_movement(
            10e-6, stage.current_position.t
        )
        needle.relative_move(z_move_out_from_post)
        logging.info(f"{lamella.current_state.stage.name}: moving needle out: {z_move_out_from_post} ({i + 1}/3)"
        )
        time.sleep(1)

    # move needle to park position
    movement.retract_needle(microscope, park_position)
    logging.info(f"{lamella.current_state.stage.name}: needle retracted.")

    # reference images
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"],
    image_settings.save=True
    image_settings.label=f"landing_lamella_final_lowres",
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.label=f"landing_lamella_final_highres"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    return lamella




def reset_needle(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella
) -> Lamella:

    # convienence
    stage = microscope.specimen.stage
    needle = microscope.specimen.manipulator

    # move sample stage out
    movement.move_sample_stage_out(microscope)
    logging.info(f"{lamella.current_state.stage.name}: moved sample stage out")

    ###################################### SHARPEN_NEEDLE ######################################

    # move needle in
    park_position = movement.insert_needle(microscope)
    z_move_in = movement.z_corrected_needle_movement(
        -180e-6, stage.current_position.t
    )
    needle.relative_move(z_move_in)
    logging.info(f"{lamella.current_state.stage.name}: insert needle for reset")

    # needle images
    image_settings.save=True
    image_settings.label=f"sharpen_needle_initial"    
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    det = calibration.validate_detection_v2(
        microscope,
        settings,
        image_settings,
        shift_type=(DetectionType.NeedleTip, DetectionType.ImageCentre),
        beam_type=BeamType.ION,
    )

    x_move = movement.x_corrected_needle_movement(det.distance_metres.x)
    needle.relative_move(x_move)
    z_distance = -det.distance_metres.y / np.sin(
        np.deg2rad(settings["system"]["stage_tilt_flat_to_ion"])
    )
    z_move = movement.z_corrected_needle_movement(z_distance, stage.current_position.t)
    needle.relative_move(z_move)
    logging.info(f"{lamella.current_state.stage.name}: moving needle to centre: x_move: {x_move}, z_move: {z_move}")

    image_settings.label = f"sharpen_needle_centre"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    det = calibration.validate_detection_v2(
        microscope,
        settings,
        image_settings,
        shift_type=(DetectionType.NeedleTip, DetectionType.ImageCentre),
        beam_type=BeamType.ION,
    )

    # create sharpening patterns
    windows.open_milling_window_v2(
        MillingPattern.Sharpen, x=det.distance_metres.x, y=det.distance_metres.y
    )

    #################################################################################################

    # take reference images
    image_settings.label = f"sharpen_needle_final"
    image_settings.save = True
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    # retract needle
    movement.retract_needle(microscope, park_position)

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
    lamella: Lamella
) -> Lamella:

    # move to the initial landing coordinates
    movement.safe_absolute_stage_movement(
        microscope=microscope,
        stage_position=lamella.landing_coordinates,
    )

    # ensure_eucentricity # TODO: Maybe remove, not required?
    windows.ask_user_movement_v2(
        microscope,
        settings,
        image_settings,
        msg_type="eucentric",
        flat_to_sem=False,
    )

    # rotate_and_tilt_to_thinning_angle
    image_settings.hfw = settings["calibration"]["reference_images"]["hfw_high_res"]
    movement.move_to_thinning_angle(microscope=microscope, settings=settings)

    # ensure_eucentricity at thinning angle
    windows.ask_user_movement_v2(
        microscope,
        settings,
        image_settings,
        msg_type="eucentric",
        flat_to_sem=False,
    )

    # lamella images
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_med_res"]
    image_settings.save=True
    image_settings.label=f"thin_lamella_0_deg_tilt"
    acquire.take_reference_images(microscope, image_settings)

    # # realign lamella to image centre
    # image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"],
    # image_settings.save=True,
    # image_settings.label=f"thin_drift_correction_medres"
    # acquire.take_reference_images(microscope, image_settings)

 
    image_settings.hfw=settings["calibration"]["reference_images"]["high_super_res"]
    image_settings.save=False    
    windows.ask_user_movement_v2(
        microscope, settings, image_settings, msg_type="alignment"
    )

    # take reference images
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.save=True
    image_settings.label=f"thin_drift_correction_highres"
    acquire.take_reference_images(microscope, image_settings)

    # thin_lamella (align and mill)
    image_settings.resolution=settings["protocol"]["thin_lamella"]["resolution"],
    image_settings.dwell_time=settings["protocol"]["thin_lamella"]["dwell_time"],
    image_settings.hfw=settings["thin_lamella"]["hfw"],
    
    # windows.open_milling_window_v2(MillingPattern.Fiducial)
    windows.open_milling_window_v2(MillingPattern.Thin)

    # take reference images
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_ultra_res"]
    image_settings.save=True
    image_settings.label=f"thin_lamella_post_superres"

    acquire.take_reference_images(microscope=microscope, image_settings=image_settings
    )

    return

def polish_lamella(    
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    lamella: Lamella
) -> Lamella:

    # restore state from thinning stage
    # ref_image = self.current_sample_position.load_reference_image("thin_lamella_crosscorrelation_ref_ib")

    # realign lamella to image centre
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"]
    image_settings.save=True
    image_settings.label=f"polish_drift_correction_medres"
    acquire.take_reference_images(microscope, image_settings)

    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"],
    image_settings.save=False

    # confirm
    windows.ask_user_movement_v2(microscope, settings, image_settings, msg_type="alignment")

    # take reference images
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"]
    image_settings.save=True
    image_settings.label=f"polish_drift_correction_highres"
    acquire.take_reference_images(microscope, image_settings)

    # polish (align and mill)
    image_settings.resolution=settings["protocol"]["polish_lamella"]["resolution"],
    image_settings.dwell_time=settings["protocol"]["polish_lamella"]["dwell_time"],
    image_settings.hfw=settings["protocol"]["polish_lamella"]["hfw"],
    windows.open_milling_window_v2(MillingPattern.Polish)

    # take reference images (ultra, super, high)
    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_ultra_res"]
    image_settings.save=True
    image_settings.label=f"polish_lamella_post_ultra_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_super_res"],
    image_settings.label=f"polish_lamella_post_super_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    image_settings.hfw=settings["calibration"]["reference_images"]["hfw_high_res"],
    image_settings.label=f"polish_lamella_post_high_res"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)

    logging.info(f"{lamella.current_state.stage.name}: polish lamella {lamella._petname} complete.")

    return lamella










# TODO: might need to pass the parent to these as well?


def setup_autoliftout():
    #TODO:
    return

def run_autoliftout_workflow(
    microscope: SdbMicroscopeClient, 
    settings: dict,  
    image_settings: ImageSettings,
    sample: Sample):

    HIGH_THROUGHPUT = False

        # autoliftout_workflow
    autoliftout_stages = {
        AutoLiftoutStage.Setup: setup_autoliftout,
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
    # if HIGH_THROUGHPUT:
    #     for terminal_stage in [
    #         AutoLiftoutStage.MillTrench,
    #         AutoLiftoutStage.MillJCut,
    #     ]:
    #         for sp in self.samples:
    #             self.current_sample_position = sp

    #             while (
    #                 sp.microscope_state.last_completed_stage.value
    #                 < terminal_stage.value
    #             ):

    #                 next_stage = AutoLiftoutStage(
    #                     sp.microscope_state.last_completed_stage.value + 1
    #                 )

    #                 # reset to the previous state
    #                 self.start_of_stage_update(next_stage=next_stage)

    #                 # run the next workflow stage
    #                 self.autoliftout_stages[next_stage]()

    #                 # advance workflow
    #                 self.end_of_stage_update(eucentric=True)

    # standard workflow
    for lamella_no, lamella in sample.positions.items():
        # self.current_sample_position = sp

        while (lamella.current_state.microscope_state.last_completed_stage.value < AutoLiftoutStage.Reset.value):

            next_stage = AutoLiftoutStage(
                lamella.current_state.microscope_state.last_completed_stage.value + 1
            )
            msg = (
                f"The last completed stage for lamella {(lamella._petname)})\nis {lamella.current_state.microscope_state.last_completed_stage.name}. "
                f"\nWould you like to continue from {next_stage.name}?\n"
            )
            response = windows.ask_user_interaction_v2(
                microscope,
                settings,
                image_settings,
                msg=msg,
                beam_type=BeamType.ION,
            )

            if response:

                # reset to the previous state
                start_of_stage_update(next_stage=next_stage)

                # run the next workflow stage
                lamella = autoliftout_stages[next_stage](microscope=microscope, settings=settings, image_settings=image_settings, lamella=lamella)

                # advance workflow
                end_of_stage_update(eucentric=True)
            else:
                break  # go to the next sample





def end_of_stage_update(self, eucentric: bool) -> None:
    """Save the current microscope state configuration to disk, and log that the stage has been completed."""
    # save state information
    microscope_state = calibration.get_current_microscope_state(
        microscope=self.microscope, stage=self.current_stage, eucentric=eucentric
    )
    self.current_sample_position.microscope_state = microscope_state
    self.current_sample_position.save_data()

    # update ui
    self.update_scroll_ui()

    logging.info(
        f"{self.current_sample_position.sample_id} | {self.current_stage.name} | FINISHED"
    )

    return

def start_of_stage_update(self, next_stage: AutoLiftoutStage) -> None:
    """Check the last completed stage and reload the microscope state if required. Log that the stage has started. """
    last_completed_stage = (
        self.current_sample_position.microscope_state.last_completed_stage
    )

    if last_completed_stage.value == next_stage.value - 1:
        logging.info(
            f"{self.current_sample_position.sample_id} restarting from end of stage: {last_completed_stage.name}"
        )
        calibration.set_microscope_state(
            self.microscope, self.current_sample_position.microscope_state
        )

    self.current_stage = next_stage
    logging.info(
        f"{self.current_sample_position.sample_id} | {self.current_stage.name}  | STARTED"
    )

    self.update_status()

    return