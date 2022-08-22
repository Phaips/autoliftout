import os

import liftout
from liftout.patterning import MillingPattern
from liftout.detection.utils import DetectionType

base_path = os.path.dirname(liftout.__file__)
config_path = os.path.join(base_path, "config")
protocol_path = os.path.join(base_path, "protocol", "protocol.yaml")


# MILLING UI

NON_CHANGEABLE_MILLING_PARAMETERS = [
    "milling_current",
    "hfw",
    "jcut_angle",
    "rotation_angle",
    "tilt_angle",
    "tilt_offset",
    "resolution",
    "dwell_time",
    "reduced_area",
    "scan_direction",
    "cleaning_cross_section",
]
NON_SCALED_MILLING_PARAMETERS = [
    "size_ratio",
    "rotation",
    "tip_angle",
    "needle_angle",
    "percentage_roi_height",
    "percentage_from_lamella_surface",
]

# # sputtering rates
# MILLING_SPUTTER_RATE = {
#     20e-12: 6.85e-3,  # 30kv
#     0.2e-9: 6.578e-2,  # 30kv
#     0.74e-9: 3.349e-1,  # 30kv
#     0.89e-9: 3.920e-1,  # 20kv
#     2.0e-9: 9.549e-1,  # 30kv
#     2.4e-9: 1.309,  # 20kv
#     6.2e-9: 2.907,  # 20kv
#     7.6e-9: 3.041,  # 30kv
# }
# # 0.89nA : 3.920e-1 um3/s
# # 2.4nA : 1.309e0 um3/s
# # 6.2nA : 2.907e0 um3/s # from microscope application files

# # 30kV
# # 7.6nA: 3.041e0 um3/s


PATTERN_PROTOCOL_MAP = {
    MillingPattern.Trench: "lamella",
    MillingPattern.JCut: "jcut",
    MillingPattern.Sever: "sever",
    MillingPattern.Weld: "weld",
    MillingPattern.Cut: "cut",
    MillingPattern.Sharpen: "sharpen",
    MillingPattern.Thin: "thin_lamella",
    MillingPattern.Polish: "polish_lamella",
    MillingPattern.Flatten: "flatten_landing",
    MillingPattern.Fiducial: "fiducial",
}









# DETECTIONS

DETECTION_TYPE_COLOURS = {
    DetectionType.LamellaCentre: (1, 0, 0, 1),
    DetectionType.NeedleTip: (0, 1, 0, 1),
    DetectionType.LamellaEdge: (1, 0.5, 0, 1),
    DetectionType.LandingPost: (0, 1, 1, 1),
    DetectionType.ImageCentre: (1, 1, 1, 1)
}



DISPLAY_REFERENCE_FNAMES = [
    "ref_lamella_low_res_ib",
    "ref_trench_high_res_ib",
    "jcut_highres_ib",
    "needle_liftout_landed_highres_ib",
    "landing_lamella_final_highres_ib",
    "sharpen_needle_final_ib",
    "thin_lamella_post_superres_ib",
    "polish_lamella_post_superres_ib",
]