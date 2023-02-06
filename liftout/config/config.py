import os

import liftout

BASE_PATH = os.path.dirname(liftout.__file__)
config_path = os.path.join(BASE_PATH, "config")
protocol_path = os.path.join(BASE_PATH, "protocol", "protocol.yaml")

LOG_DATA_PATH = os.path.join(BASE_PATH, "log_data")

os.makedirs(LOG_DATA_PATH, exist_ok=True)



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





DISPLAY_REFERENCE_FNAMES = [
    "ref_lamella_low_res_ib",
    "ref_trench_high_res_ib",
    "ref_jcut_high_res_ib",
    "ref_liftout_sever_ib",
    "ref_landing_lamella_high_res_ib",
    "ref_reset_high_res_ib",
    "ref_thin_lamella_ultra_res_ib",
    "ref_polish_lamella_ultra_res_ib",
]