import liftout
import os

base_path = os.path.dirname(liftout.__file__)
system_config = os.path.join(base_path, "config", "system.yaml")
calibration_config = os.path.join(base_path, "config", "calibration.yaml")
protocol_config = os.path.join(base_path, "protocol", "protocol.yaml")