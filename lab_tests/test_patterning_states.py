import time
from autoscript_sdb_microscope_client import SdbMicroscopeClient

microscope = SdbMicroscopeClient()
microscope.connect("localhost")
sputter_time = 20  # integer number of seconds
sputter_application_file = 'Si'

microscope.imaging.set_active_view(1)  # the electron beam view
microscope.patterning.clear_patterns()
microscope.patterning.set_default_application_file(sputter_application_file)
# Run sputtering
start_x = 0
start_y = 0
end_x = 1e-6
end_y = 1e-6
depth = 2e-6
microscope.patterning.create_line(start_x, start_y, end_x, end_y, depth)  # 1um, at zero in the FOV
microscope.beams.electron_beam.blank()
print('Original patterning state: {}'.format(microscope.patterning.state))
microscope.patterning.start()
for i in range(int((sputter_time))):
    print('Patterning state at second {}: {}'.format(i, microscope.patterning.state))
    time.sleep(1)
print("Patterning state afterwards: {}".format(microscope.patterning.state))
if microscope.patterning.state == "Running":
    microscope.patterning.stop()
print("Final patterning state: {}".format(microscope.patterning.state))
