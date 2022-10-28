import logging
from pprint import pprint


from fibsem import utils as fibsem_utils
from fibsem import calibration
from liftout.config import config
from liftout.gui.qtdesigner_files import AutoLiftoutProtocolUI 
from PyQt5 import  QtWidgets

from liftout.structures import AutoLiftoutMode

import napari 
from napari.utils import notifications

class AutoLiftoutProtocolUI(AutoLiftoutProtocolUI.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, viewer: napari.Viewer=None, protocol: dict = None):
        super(AutoLiftoutProtocolUI, self).__init__()

        # setup ui
        self.setupUi(self)

        self.setup_connections()

        if protocol is not None:
            self.update_ui_from_protocol(protocol)

    def setup_connections(self):

        print("setup connections")

        # options
        self.comboBox_options_liftout_joining_method.addItems(["None", "Weld", "Platinum"])
        self.comboBox_options_contact_direction.addItems(["Horizontal", "Vertical"])
        self.comboBox_options_landing_surface.addItems(["Half-Moon Grid"])
        self.comboBox_options_landing_joining_method.addItems(["None", "Weld", "Platinum"])
        
        # automation
        modes = [mode.name for mode in AutoLiftoutMode]
        self.comboBox_auto_mill_trench.addItems(modes)
        self.comboBox_auto_mill_jcut.addItems(modes)
        self.comboBox_auto_liftout.addItems(modes)
        self.comboBox_auto_landing.addItems(modes)
        self.comboBox_auto_reset.addItems(modes)
        self.comboBox_auto_thinning.addItems(modes)
        self.comboBox_auto_polishing.addItems(modes)

        # milling


        # buttons
        self.pushButton_save.clicked.connect(self.update_protocol_from_ui)

    def update_protocol_from_ui(self):
        

        # TODO: milling?

        self.protocol["options"] = {
            "batch_mode": self.checkBox_options_batch_mode.isChecked(),
            "confirm_advance": self.checkBox_options_confirm_next_stage.isChecked(),
            "liftout_joining_method": self.comboBox_options_liftout_joining_method.currentText(),
            "liftout_contact_direction": self.comboBox_options_contact_direction.currentText(),
            "landing_surface": self.comboBox_options_landing_surface.currentText(),
            "landing_joining_method": self.comboBox_options_landing_joining_method.currentText(),
            "auto": {
                "trench": self.comboBox_auto_mill_trench.currentText(),
                "jcut": self.comboBox_auto_mill_jcut.currentText(),
                "liftout": self.comboBox_auto_liftout.currentText(),
                "landing": self.comboBox_auto_landing.currentText(),
                "reset": self.comboBox_auto_reset.currentText(),
                "thin": self.comboBox_auto_thinning.currentText(),
                "polish": self.comboBox_auto_polishing.currentText(),
            }
        }

        from pprint import pprint
        print("update protocol")

        pprint(self.protocol)

        # TODO: save to file, update or overwrite?
        # TODO: need the rest of the protocol, how to represent


    def update_ui_from_protocol(self, protocol: dict):

        self.protocol = protocol

        options = self.protocol["options"]
        
        # options
        self.checkBox_options_batch_mode.setChecked(bool(options["batch_mode"]))
        self.checkBox_options_confirm_next_stage.setChecked(bool(options["confirm_advance"]))
        self.comboBox_options_liftout_joining_method.setCurrentText(options["liftout_joining_method"])
        self.comboBox_options_contact_direction.setCurrentText(options["liftout_contact_direction"])
        self.comboBox_options_landing_surface.setCurrentText(options["landing_surface"])
        self.comboBox_options_landing_joining_method.setCurrentText(options["landing_joining_method"])

        # automation
        self.comboBox_auto_mill_trench.setCurrentText(options["auto"]["trench"].upper())
        self.comboBox_auto_mill_jcut.setCurrentText(options["auto"]["jcut"].upper())
        self.comboBox_auto_liftout.setCurrentText(options["auto"]["liftout"].upper())
        self.comboBox_auto_landing.setCurrentText(options["auto"]["landing"].upper())
        self.comboBox_auto_reset.setCurrentText(options["auto"]["reset"].upper())
        self.comboBox_auto_thinning.setCurrentText(options["auto"]["thin"].upper())
        self.comboBox_auto_polishing.setCurrentText(options["auto"]["polish"].upper())


        # TODO: platinum
        # TODO: initial positions


def main():
    viewer = napari.Viewer()
    protocol = fibsem_utils.load_protocol(config.protocol_path)
    autoliftout_protocol_ui = AutoLiftoutProtocolUI(viewer=viewer, protocol=protocol)
    viewer.window.add_dock_widget(autoliftout_protocol_ui, area="right", add_vertical_stretch=False)
    napari.run()

if __name__ == "__main__":
    main()