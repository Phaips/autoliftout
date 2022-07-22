
    def select_sample_positions_piescope(self, initialisation=False):
        import piescope_gui.main

        if initialisation:
            # get the current sample positions..
            select_another_sample_position = self.get_current_sample_positions()

            if select_another_sample_position is False:
                return

            if self.piescope_gui_main_window is None:
                self.piescope_gui_main_window = piescope_gui.main.GUIMainWindow(
                    parent_gui=self
                )
                self.piescope_gui_main_window.window_close.connect(
                    lambda: self.finish_select_sample_positions_piescope()
                )

        if self.piescope_gui_main_window:
            # continue selecting points
            self.piescope_gui_main_window.milling_position = None
            self.piescope_gui_main_window.show()


    def get_initial_lamella_position_piescope(self):
        """Select the initial sample positions for liftout"""
        sample_position = SamplePosition(
            data_path=self.save_path, sample_no=self.sample_no
        )

        movement.safe_absolute_stage_movement(
            self.microscope, self.piescope_gui_main_window.milling_position
        )

        # save lamella coordinates
        sample_position.lamella_coordinates = StagePosition(
            x=float(self.piescope_gui_main_window.milling_position.x),
            y=float(self.piescope_gui_main_window.milling_position.y),
            z=float(self.piescope_gui_main_window.milling_position.z),
            r=float(self.piescope_gui_main_window.milling_position.r),
            t=float(self.piescope_gui_main_window.milling_position.t),
            coordinate_system=str(
                self.piescope_gui_main_window.milling_position.coordinate_system
            ),
        )
        # save microscope state
        sample_position.microscope_state = calibration.get_current_microscope_state(
            microscope=self.microscope, stage=self.current_stage, eucentric=True
        )
        sample_position.save_data()

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_med_res"],
            save=True,
            save_path=os.path.join(self.save_path, str(sample_position.sample_id)),
            label=f"ref_lamella_low_res",
        )
        acquire.take_reference_images(
            self.microscope, image_settings=self.image_settings
        )

        self.update_image_settings(
            hfw=self.settings["calibration"]["reference_images"]["hfw_super_res"],
            save=True,
            save_path=os.path.join(self.save_path, str(sample_position.sample_id)),
            label="ref_lamella_high_res",
        )
        acquire.take_reference_images(
            self.microscope, image_settings=self.image_settings
        )

        return sample_position

    def finish_select_sample_positions_piescope(self):

        try:
            self.piescope_gui_main_window.milling_window.hide()
            self.piescope_gui_main_window.hide()
            time.sleep(1)
        except:
            logging.error("Unable to close the PIEScope windows?")

        if self.piescope_gui_main_window.milling_position is not None:
            # get the lamella milling position from piescope...
            sample_position = self.get_initial_lamella_position_piescope()
            self.samples.append(sample_position)
            self.sample_no += 1

        finished_selecting = windows.ask_user_interaction_v2(
            self.microscope,
            msg=f"Do you want to select landing positions?\n"
            f"{len(self.samples)} positions selected so far.",
        )

        self.update_scroll_ui()

        # enable adding more samples with piescope
        if self.samples:
            self.pushButton_add_sample_position.setVisible(True)
            self.pushButton_add_sample_position.setEnabled(True)

        if finished_selecting:

            # only select landing positions for liftout
            if not self.AUTOLAMELLA_ENABLED:
                self.select_landing_positions()

            self.finish_setup()




####################### AUTOLAMELLA ##########
    def enable_autolamella(self):
        self.AUTOLAMELLA_ENABLED = True
        self.pushButton_autoliftout.setText("Run AutoLamella")
        self.label_title.setText("AutoLamella")
        self.pushButton_initialise.setText("Setup Autolamella")
        self.pushButton_thinning.setVisible(False)

        # connect autolamella workflow
        self.pushButton_autoliftout.disconnect()
        self.pushButton_autoliftout.clicked.connect(self.run_autolamella_workflow)

    def enable_autoliftout(self):
        self.AUTOLAMELLA_ENABLED = False
        self.pushButton_autoliftout.setText("Run AutoLiftout")
        self.label_title.setText("AutoLiftout")
        self.pushButton_initialise.setText("Setup AutoLiftout")
        self.pushButton_thinning.setVisible(True)

        # connect autoliftout workflow
        self.pushButton_autoliftout.disconnect()
        self.pushButton_autoliftout.clicked.connect(self.run_autoliftout)









# UI STUFF

########
# table options..
# TODO: get current row
# add images...

from PyQt5 import QtCore
ref https://learndataanalysis.org/display-pandas-dataframe-with-pyqt5-qtableview-widget/
class pandasModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None

from liftout.fibsem.sample import sample_to_dataframe
df = sample_to_dataframe(self.sample)
model = pandasModel(df)
view = QtWidgets.QTableView()
view.setModel(model)

view.horizontalHeader().setStretchLastSection(True)
view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
gridLayout = QtWidgets.QGridLayout()
gridLayout.addWidget(view, 1, 0)
###################