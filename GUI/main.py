from GUI.qtdesigner_files import main as gui_main
from GUI import fibsem as fibsem

from PyQt5 import QtWidgets, QtGui, QtCore
import scipy.ndimage as ndi
import numpy as np
import qimage2ndarray
import traceback
import logging
import skimage
import copy
import os
import sys
import mock
from tkinter import Tk, filedialog
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _NavigationToolbar

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

_translate = QtCore.QCoreApplication.translate
logger = logging.getLogger(__name__)

key_list_protocol = [
    "demo_mode",
    "system",
    "ip_address",
    "application_file_rectangle",
    "application_file_cleaning_cross_section",
    "imaging",
    "horizontal_field_width",
    "dwell_time",
    "resolution",
    "lamella",
    "lamella_width",
    "lamella_height",
    "total_cut_height",
    "milling_depth",
    "milling_current",
    "protocol_stages",
    "percentage_roi_height",
    "percentage_from_lamella_surface",
    "milling_current",
    "percentage_from_lamella_surface",
    "jcut",
    "jcut_milling_current",
    "jcut_angle",
    "jcut_length",
    "jcut_lamella_depth",
    "jcut_trench_thickness",
    "jcut_milling_depth",
    "extra_bit",
    "mill_lhs_jcut_pattern",
    "mill_rhs_jcut_pattern",
    "mill_top_jcut_pattern",
]

protocol_template_path = '..\\protocol_liftout.yml'
starting_positions = 6
information_keys = ['x', 'y', 'z', 'Θx', 'Θy', 'Θz']


class GUIMainWindow(gui_main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, ip_address="10.0.0.1", offline=False):
        super(GUIMainWindow, self).__init__()
        self.offline = offline
        self.setupUi(self)

        self.new_protocol()
        self.load_template_protocol()
        self.wp_SEM = None
        self.initialise_images()

        self.setup_connections()

        self.params = {}
        for parameter in range(len(information_keys)):
            self.params[information_keys[parameter]] = 0

        self.position = 0

        self.save_destination = None

        self.ip_address = ip_address
        self.microscope = None
        self.detector = None
        self.lasers = None
        self.objective_stage = None
        self.initialize_hardware(offline=offline)

        self.image_ion = None  # ion beam image (AdornedImage type) kept for milling window
        self.image_lm = None   # Fluorescence microscope image (numpy array, integer pixel values)
        self.image_volume = None
        self.image_fibsem = None  # AdornedImage (for whatever is currently displayed in the FIBSEM side of the piescope GUI)

        # self.DEFAULT_PATH = os.path.normpath(os.path.expanduser('~/Pictures/PIESCOPE'))
        self.setWindowTitle("Autoliftout User Interface Main Window")
        self.statusbar.setSizeGripEnabled(0)
        self.status = QtWidgets.QLabel(self.statusbar)
        self.status.setAlignment(QtCore.Qt.AlignRight)
        self.statusbar.addPermanentWidget(self.status, 1)

        # self.label_objective_stage_position.setText("Unknown")
        self.comboBox_resolution.setCurrentIndex(2)  # resolution "3072x2048"

        # self.liveCheck is True when ready to start live imaging,
        # and False while live imaging is running:
        # self.liveCheck = True # TODO: livecheck on the stopping mid working

        self.camera_settings = None

        self.pixmap_SEM = None
        self.pixmap_FIB = None
        # self.save_destination_FM = self.DEFAULT_PATH
        # self.save_destination_FIBSEM = self.DEFAULT_PATH
        # self.save_destination_correlation = self.DEFAULT_PATH

    def initialise_images(self):
        import matplotlib.pyplot as plt

        self.figure_SEM = plt.figure()
        self.canvas_SEM = _FigureCanvas(self.figure_SEM)
        self.toolbar_SEM = _NavigationToolbar(self.canvas_SEM, self)
        self.label_SEM.setLayout(QtWidgets.QVBoxLayout())
        self.label_SEM.layout().addWidget(self.toolbar_SEM)
        self.label_SEM.layout().addWidget(self.canvas_SEM)

        self.figure_FIB = plt.figure()
        self.canvas_FIB = _FigureCanvas(self.figure_FIB)
        self.toolbar_FIB = _NavigationToolbar(self.canvas_FIB, self)
        self.label_FIB.setLayout(QtWidgets.QVBoxLayout())
        self.label_FIB.layout().addWidget(self.toolbar_FIB)
        self.label_FIB.layout().addWidget(self.canvas_FIB)

    def initialize_hardware(self, offline=False):
        if offline is False:
            self.connect_to_fibsem_microscope(ip_address=self.ip_address)
            self.objective_stage = self.initialize_objective_stage()
        elif offline is True:
            pass
            # self.connect_to_fibsem_microscope(ip_address="localhost")

    def setup_connections(self):

        self.pushButton_Protocol_Load.clicked.connect(lambda: self.load_yaml())
        self.pushButton_Protocol_New.clicked.connect(lambda: self.new_protocol())
        self.pushButton_Protocol_Delete.clicked.connect(lambda: self.delete_protocol())
        self.pushButton_Protocol_Save.clicked.connect(lambda: self.save_protocol(self.save_destination))
        self.pushButton_Protocol_Save_As.clicked.connect(lambda: self.save_protocol())
        self.pushButton_Protocol_Rename.clicked.connect(lambda: self.rename_protocol())
        self.tabWidget_Protocol.tabBarDoubleClicked.connect(lambda: self.rename_protocol())

        self.tabWidget_Protocol.tabBar().tabMoved.connect(lambda: self.tab_moved('protocol'))
        self.tabWidget_Information.tabBar().tabMoved.connect(lambda: self.tab_moved('information'))

        self.pushButton_random_data.clicked.connect(lambda: self.random_data())

        # FIBSEM methods
        self.comboBox_resolution.currentTextChanged.connect(
            lambda: self.update_fibsem_settings())
        self.lineEdit_dwell_time.textChanged.connect(
            lambda: self.update_fibsem_settings())


        # self.actionOpen_FIBSEM_Image.triggered.connect(
        #     lambda: self.open_images("FIBSEM"))
        #
        # self.actionSave_FIBSEM_Image.triggered.connect(
        #     lambda: self.save_image("FIBSEM"))

        self.button_get_image_FIB.clicked.connect(
            lambda: self.get_FIB_image())
        self.button_get_image_SEM.clicked.connect(
            lambda: self.get_SEM_image())
        self.button_last_image_FIB.clicked.connect(
            lambda: self.get_last_FIB_image())
        self.button_last_image_SEM.clicked.connect(
            lambda: self.get_last_SEM_image())

        self.connect_microscope.clicked.connect(
            lambda: self.connect_to_fibsem_microscope(ip_address=self.ip_address))
        self.to_light_microscope.clicked.connect(
            lambda: self.move_to_light_microscope())
        self.to_electron_microscope.clicked.connect(
            lambda: self.move_to_electron_microscope())

    def new_protocol(self):
        num_index = self.tabWidget_Protocol.__len__() + 1

        # new tab for protocol
        new_protocol_tab = QtWidgets.QWidget()
        new_protocol_tab.setObjectName(f'Protocol {num_index}')
        layout_protocol_tab = QtWidgets.QGridLayout(new_protocol_tab)
        layout_protocol_tab.setObjectName(f"gridLayout_{num_index}")

        # new text edit to hold protocol
        protocol_text_edit = QtWidgets.QTextEdit()
        font = QtGui.QFont()
        font.setPointSize(10)
        protocol_text_edit.setFont(font)
        protocol_text_edit.setObjectName(f"protocol_text_edit_{num_index}")
        layout_protocol_tab.addWidget(protocol_text_edit, 0, 0, 1, 1)

        self.tabWidget_Protocol.addTab(new_protocol_tab, f'Protocol {num_index}')
        self.tabWidget_Protocol.setCurrentWidget(new_protocol_tab)
        self.load_template_protocol()

        #

        # new tab for information from FIBSEM
        new_information_tab = QtWidgets.QWidget()
        new_information_tab.setObjectName(f'Protocol {num_index}')
        layout_new_information_tab = QtWidgets.QGridLayout(new_information_tab)
        layout_new_information_tab.setObjectName(f"layout_new_information_tab_{num_index}")

        # new empty table for information to fill
        new_table_widget = QtWidgets.QTableWidget()
        new_table_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        new_table_widget.setAlternatingRowColors(True)
        new_table_widget.setObjectName(f"tableWidget_{num_index}")
        new_table_widget.setColumnCount(6)
        new_table_widget.setRowCount(starting_positions)

        # set up rows
        for row in range(starting_positions):
            table_item = QtWidgets.QTableWidgetItem()
            new_table_widget.setVerticalHeaderItem(row, table_item)
            item = new_table_widget.verticalHeaderItem(row)
            item.setText(_translate("MainWindow", f"Position {row+1}"))

        # set up columns
        for column in range(len(information_keys)):
            table_item = QtWidgets.QTableWidgetItem()
            new_table_widget.setHorizontalHeaderItem(column, table_item)
            item = new_table_widget.horizontalHeaderItem(column)
            item.setText(_translate("MainWindow", information_keys[column]))

        new_table_widget.horizontalHeader().setDefaultSectionSize(174)
        new_table_widget.horizontalHeader().setHighlightSections(True)

        layout_new_information_tab.addWidget(new_table_widget, 0, 0, 1, 1)
        self.tabWidget_Information.addTab(new_information_tab, f'Protocol {num_index}')
        self.tabWidget_Information.setCurrentWidget(new_information_tab)

    def rename_protocol(self):
        index = self.tabWidget_Protocol.currentIndex()

        top_margin = 4
        left_margin = 10

        rect = self.tabWidget_Protocol.tabBar().tabRect(index)
        self.edit = QtWidgets.QLineEdit(self.tabWidget_Protocol)
        self.edit.move(rect.left() + left_margin, rect.top() + top_margin)
        self.edit.resize(rect.width() - 2 * left_margin, rect.height() - 2 * top_margin)
        self.edit.show()
        self.edit.setFocus()
        self.edit.selectAll()
        self.edit.editingFinished.connect(lambda: self.finish_rename())

    def finish_rename(self):
        self.tabWidget_Protocol.setTabText(self.tabWidget_Protocol.currentIndex(), self.edit.text())
        tab = self.tabWidget_Protocol.currentWidget()
        tab.setObjectName(self.edit.text())

        self.tabWidget_Information.setTabText(self.tabWidget_Information.currentIndex(), self.edit.text())
        tab2 = self.tabWidget_Information.currentWidget()
        tab2.setObjectName(self.edit.text())
        self.edit.deleteLater()

    def delete_protocol(self):
        index = self.tabWidget_Protocol.currentIndex()
        self.tabWidget_Information.removeTab(index)
        self.tabWidget_Protocol.removeTab(index)
        pass

    def load_template_protocol(self):
        with open(protocol_template_path, "r") as file:
            _dict = yaml.safe_load(file)
        checked = self.check_yaml(_dict)
        if not checked:
            print(f'Unexpected parameter in template file')
        self.load_protocol_text(_dict)

    def load_protocol_text(self, dictionary):
        protocol_text = str()
        count = 0
        jcut = 0
        _dict = dictionary

        for key, value in _dict.items():
            if type(value) is dict:
                if jcut == 1:
                    protocol_text += f'{key}:'  # jcut
                    jcut = 0
                else:
                    protocol_text += f'\n{key}:'  # first level excluding jcut
                for key2, value2 in value.items():
                    if type(value2) is list:
                        jcut = 1
                        protocol_text += f'\n  {key2}:'  # protocol stages
                        for item in value2:
                            if count == 0:
                                protocol_text += f'\n    # rough_cut'
                                count = 1
                                count2 = 0
                            else:
                                protocol_text += f'    # regular_cut'
                                count = 0
                                count2 = 0
                            protocol_text += f'\n    -'
                            for key6, value6 in item.items():
                                if count2 == 0:
                                    protocol_text += f' {key6}: {value6}\n'  # first after list
                                    count2 = 1
                                else:
                                    protocol_text += f'      {key6}: {value6}\n'  # rest of list
                    else:
                        protocol_text += f'\n  {key2}: {value2}'  # values not in list
            else:
                protocol_text += f'{key}: {value}'  # demo mode
        self.tabWidget_Protocol.currentWidget().findChild(QtWidgets.QTextEdit).setText(protocol_text)

    def save_protocol(self, destination=None):
        dest = destination
        if dest is None:
            dest = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder to save protocol in')
            index = self.tabWidget_Protocol.currentIndex()
            tab = self.tabWidget_Protocol.tabBar().tabText(index)
            dest = f'{dest}/{tab}.yml'
        protocol_text = self.tabWidget_Protocol.currentWidget().findChild(QtWidgets.QTextEdit).toPlainText()
        # protocol_text = self.tabWidget_Protocol.currentWidget().findChild(QtWidgets.QTextEdit).toMarkdown()
        print(protocol_text)
        p = yaml.safe_load(protocol_text)
        print(p)
        with open(dest, 'w') as file:
            yaml.dump(p, file, sort_keys=False)
        self.save_destination = dest
        # save

    def load_yaml(self):
        """Ask the user to choose a protocol file to load

        Returns
        -------
        str
            Path to file for parameter loading
        """
        checked = False
        print(f"Please select protocol file (yml)")
        root = Tk()
        root.withdraw()
        _dict = None
        while not checked:
            try:
                load_directory = filedialog.askopenfile(mode='r', filetypes =[('yml files', '*.yml')])
                if load_directory is None:
                    return

                while not load_directory.name.endswith('.yml'):
                    print('Not a yml configuration file')
                    load_directory = filedialog.askopenfile(mode='r', filetypes=[('yml files', '*.yml')])

                with open(load_directory.name, "r") as file:
                    _dict = yaml.safe_load(file)
                    file.close()

            except Exception as e:
                display_error_message(traceback.format_exc())

            checked = self.check_yaml(_dict)
            if not checked:
                print(f'Unexpected parameter in protocol file')

        root.destroy()
        # print(_dict)

        self.load_protocol_text(_dict)

    def check_yaml(self, dict):
        for key in dict:
            if key not in key_list_protocol:
                return False
        return True

    def tab_moved(self, moved):
        if moved == 'protocol':
            self.tabWidget_Information.tabBar().moveTab(self.tabWidget_Information.currentIndex(), self.tabWidget_Protocol.currentIndex())
        elif moved == 'information':
            self.tabWidget_Protocol.tabBar().moveTab(self.tabWidget_Protocol.currentIndex(), self.tabWidget_Information.currentIndex())

    def random_data(self):
        for parameter in range(len(self.params)):
            self.params[information_keys[parameter]] = np.random.rand()
        self.fill_information()
        self.position = (self.position + 1) % self.tabWidget_Information.currentWidget().findChild(QtWidgets.QTableWidget).rowCount()

    def fill_information(self):
        information = self.tabWidget_Information.currentWidget().findChild(QtWidgets.QTableWidget)
        row = self.position

        for column in range(len(information_keys)):
            item = QtWidgets.QTableWidgetItem()
            item.setText(str(self.params[information_keys[column]]))
            information.setItem(row, column, item)

    def disconnect(self):
        print('Running cleanup/teardown')
        logging.debug('Running cleanup/teardown')
        if self.objective_stage is not None and self.offline is False:
            # Return objective lens stage to the "out" position and disconnect.
            self.move_absolute_objective_stage(self.objective_stage, position=0)
            self.objective_stage.disconnect()
        if self.microscope is not None:
            self.microscope.disconnect()

    ############## FIBSEM microscope methods ##############
    def connect_to_fibsem_microscope(self, ip_address="10.0.0.1"):
        """Connect to the FIBSEM microscope."""
        try:
            from GUI import fibsem
            self.microscope = fibsem.initialize(ip_address=ip_address)
            self.camera_settings = self.update_fibsem_settings()
        except Exception as e:
            display_error_message(traceback.format_exc())

    def update_fibsem_settings(self):
        # if not self.microscope:
        #     self.connect_to_fibsem_microscope()
        try:
            from GUI import fibsem
            dwell_time = float(self.lineEdit_dwell_time.text())*1.e-6
            resolution = self.comboBox_resolution.currentText()
            fibsem_settings = fibsem.update_camera_settings(dwell_time, resolution)
            self.camera_settings = fibsem_settings
            return fibsem_settings
        except Exception as e:
            display_error_message(traceback.format_exc())

    ############## FIBSEM image methods ##############
    def get_FIB_image(self, autosave=True):
        if self.offline:
            data = np.random.rand(self.label_FIB.maximumHeight(), self.label_FIB.maximumWidth())
            self.figure_FIB.clear()
            ax = self.figure_FIB.add_subplot(111)
            ax.imshow(data)
            self.canvas_FIB.draw()
            return

        try:
            if self.checkBox_Autocontrast.isChecked():
                self.image_fibsem = self.autocontrast_ion_beam()
            else:
                self.image_fibsem = fibsem.new_ion_image(
                    self.microscope, self.camera_settings)
            # TODO: Do we really need skimage img_as_ubyte? Display only?
            self.array_list_FIBSEM = skimage.util.img_as_ubyte(
                self.image_fibsem.data)
            # Update display
            self.update_display("FIBSEM")
        except Exception as e:
            display_error_message(traceback.format_exc())
        else:
            self.image_ion = copy.deepcopy(self.image_fibsem)
            return self.image_fibsem

    def get_last_FIB_image(self):
        try:
            self.image_fibsem = fibsem.last_ion_image(self.microscope)
            self.array_list_FIBSEM = skimage.util.img_as_ubyte(self.image_fibsem.data)
            self.update_display("FIBSEM")
        except Exception as e:
            display_error_message(traceback.format_exc())
        else:
            self.image_ion = copy.deepcopy(self.image_fibsem)
            return self.image_fibsem

    def get_SEM_image(self, autosave=True):
        if self.offline:
            data = np.random.rand(self.label_SEM.maximumHeight()*2, self.label_SEM.maximumWidth()*2)
            self.figure_SEM.clear()
            ax = self.figure_SEM.add_subplot(111)
            ax.imshow(data)
            self.canvas_SEM.draw()
            return
        try:
            self.image_fibsem = fibsem.new_electron_image(self.microscope, self.camera_settings)
            # TODO: should this be copied? Should it be skimage img_as_ubyte?
            self.array_list_FIBSEM = np.copy(self.image_fibsem.data)
            # TODO: Inconsistent median filtering for display - should be in update_display('FIBSEM'), if anything.
            # Also consider correlation and milling window displays
            self.array_list_FIBSEM = ndi.median_filter(self.array_list_FIBSEM, 2)
            # update display
            self.update_display("SEM")
        except Exception as e:
            display_error_message(traceback.format_exc())
        else:
            return self.image_fibsem

    def get_last_SEM_image(self):
        try:
            self.image_fibsem = fibsem.last_electron_image(self.microscope)
            self.array_list_FIBSEM = self.image_fibsem.data
            self.array_list_FIBSEM = skimage.util.img_as_ubyte(self.array_list_FIBSEM)
            self.update_display("SEM")
        except Exception as e:
            display_error_message(traceback.format_exc())
        else:
            return self.image_fibsem

    def autocontrast_ion_beam(self):
        try:
            self.microscope.imaging.set_active_view(2)  # the ion beam view
            fibsem.autocontrast(self.microscope)
            self.image_fibsem = fibsem.last_ion_image(self.microscope)
        except Exception as e:
            display_error_message(traceback.format_exc())
        else:
            self.image_ion = copy.deepcopy(self.image_fibsem)
            return self.image_fibsem

    def update_display(self, modality):
        """Update the GUI display with the current image"""
        try:
            self.image_fibsem = 255*np.random.rand(self.frame_SEM.maximumHeight(),self.label_SEM.maximumWidth())
            image_array = self.image_fibsem
            # image_array = self.image_fibsem.data
            # Ensure image for display is RGB
            # if image_array.ndim >= 4:
            #     msg = "Please select a 2D image for display.\n" + \
            #           "Image shape here is {}".format(image_array.shape)
            #     logging.warning(msg)
            #     display_error_message(msg)
            #     return
            # elif image_array.ndim == 3:
            #     shape = image_array.shape
            #     if shape[-1] > 3:
            #         image_array = np.moveaxis(image_array, 0, -1)
            #     # After any swap axis...
            #     if shape[-1] > 3:  # should be the channel axis, can't have more than RGB
            #         msg = "Please select a 2D image with no more than 3 color channels for display.\n" + \
            #               "Image shape here is {}".format(image_array.shape)
            #         logging.warning(msg)
            #         display_error_message(msg)
            #         return

            self.image_fibsem = qimage2ndarray.array2qimage(image_array.copy())

            self.pixmap_FIBSEM = QtGui.QPixmap.fromImage(self.image_fibsem)
            self.pixmap_FIBSEM = self.pixmap_FIBSEM.scaled(640, 400, QtCore.Qt.KeepAspectRatio)
            self.label_SEM.setPixmap(self.pixmap_FIBSEM)
            self.frame_SEM.layout().addWidget(_NavigationToolbar(self.frame_SEM.layout(), self.frame_SEM))


        except Exception as e:
            display_error_message(traceback.format_exc())



def display_error_message(message):
    """PyQt dialog box displaying an error message."""
    print('display_error_message')
    logging.exception(message)
    error_dialog = QtWidgets.QErrorMessage()
    error_dialog.showMessage(message)
    error_dialog.exec_()
    return error_dialog


def main(offline='True'):
    if offline.lower() == 'false':
        logging.basicConfig(level=logging.WARNING)
        launch_gui(ip_address='10.0.0.1', offline=False)
    elif offline.lower() == 'true':
        logging.basicConfig(level=logging.DEBUG)
        with mock.patch.dict('os.environ', {'PYLON_CAMEMU': '1'}):
                try:
                    launch_gui(ip_address="localhost", offline=True)
                except Exception:
                    import pdb
                    traceback.print_exc()
                    pdb.set_trace()


def launch_gui(ip_address='10.0.0.1', offline=False):
    """Launch the `autoliftout` main application window."""
    app = QtWidgets.QApplication([])
    qt_app = GUIMainWindow(ip_address=ip_address, offline=offline)
    app.aboutToQuit.connect(qt_app.disconnect)  # cleanup & teardown
    qt_app.show()
    sys.exit(app.exec_())


main(offline='True')
