from liftout.gui.qtdesigner_files import main as gui_main
from liftout.fibsem.acquire import *
from liftout.fibsem.movement import *
from liftout.fibsem.utils import *
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import traceback
import logging
import sys
import mock
from tkinter import Tk, filedialog
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _NavigationToolbar
import matplotlib.pyplot as plt

_translate = QtCore.QCoreApplication.translate
logger = logging.getLogger(__name__)

protocol_template_path = '..\\protocol_liftout.yml'
starting_positions = 1
information_keys = ['x', 'y', 'z', 'rotation', 'tilt', 'comments']


class GUIMainWindow(gui_main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, ip_address='10.0.0.1', offline=False):
        super(GUIMainWindow, self).__init__()
        # from liftout import AutoLiftout
        from liftout.main2 import AutoLiftout
        # TODO: replace "SEM, FIB" with BeamType calls
        self.offline = offline
        self.setupUi(self)
        self.setWindowTitle('Autoliftout User Interface Main Window')
        self.statusbar.setSizeGripEnabled(0)
        self.status = QtWidgets.QLabel(self.statusbar)
        self.status.setAlignment(QtCore.Qt.AlignRight)
        self.statusbar.addPermanentWidget(self.status, 1)

        self.setup_connections()

        # initialise image frames and images
        self.image_SEM = None
        self.image_FIB = None
        self.figure_SEM = None
        self.figure_FIB = None
        self.canvas_SEM = None
        self.canvas_FIB = None
        self.ax_SEM = None
        self.ax_FIB = None
        self.toolbar_SEM = None
        self.toolbar_FIB = None

        self.initialise_image_frames()

        # image frame interaction
        self.xclick = None
        self.yclick = None

        # initialise template and information
        self.edit = None
        self.position = 0
        self.save_destination = None
        self.key_list_protocol = list()
        self.params = {}
        for parameter in range(len(information_keys)):
            self.params[information_keys[parameter]] = 0

        self.new_protocol()

        # initialise hardware
        self.ip_address = ip_address
        self.microscope = None
        self.detector = None
        self.lasers = None
        self.objective_stage = None
        self.camera_settings = None
        self.comboBox_resolution.setCurrentIndex(2)  # resolution '3072x2048'

        self.initialize_hardware(offline=offline)

        self.auto = AutoLiftout(microscope=self.microscope)

        self.image_SEM = last_image(self.microscope, beam_type=BeamType.ELECTRON)
        self.update_display("SEM")

        self.image_FIB = last_image(self.microscope, beam_type=BeamType.ION)
        self.update_display("FIB")

        self.ask_user(message='Do you want to sputter the whole sample grid with platinum?')

    def on_gui_click(self, event, modality):
        image = None
        if beam_type is BeamType.ELECTRON:
            image = self.image_SEM
        if beam_type is BeamType.ION:
            image = self.image_FIB

        if event.button == 1:
            if event.dblclick:
                if image:
                    self.xclick = event.xdata
                    self.yclick = event.ydata
                    x, y = pixel_to_realspace_coordinate([self.xclick, self.yclick], image)
                    # print(f'Moving {modality} in x by {round(x*1e6, 2)}um')
                    # print(f'Moving {modality} in y by {round(y*1e6, 2)}um\n')

                    move_relative(self.microscope, x, y)
                    last_image(microscope=self.microscope, beam_type=beam_type)

    def initialise_image_frames(self):
        self.figure_SEM = plt.figure()
        self.canvas_SEM = _FigureCanvas(self.figure_SEM)
        self.toolbar_SEM = _NavigationToolbar(self.canvas_SEM, self)
        self.label_SEM.setLayout(QtWidgets.QVBoxLayout())
        self.label_SEM.layout().addWidget(self.toolbar_SEM)
        self.label_SEM.layout().addWidget(self.canvas_SEM)

        self.canvas_SEM.mpl_connect('button_press_event', lambda event: self.on_gui_click(event, modality='SEM'))

        self.figure_FIB = plt.figure()
        self.canvas_FIB = _FigureCanvas(self.figure_FIB)
        self.toolbar_FIB = _NavigationToolbar(self.canvas_FIB, self)
        self.label_FIB.setLayout(QtWidgets.QVBoxLayout())
        self.label_FIB.layout().addWidget(self.toolbar_FIB)
        self.label_FIB.layout().addWidget(self.canvas_FIB)

        self.canvas_FIB.mpl_connect('button_press_event', lambda event: self.on_gui_click(event, modality='FIB'))

    def initialize_hardware(self, offline=False):
        if offline is False:
            self.connect_to_microscope(ip_address=self.ip_address)
        elif offline is True:
            pass
            # self.connect_to_microscope(ip_address='localhost')

    def setup_connections(self):
        # Protocol and information table connections
        self.pushButton_Protocol_Load.clicked.connect(lambda: self.load_yaml())
        self.pushButton_Protocol_New.clicked.connect(lambda: self.new_protocol())
        self.pushButton_Protocol_Delete.clicked.connect(lambda: self.delete_protocol())
        self.pushButton_Protocol_Save.clicked.connect(lambda: self.save_protocol(self.save_destination))
        self.pushButton_Protocol_Save_As.clicked.connect(lambda: self.save_protocol())
        self.pushButton_Protocol_Rename.clicked.connect(lambda: self.rename_protocol())
        self.tabWidget_Protocol.tabBarDoubleClicked.connect(lambda: self.rename_protocol())
        self.tabWidget_Protocol.tabBar().tabMoved.connect(lambda: self.tab_moved('protocol'))
        self.tabWidget_Information.tabBar().tabMoved.connect(lambda: self.tab_moved('information'))

        # Remove later
        self.pushButton_random_data.clicked.connect(lambda: self.random_data())

        # FIBSEM methods
        self.button_get_image_FIB.clicked.connect(lambda: self.get_image(modality='FIB'))
        self.button_get_image_SEM.clicked.connect(lambda: self.get_image(modality='SEM'))
        self.connect_microscope.clicked.connect(lambda: self.connect_to_microscope(ip_address=self.ip_address))

    def ask_user(self, image=None, message=None):
        self.popup = QtWidgets.QWidget()

        if message is None:
            message = "ok?"

        question = QtWidgets.QLabel(self.popup)
        print('1')
        font = QtGui.QFont()
        font.setPointSize(24)
        question.setText(message)
        question.setFont(font)

        question.setAlignment(QtCore.Qt.AlignCenter)
        question_layout = QtWidgets.QHBoxLayout()
        question.setLayout(question_layout)
        print('2')
        print('3')
        print('4')

        button_box = QtWidgets.QWidget(self.popup)
        button_layout = QtWidgets.QHBoxLayout()
        yes = QtWidgets.QPushButton('Yes')
        no = QtWidgets.QPushButton('No')

        yes.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        no.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))

        button_box.setLayout(button_layout)
        button_box.layout().addWidget(yes)
        button_box.layout().addWidget(no)

        self.popup.setLayout(QtWidgets.QVBoxLayout())

        if image:
            fig = plt.figure()
            canvas = _FigureCanvas(fig)
            image_array = image.data
            fig.clear()
            ax = fig.add_subplot(111)
            ax.imshow(image_array, cmap='gray')
            canvas.draw()
            self.popup.layout().addWidget(canvas, 4)

        self.popup.layout().addWidget(question, 1)
        self.popup.layout().addWidget(button_box, 1)
        self.popup.show()

        print('5')

    def new_protocol(self):
        num_index = self.tabWidget_Protocol.__len__() + 1

        # new tab for protocol
        new_protocol_tab = QtWidgets.QWidget()
        new_protocol_tab.setObjectName(f'Protocol {num_index}')
        layout_protocol_tab = QtWidgets.QGridLayout(new_protocol_tab)
        layout_protocol_tab.setObjectName(f'gridLayout_{num_index}')

        # new text edit to hold protocol
        protocol_text_edit = QtWidgets.QTextEdit()
        font = QtGui.QFont()
        font.setPointSize(10)
        protocol_text_edit.setFont(font)
        protocol_text_edit.setObjectName(f'protocol_text_edit_{num_index}')
        layout_protocol_tab.addWidget(protocol_text_edit, 0, 0, 1, 1)

        self.tabWidget_Protocol.addTab(new_protocol_tab, f'Protocol {num_index}')
        self.tabWidget_Protocol.setCurrentWidget(new_protocol_tab)
        self.load_template_protocol()

        #

        # new tab for information from FIBSEM
        new_information_tab = QtWidgets.QWidget()
        new_information_tab.setObjectName(f'Protocol {num_index}')
        layout_new_information_tab = QtWidgets.QGridLayout(new_information_tab)
        layout_new_information_tab.setObjectName(f'layout_new_information_tab_{num_index}')

        # new empty table for information to fill
        new_table_widget = QtWidgets.QTableWidget()
        new_table_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        new_table_widget.setAlternatingRowColors(True)
        new_table_widget.setObjectName(f'tableWidget_{num_index}')
        new_table_widget.setColumnCount(len(information_keys))
        new_table_widget.setRowCount(starting_positions)

        # set up rows
        for row in range(starting_positions):
            table_item = QtWidgets.QTableWidgetItem()
            new_table_widget.setVerticalHeaderItem(row, table_item)
            item = new_table_widget.verticalHeaderItem(row)
            item.setText(_translate('MainWindow', f'Position {row+1}'))

        # set up columns
        for column in range(len(information_keys)):
            table_item = QtWidgets.QTableWidgetItem()
            new_table_widget.setHorizontalHeaderItem(column, table_item)
            item = new_table_widget.horizontalHeaderItem(column)
            item.setText(_translate('MainWindow', information_keys[column]))

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
        with open(protocol_template_path, 'r') as file:
            _dict = yaml.safe_load(file)

        for key, value in _dict.items():
            self.key_list_protocol.append(key)
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if key2 not in self.key_list_protocol:
                        self.key_list_protocol.append(key)
                if isinstance(value, dict):
                    for key3, value3 in value.items():
                        if key3 not in self.key_list_protocol:
                            self.key_list_protocol.append(key)
                elif isinstance(value, list):
                    for dictionary in value:
                        for key4, value4 in dictionary.items():
                            if key4 not in self.key_list_protocol:
                                self.key_list_protocol.append(key)

        self.load_protocol_text(_dict)

    def load_protocol_text(self, dictionary):
        protocol_text = str()
        count = 0
        jcut = 0
        _dict = dictionary

        for key in _dict:
            if key not in self.key_list_protocol:
                print(f'Unexpected parameter in template file')
                return

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
        checked = 0
        print(f'Please select protocol file (yml)')
        root = Tk()
        root.withdraw()
        _dict = None
        while not checked:
            checked = 1
            try:
                load_directory = filedialog.askopenfile(mode='r', filetypes=[('yml files', '*.yml')])
                if load_directory is None:
                    return

                while not load_directory.name.endswith('.yml'):
                    print('Not a yml configuration file')
                    load_directory = filedialog.askopenfile(mode='r', filetypes=[('yml files', '*.yml')])

                with open(load_directory.name, 'r') as file:
                    _dict = yaml.safe_load(file)
                    file.close()
            except Exception:
                display_error_message(traceback.format_exc())
            for key in _dict:
                if key not in self.key_list_protocol:
                    if checked:
                        print(f'Unexpected parameter in protocol file')
                    checked = 0

        root.destroy()
        print(_dict)

        self.load_protocol_text(_dict)

    def tab_moved(self, moved):
        if moved == 'protocol':
            self.tabWidget_Information.tabBar().moveTab(self.tabWidget_Information.currentIndex(), self.tabWidget_Protocol.currentIndex())
        elif moved == 'information':
            self.tabWidget_Protocol.tabBar().moveTab(self.tabWidget_Protocol.currentIndex(), self.tabWidget_Information.currentIndex())

    def random_data(self):
        for parameter in range(len(self.params)):
            self.params[information_keys[parameter]] = np.random.randint(0, 1)
        self.fill_information()
        self.position = (self.position + 1) % self.tabWidget_Information.currentWidget().findChild(QtWidgets.QTableWidget).rowCount()

    def fill_information(self):
        information = self.tabWidget_Information.currentWidget().findChild(QtWidgets.QTableWidget)
        row = self.position

        for column in range(len(information_keys)):
            item = QtWidgets.QTableWidgetItem()
            item.setText(str(self.params[information_keys[column]]))
            information.setItem(row, column, item)

    def connect_to_microscope(self, ip_address='10.0.0.1'):
        """Connect to the FIBSEM microscope."""
        try:
            self.microscope = initialise_fibsem(ip_address=ip_address)
        except Exception:
            display_error_message(traceback.format_exc())


    def update_display(self, modality):
        """Update the GUI display with the current image"""
        try:
            if modality == 'SEM':
                image_array = self.image_SEM.data
                self.figure_SEM.clear()
                self.ax_SEM = self.figure_SEM.add_subplot(111)
                self.ax_SEM.imshow(image_array, cmap='gray')
                self.canvas_SEM.draw()
            elif modality == 'FIB':
                image_array = self.image_FIB.data
                self.figure_FIB.clear()
                self.ax_FIB = self.figure_FIB.add_subplot(111)
                self.ax_FIB.imshow(image_array, cmap='gray')
                self.canvas_FIB.draw()

        except Exception:
            display_error_message(traceback.format_exc())

    def disconnect(self):
        print('Running cleanup/teardown')
        logging.debug('Running cleanup/teardown')
        if self.objective_stage and self.offline is False:
            # Return objective lens stage to the 'out' position and disconnect.
            self.move_absolute_objective_stage(self.objective_stage, position=0)
            self.objective_stage.disconnect()
        if self.microscope:
            self.microscope.disconnect()


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
                launch_gui(ip_address='localhost', offline=True)
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


main(offline='False')
# main(offline='True')
