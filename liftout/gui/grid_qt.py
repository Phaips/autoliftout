import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QLabel
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5 import QtWidgets

from liftout.fibsem.sampleposition import SamplePosition
from autoscript_sdb_microscope_client.structures import AdornedImage

import os

class App(QDialog):

    def __init__(self):
        super().__init__()
        self.title = "GRID LAYOUT TEST"
        self.left = 100
        self.top = 100
        self.width = 1200
        self.height = 800
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.createGridLayout()
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(self.horizontalGroupBox)

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(scroll_area)
        self.setLayout(windowLayout)

        self.show()

    def createGridLayout(self):
        self.horizontalGroupBox = QGroupBox("Experiment Data")

        experiment_path = r"C:\Users\Admin\Github\autoliftout\liftout\log\experiment_name_20220207.174055" # all
        # experiment_path = r"C:\Users\Admin\Github\autoliftout\liftout\log\test_exp_20220208.111741" # some iamges
        # experiment_path = r"C:\Users\Admin\Github\autoliftout\liftout\log\testtest_20220208.103251" # no sample or iamges...
        # use the sample class:


        sample = SamplePosition(experiment_path, None)
        yaml_file = sample.setup_yaml_file()
        if len(yaml_file["sample"]) == 0:
            print("NO SAMPLES IN THIS FILE")

            # return

        # reset previously loaded samples
        self.samples = []
        self.current_sample_position = None

        # load the samples
        for sample_no in yaml_file["sample"].keys():
            sample = SamplePosition(experiment_path, sample_no)
            sample.load_data_from_file()
            self.samples.append(sample)

        self.draw_sample_grid_ui()

    def draw_sample_grid_ui(self):
        self.horizontalGroupBox = QGroupBox("Experiment Data")

        ###################
        # TODO: port to liftout gui

        gridLayout = QGridLayout()

        # Only add data is sample positions are added
        if len(self.samples) == 0:
            label = QLabel()
            label.setText("No Sample Positions Selected.")
            gridLayout.addWidget(label)
            self.horizontalGroupBox.setLayout(gridLayout)
            return

        sample_images = [[] for _ in self.samples]

        # initial, mill, jcut, liftout, land, reset, thin, polish
        exemplar_filenames = ["ref_lamella_low_res_eb", "ref_trench_high_res_ib", "jcut_highres_ib",
                              "needle_liftout_landed_highres_ib", "landing_lamella_final_cut_highres_ib", "sharpen_needle_final_ib",
                              "thinning_lamella_stage_2_ib", "thinning_lamella_final_polishing_ib"]

        # headers
        headers = ["Sample No", "Position", "Reference", "Milling", "J-Cut", "Liftout", "Landing", "Reset", "Thinning", "Polishing"]
        for j, title in enumerate(headers):
            label_header = QLabel()
            label_header.setText(title)
            label_header.setStyleSheet("font-family: Arial; font-weight: bold; font-size: 18px;")
            label_header.setAlignment(Qt.AlignCenter)
            gridLayout.addWidget(label_header, 0, j)

        # TODO: can add more fields as neccessary
        for i, sp in enumerate(self.samples):

            # load the exemplar images for each sample
            qimage_labels = []
            for img_basename in exemplar_filenames:
                fname = os.path.join(sp.data_path, sp.sample_id, f"{img_basename}.tif")
                imageLabel = QLabel()


                if os.path.exists(fname):
                    adorned_img = AdornedImage.load(fname)
                    image = QImage(adorned_img.data, adorned_img.data.shape[1], adorned_img.data.shape[0], QImage.Format_Grayscale8)
                    imageLabel.setPixmap(QPixmap.fromImage(image).scaled(150, 150))
                qimage_labels.append(imageLabel)
            # TODO: sort out sizing / spacing when there are no images present
            sample_images[i] = qimage_labels

            # display information on grid
            row_id = i + 1

            # display sample no
            label_sample = QLabel()
            label_sample.setText(f"""Sample {row_id:02d} \n\nStage: {sp.microscope_state.last_completed_stage}""")
            gridLayout.addWidget(label_sample, row_id, 0)

            # display sample position
            # TOD0: replace this with a plot showing the position?
            def testColourMap(x, y):
                # ref: https://stackoverflow.com/questions/30646152/python-matplotlib-pyqt-plot-to-qpixmap
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_qt5agg import \
                    FigureCanvasQTAgg as _FigureCanvas

                fig = plt.Figure()
                canvas = _FigureCanvas(fig)
                ax = fig.add_subplot(111)
                # gradient = np.linspace(0, 1, 256)
                # gradient = np.vstack((gradient, gradient))
                # ax.imshow(gradient, aspect=10, cmap="magma")
                ax.scatter(x=x, y=y, c="blue")
                ax.set_axis_off()
                canvas.draw()
                size = canvas.size()
                width, height = size.width(), size.height()
                im = QImage(canvas.buffer_rgba(), width, height, QImage.Format_ARGB32RGB32).scaled(150, 150)
                return QPixmap(im)

            label_pos = QLabel()
            # label_pos.setPixmap(testColourMap(sp.lamella_coordinates.x, sp.landing_coordinates.y))
            label_pos.setText(f"""
            Pos: x:{sp.lamella_coordinates.x:.3f}, y:{sp.lamella_coordinates.y:.3f}, z:{sp.lamella_coordinates.z:.3f}\n
            Land: x:{sp.landing_coordinates.x:.3f}, y:{sp.landing_coordinates.y:.3f}, z:{sp.landing_coordinates.z:.3f}\n
            """)
            gridLayout.addWidget(label_pos, row_id, 1)

            # display exemplar images
            gridLayout.addWidget(sample_images[i][0], row_id, 2)
            gridLayout.addWidget(sample_images[i][1], row_id, 3)
            gridLayout.addWidget(sample_images[i][2], row_id, 4)
            gridLayout.addWidget(sample_images[i][3], row_id, 5)
            gridLayout.addWidget(sample_images[i][4], row_id, 6)
            gridLayout.addWidget(sample_images[i][5], row_id, 7)
            gridLayout.addWidget(sample_images[i][6], row_id, 8)
            gridLayout.addWidget(sample_images[i][7], row_id, 9)

        self.horizontalGroupBox.setLayout(gridLayout)
        ###################

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())