# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'milling_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1012, 676)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_main = QtWidgets.QFrame(Dialog)
        self.frame_main.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_main.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_main.setObjectName("frame_main")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_main)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame_image = QtWidgets.QFrame(self.frame_main)
        self.frame_image.setMaximumSize(QtCore.QSize(1536, 1024))
        self.frame_image.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_image.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_image.setObjectName("frame_image")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_image)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_image = QtWidgets.QLabel(self.frame_image)
        self.label_image.setText("")
        self.label_image.setObjectName("label_image")
        self.gridLayout_3.addWidget(self.label_image, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame_image, 1, 1, 2, 1)
        self.frame_parameters = QtWidgets.QFrame(self.frame_main)
        self.frame_parameters.setMaximumSize(QtCore.QSize(200, 16777215))
        self.frame_parameters.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_parameters.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_parameters.setObjectName("frame_parameters")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_parameters)
        self.gridLayout_4.setVerticalSpacing(6)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.doubleSpinBox_11 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_11.setObjectName("doubleSpinBox_11")
        self.gridLayout_4.addWidget(self.doubleSpinBox_11, 20, 1, 1, 1)
        self.label_09 = QtWidgets.QLabel(self.frame_parameters)
        self.label_09.setObjectName("label_09")
        self.gridLayout_4.addWidget(self.label_09, 17, 0, 1, 1)
        self.doubleSpinBox_01 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_01.setObjectName("doubleSpinBox_01")
        self.gridLayout_4.addWidget(self.doubleSpinBox_01, 4, 1, 1, 1)
        self.label_08 = QtWidgets.QLabel(self.frame_parameters)
        self.label_08.setObjectName("label_08")
        self.gridLayout_4.addWidget(self.label_08, 16, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.frame_parameters)
        self.label_11.setObjectName("label_11")
        self.gridLayout_4.addWidget(self.label_11, 20, 0, 1, 1)
        self.doubleSpinBox_09 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_09.setObjectName("doubleSpinBox_09")
        self.gridLayout_4.addWidget(self.doubleSpinBox_09, 17, 1, 1, 1)
        self.doubleSpinBox_03 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_03.setObjectName("doubleSpinBox_03")
        self.gridLayout_4.addWidget(self.doubleSpinBox_03, 6, 1, 1, 1)
        self.doubleSpinBox_05 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_05.setObjectName("doubleSpinBox_05")
        self.gridLayout_4.addWidget(self.doubleSpinBox_05, 10, 1, 1, 1)
        self.doubleSpinBox_02 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_02.setObjectName("doubleSpinBox_02")
        self.gridLayout_4.addWidget(self.doubleSpinBox_02, 5, 1, 1, 1)
        self.doubleSpinBox_04 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_04.setObjectName("doubleSpinBox_04")
        self.gridLayout_4.addWidget(self.doubleSpinBox_04, 8, 1, 1, 1)
        self.label_02 = QtWidgets.QLabel(self.frame_parameters)
        self.label_02.setObjectName("label_02")
        self.gridLayout_4.addWidget(self.label_02, 5, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem, 28, 0, 1, 2)
        self.doubleSpinBox_06 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_06.setObjectName("doubleSpinBox_06")
        self.gridLayout_4.addWidget(self.doubleSpinBox_06, 12, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.frame_parameters)
        self.label_12.setObjectName("label_12")
        self.gridLayout_4.addWidget(self.label_12, 22, 0, 1, 1)
        self.comboBox_pattern_stage = QtWidgets.QComboBox(self.frame_parameters)
        self.comboBox_pattern_stage.setObjectName("comboBox_pattern_stage")
        self.gridLayout_4.addWidget(self.comboBox_pattern_stage, 2, 0, 1, 2)
        self.doubleSpinBox_07 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_07.setObjectName("doubleSpinBox_07")
        self.gridLayout_4.addWidget(self.doubleSpinBox_07, 14, 1, 1, 1)
        self.label_01 = QtWidgets.QLabel(self.frame_parameters)
        self.label_01.setObjectName("label_01")
        self.gridLayout_4.addWidget(self.label_01, 4, 0, 1, 1)
        self.label_header_2 = QtWidgets.QLabel(self.frame_parameters)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_header_2.setFont(font)
        self.label_header_2.setObjectName("label_header_2")
        self.gridLayout_4.addWidget(self.label_header_2, 1, 0, 1, 1)
        self.label_05 = QtWidgets.QLabel(self.frame_parameters)
        self.label_05.setObjectName("label_05")
        self.gridLayout_4.addWidget(self.label_05, 10, 0, 1, 1)
        self.label_07 = QtWidgets.QLabel(self.frame_parameters)
        self.label_07.setObjectName("label_07")
        self.gridLayout_4.addWidget(self.label_07, 14, 0, 1, 1)
        self.doubleSpinBox_10 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_10.setObjectName("doubleSpinBox_10")
        self.gridLayout_4.addWidget(self.doubleSpinBox_10, 19, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.frame_parameters)
        self.label_10.setObjectName("label_10")
        self.gridLayout_4.addWidget(self.label_10, 19, 0, 1, 1)
        self.pushButton_runMilling = QtWidgets.QPushButton(self.frame_parameters)
        self.pushButton_runMilling.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_runMilling.setObjectName("pushButton_runMilling")
        self.gridLayout_4.addWidget(self.pushButton_runMilling, 29, 0, 1, 2)
        self.doubleSpinBox_08 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_08.setObjectName("doubleSpinBox_08")
        self.gridLayout_4.addWidget(self.doubleSpinBox_08, 16, 1, 1, 1)
        self.label_header = QtWidgets.QLabel(self.frame_parameters)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_header.setFont(font)
        self.label_header.setObjectName("label_header")
        self.gridLayout_4.addWidget(self.label_header, 3, 0, 1, 2)
        self.label_04 = QtWidgets.QLabel(self.frame_parameters)
        self.label_04.setObjectName("label_04")
        self.gridLayout_4.addWidget(self.label_04, 8, 0, 1, 1)
        self.label_06 = QtWidgets.QLabel(self.frame_parameters)
        self.label_06.setObjectName("label_06")
        self.gridLayout_4.addWidget(self.label_06, 12, 0, 1, 1)
        self.label_03 = QtWidgets.QLabel(self.frame_parameters)
        self.label_03.setObjectName("label_03")
        self.gridLayout_4.addWidget(self.label_03, 6, 0, 1, 1)
        self.doubleSpinBox_12 = QtWidgets.QDoubleSpinBox(self.frame_parameters)
        self.doubleSpinBox_12.setObjectName("doubleSpinBox_12")
        self.gridLayout_4.addWidget(self.doubleSpinBox_12, 22, 1, 1, 1)
        self.gridLayout_2.addWidget(self.frame_parameters, 1, 0, 2, 1)
        self.milling_title_label = QtWidgets.QLabel(self.frame_main)
        self.milling_title_label.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(26)
        font.setBold(True)
        font.setWeight(75)
        self.milling_title_label.setFont(font)
        self.milling_title_label.setTextFormat(QtCore.Qt.PlainText)
        self.milling_title_label.setObjectName("milling_title_label")
        self.gridLayout_2.addWidget(self.milling_title_label, 0, 0, 1, 2)
        self.gridLayout.addWidget(self.frame_main, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_09.setText(_translate("Dialog", "Label 9"))
        self.label_08.setText(_translate("Dialog", "Label 8"))
        self.label_11.setText(_translate("Dialog", "Label 11"))
        self.label_02.setText(_translate("Dialog", "Label 2"))
        self.label_12.setText(_translate("Dialog", "Label 12"))
        self.label_01.setText(_translate("Dialog", "Label 1"))
        self.label_header_2.setText(_translate("Dialog", "Milling Stage"))
        self.label_05.setText(_translate("Dialog", "Label 5"))
        self.label_07.setText(_translate("Dialog", "Label 7"))
        self.label_10.setText(_translate("Dialog", "Label 10"))
        self.pushButton_runMilling.setText(_translate("Dialog", "Run Milling"))
        self.label_header.setText(_translate("Dialog", "Pattern Parameters"))
        self.label_04.setText(_translate("Dialog", "Label 4"))
        self.label_06.setText(_translate("Dialog", "Label 6"))
        self.label_03.setText(_translate("Dialog", "Label 3"))
        self.milling_title_label.setText(_translate("Dialog", "AutoLiftout Milling"))

