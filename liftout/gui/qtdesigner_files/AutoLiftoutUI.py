# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AutoLiftoutUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(454, 498)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMaximumSize(QtCore.QSize(16777215, 600))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.frame)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_general = QtWidgets.QWidget()
        self.tab_general.setObjectName("tab_general")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_general)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_protocol_name = QtWidgets.QLabel(self.tab_general)
        self.label_protocol_name.setObjectName("label_protocol_name")
        self.gridLayout_3.addWidget(self.label_protocol_name, 1, 0, 1, 1)
        self.pushButton_test_button = QtWidgets.QPushButton(self.tab_general)
        self.pushButton_test_button.setObjectName("pushButton_test_button")
        self.gridLayout_3.addWidget(self.pushButton_test_button, 5, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem, 7, 0, 1, 1)
        self.pushButton_run_autoliftout = QtWidgets.QPushButton(self.tab_general)
        self.pushButton_run_autoliftout.setObjectName("pushButton_run_autoliftout")
        self.gridLayout_3.addWidget(self.pushButton_run_autoliftout, 3, 0, 1, 1)
        self.pushButton_setup_autoliftout = QtWidgets.QPushButton(self.tab_general)
        self.pushButton_setup_autoliftout.setObjectName("pushButton_setup_autoliftout")
        self.gridLayout_3.addWidget(self.pushButton_setup_autoliftout, 2, 0, 1, 1)
        self.label_general_info = QtWidgets.QLabel(self.tab_general)
        self.label_general_info.setMinimumSize(QtCore.QSize(0, 50))
        self.label_general_info.setObjectName("label_general_info")
        self.gridLayout_3.addWidget(self.label_general_info, 6, 0, 1, 1)
        self.pushButton_run_polishing = QtWidgets.QPushButton(self.tab_general)
        self.pushButton_run_polishing.setObjectName("pushButton_run_polishing")
        self.gridLayout_3.addWidget(self.pushButton_run_polishing, 4, 0, 1, 1)
        self.label_experiment_name = QtWidgets.QLabel(self.tab_general)
        self.label_experiment_name.setObjectName("label_experiment_name")
        self.gridLayout_3.addWidget(self.label_experiment_name, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_general, "")
        self.tab_lamella = QtWidgets.QWidget()
        self.tab_lamella.setObjectName("tab_lamella")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_lamella)
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem1, 5, 0, 1, 2)
        self.comboBox_lamella_select = QtWidgets.QComboBox(self.tab_lamella)
        self.comboBox_lamella_select.setObjectName("comboBox_lamella_select")
        self.gridLayout_4.addWidget(self.comboBox_lamella_select, 2, 0, 1, 2)
        self.label_lamella_header = QtWidgets.QLabel(self.tab_lamella)
        self.label_lamella_header.setObjectName("label_lamella_header")
        self.gridLayout_4.addWidget(self.label_lamella_header, 0, 0, 1, 2)
        self.checkBox_lamella_landing_selected = QtWidgets.QCheckBox(self.tab_lamella)
        self.checkBox_lamella_landing_selected.setObjectName("checkBox_lamella_landing_selected")
        self.gridLayout_4.addWidget(self.checkBox_lamella_landing_selected, 4, 0, 1, 1)
        self.checkBox_lamella_mark_failure = QtWidgets.QCheckBox(self.tab_lamella)
        self.checkBox_lamella_mark_failure.setObjectName("checkBox_lamella_mark_failure")
        self.gridLayout_4.addWidget(self.checkBox_lamella_mark_failure, 4, 1, 1, 1)
        self.label_lamella_status = QtWidgets.QLabel(self.tab_lamella)
        self.label_lamella_status.setMinimumSize(QtCore.QSize(0, 0))
        self.label_lamella_status.setObjectName("label_lamella_status")
        self.gridLayout_4.addWidget(self.label_lamella_status, 3, 0, 1, 2)
        self.tabWidget.addTab(self.tab_lamella, "")
        self.gridLayout_2.addWidget(self.tabWidget, 10, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout_2.addWidget(self.label_title, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 454, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_Experiment = QtWidgets.QAction(MainWindow)
        self.actionLoad_Experiment.setObjectName("actionLoad_Experiment")
        self.actionLoad_Protocol = QtWidgets.QAction(MainWindow)
        self.actionLoad_Protocol.setObjectName("actionLoad_Protocol")
        self.actionSputter_Platinum = QtWidgets.QAction(MainWindow)
        self.actionSputter_Platinum.setObjectName("actionSputter_Platinum")
        self.actionSharpen_Needle = QtWidgets.QAction(MainWindow)
        self.actionSharpen_Needle.setObjectName("actionSharpen_Needle")
        self.actionCalibrate_Needle = QtWidgets.QAction(MainWindow)
        self.actionCalibrate_Needle.setObjectName("actionCalibrate_Needle")
        self.actionEdit_Protocol = QtWidgets.QAction(MainWindow)
        self.actionEdit_Protocol.setObjectName("actionEdit_Protocol")
        self.actionEdit_Settings = QtWidgets.QAction(MainWindow)
        self.actionEdit_Settings.setObjectName("actionEdit_Settings")
        self.actionConnect_to_Microscope = QtWidgets.QAction(MainWindow)
        self.actionConnect_to_Microscope.setObjectName("actionConnect_to_Microscope")
        self.actionValidate_Microscope = QtWidgets.QAction(MainWindow)
        self.actionValidate_Microscope.setObjectName("actionValidate_Microscope")
        self.menuFile.addAction(self.actionConnect_to_Microscope)
        self.menuFile.addAction(self.actionLoad_Experiment)
        self.menuFile.addAction(self.actionLoad_Protocol)
        self.menuTools.addAction(self.actionValidate_Microscope)
        self.menuTools.addAction(self.actionSputter_Platinum)
        self.menuTools.addAction(self.actionSharpen_Needle)
        self.menuTools.addAction(self.actionCalibrate_Needle)
        self.menuEdit.addAction(self.actionEdit_Protocol)
        self.menuEdit.addAction(self.actionEdit_Settings)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_protocol_name.setText(_translate("MainWindow", "Protocol"))
        self.pushButton_test_button.setText(_translate("MainWindow", "Test Button"))
        self.pushButton_run_autoliftout.setText(_translate("MainWindow", "Run AutoLiftout"))
        self.pushButton_setup_autoliftout.setText(_translate("MainWindow", "Setup AutoLiftout"))
        self.label_general_info.setText(_translate("MainWindow", "Info"))
        self.pushButton_run_polishing.setText(_translate("MainWindow", "Run Polishing"))
        self.label_experiment_name.setText(_translate("MainWindow", "Experiment"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_general), _translate("MainWindow", "General"))
        self.label_lamella_header.setText(_translate("MainWindow", "Lamella Data"))
        self.checkBox_lamella_landing_selected.setText(_translate("MainWindow", "Landing Selected"))
        self.checkBox_lamella_mark_failure.setText(_translate("MainWindow", "Mark as Failure"))
        self.label_lamella_status.setText(_translate("MainWindow", "Status"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_lamella), _translate("MainWindow", "Lamella"))
        self.label_title.setText(_translate("MainWindow", "AutoLiftout"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuTools.setTitle(_translate("MainWindow", "Tools"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.actionLoad_Experiment.setText(_translate("MainWindow", "Load Experiment"))
        self.actionLoad_Protocol.setText(_translate("MainWindow", "Load Protocol"))
        self.actionSputter_Platinum.setText(_translate("MainWindow", "Sputter Platinum"))
        self.actionSharpen_Needle.setText(_translate("MainWindow", "Sharpen Needle"))
        self.actionCalibrate_Needle.setText(_translate("MainWindow", "Calibrate Needle"))
        self.actionEdit_Protocol.setText(_translate("MainWindow", "Edit Protocol"))
        self.actionEdit_Settings.setText(_translate("MainWindow", "Edit Settings"))
        self.actionConnect_to_Microscope.setText(_translate("MainWindow", "Connect to Microscope"))
        self.actionValidate_Microscope.setText(_translate("MainWindow", "Validate Microscope"))
