# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1612, 876)
        MainWindow.setMinimumSize(QtCore.QSize(0, 825))
        MainWindow.setMaximumSize(QtCore.QSize(100000, 100000))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(0, 784))
        self.centralwidget.setMaximumSize(QtCore.QSize(100000, 100000))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.Frame_Main = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Frame_Main.sizePolicy().hasHeightForWidth())
        self.Frame_Main.setSizePolicy(sizePolicy)
        self.Frame_Main.setMinimumSize(QtCore.QSize(0, 0))
        self.Frame_Main.setMaximumSize(QtCore.QSize(100000, 100000))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Frame_Main.setFont(font)
        self.Frame_Main.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_Main.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_Main.setObjectName("Frame_Main")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.Frame_Main)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setSpacing(0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.frame_Images = QtWidgets.QFrame(self.Frame_Main)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_Images.setFont(font)
        self.frame_Images.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_Images.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_Images.setObjectName("frame_Images")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_Images)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.frame_Images)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame.setFont(font)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_8.setContentsMargins(0, -1, -1, 0)
        self.gridLayout_8.setSpacing(0)
        self.gridLayout_8.setObjectName("gridLayout_8")
        spacerItem = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_8.addItem(spacerItem, 0, 1, 1, 1)
        self.frame_grid = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_grid.sizePolicy().hasHeightForWidth())
        self.frame_grid.setSizePolicy(sizePolicy)
        self.frame_grid.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_grid.setMaximumSize(QtCore.QSize(10000, 1001))
        self.frame_grid.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_grid.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_grid.setObjectName("frame_grid")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.frame_grid)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.scroll_area = QtWidgets.QScrollArea(self.frame_grid)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("scroll_area")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1393, 740))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scroll_area.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_7.addWidget(self.scroll_area, 0, 0, 1, 1)
        self.gridLayout_8.addWidget(self.frame_grid, 0, 2, 1, 1)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMinimumSize(QtCore.QSize(150, 704))
        self.frame_3.setMaximumSize(QtCore.QSize(150, 100000))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout.setContentsMargins(0, -1, 0, -1)
        self.verticalLayout.setSpacing(15)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_load_sample_data = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_load_sample_data.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_load_sample_data.setObjectName("pushButton_load_sample_data")
        self.verticalLayout.addWidget(self.pushButton_load_sample_data)
        self.pushButton_initialise = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_initialise.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_initialise.setObjectName("pushButton_initialise")
        self.verticalLayout.addWidget(self.pushButton_initialise)
        self.pushButton_autoliftout = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_autoliftout.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_autoliftout.setObjectName("pushButton_autoliftout")
        self.verticalLayout.addWidget(self.pushButton_autoliftout)
        self.pushButton_thinning = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_thinning.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_thinning.setObjectName("pushButton_thinning")
        self.verticalLayout.addWidget(self.pushButton_thinning)
        self.pushButton_test_popup = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_test_popup.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_test_popup.setObjectName("pushButton_test_popup")
        self.verticalLayout.addWidget(self.pushButton_test_popup)
        self.pushButton_add_sample_position = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_add_sample_position.setEnabled(False)
        self.pushButton_add_sample_position.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_add_sample_position.setObjectName("pushButton_add_sample_position")
        self.verticalLayout.addWidget(self.pushButton_add_sample_position)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.label_stage = QtWidgets.QLabel(self.frame_3)
        self.label_stage.setMinimumSize(QtCore.QSize(0, 80))
        self.label_stage.setMaximumSize(QtCore.QSize(150, 16777215))
        self.label_stage.setObjectName("label_stage")
        self.verticalLayout.addWidget(self.label_stage)
        self.gridLayout_8.addWidget(self.frame_3, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.frame)
        self.frame_SEM = QtWidgets.QFrame(self.frame_Images)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_SEM.sizePolicy().hasHeightForWidth())
        self.frame_SEM.setSizePolicy(sizePolicy)
        self.frame_SEM.setMinimumSize(QtCore.QSize(1, 1))
        self.frame_SEM.setMaximumSize(QtCore.QSize(1, 1))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_SEM.setFont(font)
        self.frame_SEM.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_SEM.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_SEM.setLineWidth(0)
        self.frame_SEM.setObjectName("frame_SEM")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_SEM)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_SEM = QtWidgets.QLabel(self.frame_SEM)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_SEM.sizePolicy().hasHeightForWidth())
        self.label_SEM.setSizePolicy(sizePolicy)
        self.label_SEM.setMinimumSize(QtCore.QSize(1, 1))
        self.label_SEM.setMaximumSize(QtCore.QSize(1, 1))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_SEM.setFont(font)
        self.label_SEM.setText("")
        self.label_SEM.setObjectName("label_SEM")
        self.gridLayout.addWidget(self.label_SEM, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.frame_SEM)
        self.frame_FIB = QtWidgets.QFrame(self.frame_Images)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_FIB.sizePolicy().hasHeightForWidth())
        self.frame_FIB.setSizePolicy(sizePolicy)
        self.frame_FIB.setMinimumSize(QtCore.QSize(1, 1))
        self.frame_FIB.setMaximumSize(QtCore.QSize(1, 1))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_FIB.setFont(font)
        self.frame_FIB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_FIB.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_FIB.setObjectName("frame_FIB")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_FIB)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_FIB = QtWidgets.QLabel(self.frame_FIB)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_FIB.sizePolicy().hasHeightForWidth())
        self.label_FIB.setSizePolicy(sizePolicy)
        self.label_FIB.setMinimumSize(QtCore.QSize(1, 1))
        self.label_FIB.setMaximumSize(QtCore.QSize(1, 1))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_FIB.setFont(font)
        self.label_FIB.setText("")
        self.label_FIB.setObjectName("label_FIB")
        self.gridLayout_2.addWidget(self.label_FIB, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.frame_FIB)
        self.gridLayout_4.addWidget(self.frame_Images, 1, 0, 1, 1)
        self.frame_header = QtWidgets.QFrame(self.Frame_Main)
        self.frame_header.setMinimumSize(QtCore.QSize(0, 40))
        self.frame_header.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_header.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_header.setObjectName("frame_header")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_header)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_title = QtWidgets.QLabel(self.frame_header)
        self.label_title.setTextFormat(QtCore.Qt.AutoText)
        self.label_title.setObjectName("label_title")
        self.horizontalLayout_2.addWidget(self.label_title)
        self.gridLayout_4.addWidget(self.frame_header, 0, 0, 1, 1)
        self.gridLayout_6.addWidget(self.Frame_Main, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1612, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuChange_Mode = QtWidgets.QMenu(self.menuFile)
        self.menuChange_Mode.setObjectName("menuChange_Mode")
        self.menuUtilities = QtWidgets.QMenu(self.menubar)
        self.menuUtilities.setObjectName("menuUtilities")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_Experiment = QtWidgets.QAction(MainWindow)
        self.actionLoad_Experiment.setObjectName("actionLoad_Experiment")
        self.actionLoad_Configuration = QtWidgets.QAction(MainWindow)
        self.actionLoad_Configuration.setObjectName("actionLoad_Configuration")
        self.actionMark_Sample_Position_Failed = QtWidgets.QAction(MainWindow)
        self.actionMark_Sample_Position_Failed.setObjectName("actionMark_Sample_Position_Failed")
        self.actionAutoLiftout = QtWidgets.QAction(MainWindow)
        self.actionAutoLiftout.setObjectName("actionAutoLiftout")
        self.actionAutoLamella = QtWidgets.QAction(MainWindow)
        self.actionAutoLamella.setObjectName("actionAutoLamella")
        self.actionSputter_Platinum = QtWidgets.QAction(MainWindow)
        self.actionSputter_Platinum.setObjectName("actionSputter_Platinum")
        self.actionSharpen_Needle = QtWidgets.QAction(MainWindow)
        self.actionSharpen_Needle.setObjectName("actionSharpen_Needle")
        self.menuChange_Mode.addAction(self.actionAutoLiftout)
        self.menuChange_Mode.addAction(self.actionAutoLamella)
        self.menuFile.addAction(self.actionLoad_Experiment)
        self.menuFile.addAction(self.actionLoad_Configuration)
        self.menuFile.addAction(self.actionMark_Sample_Position_Failed)
        self.menuFile.addAction(self.menuChange_Mode.menuAction())
        self.menuUtilities.addAction(self.actionSputter_Platinum)
        self.menuUtilities.addAction(self.actionSharpen_Needle)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuUtilities.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_load_sample_data.setText(_translate("MainWindow", "Load Sample Data"))
        self.pushButton_initialise.setText(_translate("MainWindow", "Setup AutoLiftout"))
        self.pushButton_autoliftout.setText(_translate("MainWindow", "Run AutoLiftout"))
        self.pushButton_thinning.setText(_translate("MainWindow", "Run Thinning"))
        self.pushButton_test_popup.setText(_translate("MainWindow", "Test Mode"))
        self.pushButton_add_sample_position.setText(_translate("MainWindow", "Add Sample Position"))
        self.label_stage.setText(_translate("MainWindow", "Status:"))
        self.label_title.setText(_translate("MainWindow", "AutoLiftout"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuChange_Mode.setTitle(_translate("MainWindow", "Change Mode"))
        self.menuUtilities.setTitle(_translate("MainWindow", "Utilities"))
        self.actionLoad_Experiment.setText(_translate("MainWindow", "Load Experiment"))
        self.actionLoad_Configuration.setText(_translate("MainWindow", "Load Configuration"))
        self.actionMark_Sample_Position_Failed.setText(_translate("MainWindow", "Mark Sample Position Failed"))
        self.actionAutoLiftout.setText(_translate("MainWindow", "AutoLiftout"))
        self.actionAutoLamella.setText(_translate("MainWindow", "AutoLamella"))
        self.actionSputter_Platinum.setText(_translate("MainWindow", "Sputter Platinum"))
        self.actionSharpen_Needle.setText(_translate("MainWindow", "Sharpen Needle"))
