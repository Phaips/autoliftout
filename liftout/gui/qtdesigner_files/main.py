# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1800, 895)
        MainWindow.setMinimumSize(QtCore.QSize(1600, 850))
        MainWindow.setMaximumSize(QtCore.QSize(1800, 1000))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
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
        self.Frame_Main.setMaximumSize(QtCore.QSize(1800, 850))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Frame_Main.setFont(font)
        self.Frame_Main.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_Main.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_Main.setObjectName("Frame_Main")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.Frame_Main)
        self.gridLayout_5.setContentsMargins(-1, -1, -1, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.Frame_Protocol = QtWidgets.QFrame(self.Frame_Main)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Frame_Protocol.sizePolicy().hasHeightForWidth())
        self.Frame_Protocol.setSizePolicy(sizePolicy)
        self.Frame_Protocol.setMinimumSize(QtCore.QSize(375, 0))
        self.Frame_Protocol.setMaximumSize(QtCore.QSize(300, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Frame_Protocol.setFont(font)
        self.Frame_Protocol.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_Protocol.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_Protocol.setObjectName("Frame_Protocol")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.Frame_Protocol)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tabWidget_Protocol = QtWidgets.QTabWidget(self.Frame_Protocol)
        self.tabWidget_Protocol.setEnabled(True)
        self.tabWidget_Protocol.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tabWidget_Protocol.setFont(font)
        self.tabWidget_Protocol.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.tabWidget_Protocol.setAccessibleName("")
        self.tabWidget_Protocol.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget_Protocol.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget_Protocol.setIconSize(QtCore.QSize(16, 16))
        self.tabWidget_Protocol.setElideMode(QtCore.Qt.ElideLeft)
        self.tabWidget_Protocol.setUsesScrollButtons(False)
        self.tabWidget_Protocol.setDocumentMode(False)
        self.tabWidget_Protocol.setTabsClosable(False)
        self.tabWidget_Protocol.setMovable(True)
        self.tabWidget_Protocol.setTabBarAutoHide(False)
        self.tabWidget_Protocol.setObjectName("tabWidget_Protocol")
        self.gridLayout_4.addWidget(self.tabWidget_Protocol, 0, 0, 1, 1)
        self.Frame_load_save = QtWidgets.QFrame(self.Frame_Protocol)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Frame_load_save.setFont(font)
        self.Frame_load_save.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_load_save.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_load_save.setObjectName("Frame_load_save")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.Frame_load_save)
        self.gridLayout_13.setContentsMargins(-1, 1, -1, 1)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.pushButton_Protocol_Load = QtWidgets.QPushButton(self.Frame_load_save)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_Protocol_Load.setFont(font)
        self.pushButton_Protocol_Load.setObjectName("pushButton_Protocol_Load")
        self.gridLayout_13.addWidget(self.pushButton_Protocol_Load, 0, 0, 1, 1)
        self.pushButton_Protocol_Save = QtWidgets.QPushButton(self.Frame_load_save)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_Protocol_Save.setFont(font)
        self.pushButton_Protocol_Save.setObjectName("pushButton_Protocol_Save")
        self.gridLayout_13.addWidget(self.pushButton_Protocol_Save, 0, 1, 1, 1)
        self.pushButton_Protocol_Save_As = QtWidgets.QPushButton(self.Frame_load_save)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_Protocol_Save_As.setFont(font)
        self.pushButton_Protocol_Save_As.setObjectName("pushButton_Protocol_Save_As")
        self.gridLayout_13.addWidget(self.pushButton_Protocol_Save_As, 0, 2, 1, 1)
        self.gridLayout_4.addWidget(self.Frame_load_save, 2, 0, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.Frame_Protocol)
        self.frame_2.setMinimumSize(QtCore.QSize(0, 24))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_2.setFont(font)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout.setContentsMargins(-1, 1, -1, 1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_Protocol_New = QtWidgets.QPushButton(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_Protocol_New.setFont(font)
        self.pushButton_Protocol_New.setObjectName("pushButton_Protocol_New")
        self.horizontalLayout.addWidget(self.pushButton_Protocol_New)
        self.pushButton_Protocol_Rename = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_Protocol_Rename.setObjectName("pushButton_Protocol_Rename")
        self.horizontalLayout.addWidget(self.pushButton_Protocol_Rename)
        self.pushButton_Protocol_Delete = QtWidgets.QPushButton(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_Protocol_Delete.setFont(font)
        self.pushButton_Protocol_Delete.setObjectName("pushButton_Protocol_Delete")
        self.horizontalLayout.addWidget(self.pushButton_Protocol_Delete)
        self.gridLayout_4.addWidget(self.frame_2, 1, 0, 1, 1)
        self.gridLayout_5.addWidget(self.Frame_Protocol, 0, 0, 1, 1)
        self.frame_Images = QtWidgets.QFrame(self.Frame_Main)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_Images.setFont(font)
        self.frame_Images.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_Images.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_Images.setObjectName("frame_Images")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_Images)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_SEM = QtWidgets.QFrame(self.frame_Images)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_SEM.sizePolicy().hasHeightForWidth())
        self.frame_SEM.setSizePolicy(sizePolicy)
        self.frame_SEM.setMinimumSize(QtCore.QSize(675, 450))
        self.frame_SEM.setMaximumSize(QtCore.QSize(675, 450))
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
        self.label_SEM.setMinimumSize(QtCore.QSize(675, 450))
        self.label_SEM.setMaximumSize(QtCore.QSize(675, 450))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_SEM.setFont(font)
        self.label_SEM.setText("")
        self.label_SEM.setObjectName("label_SEM")
        self.gridLayout.addWidget(self.label_SEM, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_SEM, 0, 0, 1, 1)
        self.frame_FIB = QtWidgets.QFrame(self.frame_Images)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_FIB.sizePolicy().hasHeightForWidth())
        self.frame_FIB.setSizePolicy(sizePolicy)
        self.frame_FIB.setMinimumSize(QtCore.QSize(675, 450))
        self.frame_FIB.setMaximumSize(QtCore.QSize(675, 450))
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
        self.label_FIB.setMinimumSize(QtCore.QSize(675, 450))
        self.label_FIB.setMaximumSize(QtCore.QSize(675, 450))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_FIB.setFont(font)
        self.label_FIB.setText("")
        self.label_FIB.setObjectName("label_FIB")
        self.gridLayout_2.addWidget(self.label_FIB, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_FIB, 0, 1, 1, 1)
        self.frame_FIBSEM_controls = QtWidgets.QFrame(self.frame_Images)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_FIBSEM_controls.sizePolicy().hasHeightForWidth())
        self.frame_FIBSEM_controls.setSizePolicy(sizePolicy)
        self.frame_FIBSEM_controls.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_FIBSEM_controls.setFont(font)
        self.frame_FIBSEM_controls.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_FIBSEM_controls.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_FIBSEM_controls.setObjectName("frame_FIBSEM_controls")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.frame_FIBSEM_controls)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.frame_acquire_FIB = QtWidgets.QFrame(self.frame_FIBSEM_controls)
        self.frame_acquire_FIB.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_acquire_FIB.setFont(font)
        self.frame_acquire_FIB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_acquire_FIB.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_acquire_FIB.setObjectName("frame_acquire_FIB")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_acquire_FIB)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.button_get_image_FIB = QtWidgets.QPushButton(self.frame_acquire_FIB)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_get_image_FIB.sizePolicy().hasHeightForWidth())
        self.button_get_image_FIB.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_get_image_FIB.setFont(font)
        self.button_get_image_FIB.setObjectName("button_get_image_FIB")
        self.verticalLayout.addWidget(self.button_get_image_FIB)
        self.button_last_image_FIB = QtWidgets.QPushButton(self.frame_acquire_FIB)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_last_image_FIB.sizePolicy().hasHeightForWidth())
        self.button_last_image_FIB.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_last_image_FIB.setFont(font)
        self.button_last_image_FIB.setObjectName("button_last_image_FIB")
        self.verticalLayout.addWidget(self.button_last_image_FIB)
        self.gridLayout_12.addWidget(self.frame_acquire_FIB, 0, 1, 1, 1)
        self.frame_3 = QtWidgets.QFrame(self.frame_FIBSEM_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMaximumSize(QtCore.QSize(245, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_3.setFont(font)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_9.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_10 = QtWidgets.QLabel(self.frame_3)
        self.label_10.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout_9.addWidget(self.label_10, 0, 0, 1, 1)
        self.lineEdit_dwell_time = QtWidgets.QLineEdit(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_dwell_time.sizePolicy().hasHeightForWidth())
        self.lineEdit_dwell_time.setSizePolicy(sizePolicy)
        self.lineEdit_dwell_time.setMaximumSize(QtCore.QSize(90, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_dwell_time.setFont(font)
        self.lineEdit_dwell_time.setObjectName("lineEdit_dwell_time")
        self.gridLayout_9.addWidget(self.lineEdit_dwell_time, 1, 0, 1, 1)
        self.checkBox_Autocontrast = QtWidgets.QCheckBox(self.frame_3)
        self.checkBox_Autocontrast.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.checkBox_Autocontrast.setFont(font)
        self.checkBox_Autocontrast.setObjectName("checkBox_Autocontrast")
        self.gridLayout_9.addWidget(self.checkBox_Autocontrast, 2, 0, 1, 1)
        self.gridLayout_12.addWidget(self.frame_3, 0, 4, 1, 1)
        self.connect_microscope = QtWidgets.QPushButton(self.frame_FIBSEM_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.connect_microscope.sizePolicy().hasHeightForWidth())
        self.connect_microscope.setSizePolicy(sizePolicy)
        self.connect_microscope.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.connect_microscope.setFont(font)
        self.connect_microscope.setObjectName("connect_microscope")
        self.gridLayout_12.addWidget(self.connect_microscope, 0, 2, 1, 1)
        self.frame_acquire_SEM = QtWidgets.QFrame(self.frame_FIBSEM_controls)
        self.frame_acquire_SEM.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_acquire_SEM.setFont(font)
        self.frame_acquire_SEM.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_acquire_SEM.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_acquire_SEM.setObjectName("frame_acquire_SEM")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_acquire_SEM)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.button_get_image_SEM = QtWidgets.QPushButton(self.frame_acquire_SEM)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_get_image_SEM.sizePolicy().hasHeightForWidth())
        self.button_get_image_SEM.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_get_image_SEM.setFont(font)
        self.button_get_image_SEM.setObjectName("button_get_image_SEM")
        self.verticalLayout_5.addWidget(self.button_get_image_SEM)
        self.button_last_image_SEM = QtWidgets.QPushButton(self.frame_acquire_SEM)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_last_image_SEM.sizePolicy().hasHeightForWidth())
        self.button_last_image_SEM.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_last_image_SEM.setFont(font)
        self.button_last_image_SEM.setObjectName("button_last_image_SEM")
        self.verticalLayout_5.addWidget(self.button_last_image_SEM)
        self.gridLayout_12.addWidget(self.frame_acquire_SEM, 0, 0, 1, 1)
        self.frame_4 = QtWidgets.QFrame(self.frame_FIBSEM_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setMaximumSize(QtCore.QSize(245, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame_4.setFont(font)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_10.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_13 = QtWidgets.QLabel(self.frame_4)
        self.label_13.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_10.addWidget(self.label_13, 0, 0, 1, 1)
        self.comboBox_resolution = QtWidgets.QComboBox(self.frame_4)
        self.comboBox_resolution.setMinimumSize(QtCore.QSize(0, 0))
        self.comboBox_resolution.setMaximumSize(QtCore.QSize(100, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.comboBox_resolution.setFont(font)
        self.comboBox_resolution.setObjectName("comboBox_resolution")
        self.comboBox_resolution.addItem("")
        self.comboBox_resolution.addItem("")
        self.comboBox_resolution.addItem("")
        self.comboBox_resolution.addItem("")
        self.gridLayout_10.addWidget(self.comboBox_resolution, 1, 0, 1, 1)
        self.gridLayout_12.addWidget(self.frame_4, 0, 3, 1, 1)
        self.gridLayout_3.addWidget(self.frame_FIBSEM_controls, 1, 0, 1, 2)
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
        self.gridLayout_8.setContentsMargins(-1, -1, -1, 0)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_6 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_8.addWidget(self.label_6, 0, 5, 1, 1)
        self.label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_8.addWidget(self.label, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_8.addWidget(self.label_5, 0, 4, 1, 1)
        self.tabWidget_Information = QtWidgets.QTabWidget(self.frame)
        self.tabWidget_Information.setMinimumSize(QtCore.QSize(0, 200))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tabWidget_Information.setFont(font)
        self.tabWidget_Information.setMovable(True)
        self.tabWidget_Information.setObjectName("tabWidget_Information")
        self.gridLayout_8.addWidget(self.tabWidget_Information, 2, 0, 1, 7)
        self.label_3 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_8.addWidget(self.label_3, 0, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_8.addWidget(self.label_4, 0, 3, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_8.addWidget(self.label_7, 0, 6, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_8.addWidget(self.label_2, 0, 1, 1, 1)
        self.pushButton_autoliftout = QtWidgets.QPushButton(self.frame)
        self.pushButton_autoliftout.setObjectName("pushButton_autoliftout")
        self.gridLayout_8.addWidget(self.pushButton_autoliftout, 1, 6, 1, 1)
        self.pushButton_initialise = QtWidgets.QPushButton(self.frame)
        self.pushButton_initialise.setObjectName("pushButton_initialise")
        self.gridLayout_8.addWidget(self.pushButton_initialise, 1, 5, 1, 1)
        self.pushButton_load_sample_data = QtWidgets.QPushButton(self.frame)
        self.pushButton_load_sample_data.setObjectName("pushButton_load_sample_data")
        self.gridLayout_8.addWidget(self.pushButton_load_sample_data, 1, 4, 1, 1)
        self.pushButton_test_popup = QtWidgets.QPushButton(self.frame)
        self.pushButton_test_popup.setObjectName("pushButton_test_popup")
        self.gridLayout_8.addWidget(self.pushButton_test_popup, 1, 3, 1, 1)
        self.gridLayout_3.addWidget(self.frame, 2, 0, 1, 2)
        self.gridLayout_5.addWidget(self.frame_Images, 0, 1, 1, 1)
        self.gridLayout_6.addWidget(self.Frame_Main, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget_Protocol.setCurrentIndex(-1)
        self.tabWidget_Information.setCurrentIndex(-1)
        self.tabWidget_Protocol.currentChanged['int'].connect(self.tabWidget_Information.setCurrentIndex)
        self.tabWidget_Information.currentChanged['int'].connect(self.tabWidget_Protocol.setCurrentIndex)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_Protocol_Load.setText(_translate("MainWindow", "Load"))
        self.pushButton_Protocol_Save.setText(_translate("MainWindow", "Save"))
        self.pushButton_Protocol_Save_As.setText(_translate("MainWindow", "Save As.."))
        self.pushButton_Protocol_New.setText(_translate("MainWindow", "New"))
        self.pushButton_Protocol_Rename.setText(_translate("MainWindow", "Rename"))
        self.pushButton_Protocol_Delete.setText(_translate("MainWindow", "Delete"))
        self.button_get_image_FIB.setText(_translate("MainWindow", "Get FIB Image"))
        self.button_last_image_FIB.setText(_translate("MainWindow", "Grab Last FIB"))
        self.label_10.setText(_translate("MainWindow", "Dwell Time (us)"))
        self.lineEdit_dwell_time.setText(_translate("MainWindow", "1"))
        self.checkBox_Autocontrast.setText(_translate("MainWindow", "Autocontrast"))
        self.connect_microscope.setText(_translate("MainWindow", "Connect to microscope"))
        self.button_get_image_SEM.setText(_translate("MainWindow", "Get SEM Image"))
        self.button_last_image_SEM.setText(_translate("MainWindow", "Grab Last SEM"))
        self.label_13.setText(_translate("MainWindow", "Resolution"))
        self.comboBox_resolution.setItemText(0, _translate("MainWindow", "768x512"))
        self.comboBox_resolution.setItemText(1, _translate("MainWindow", "1536x1024"))
        self.comboBox_resolution.setItemText(2, _translate("MainWindow", "3072x2048"))
        self.comboBox_resolution.setItemText(3, _translate("MainWindow", "6144x4096"))
        self.label_6.setText(_translate("MainWindow", "TextLabel"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.label_5.setText(_translate("MainWindow", "TextLabel"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))
        self.label_7.setText(_translate("MainWindow", "TextLabel"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_autoliftout.setText(_translate("MainWindow", "Run autoliftout"))
        self.pushButton_initialise.setText(_translate("MainWindow", "Initialise"))
        self.pushButton_load_sample_data.setText(_translate("MainWindow", "Load Sample Data"))
        self.pushButton_test_popup.setText(_translate("MainWindow", "Test_Popup"))
