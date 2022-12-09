# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AutoLiftoutProtocolUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(366, 367)
        Dialog.setMaximumSize(QtCore.QSize(16777215, 400))
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_exit = QtWidgets.QPushButton(Dialog)
        self.pushButton_exit.setObjectName("pushButton_exit")
        self.gridLayout.addWidget(self.pushButton_exit, 2, 1, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_option = QtWidgets.QWidget()
        self.tab_option.setObjectName("tab_option")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_option)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_options_liftout_joining_method = QtWidgets.QLabel(self.tab_option)
        self.label_options_liftout_joining_method.setObjectName("label_options_liftout_joining_method")
        self.gridLayout_2.addWidget(self.label_options_liftout_joining_method, 1, 0, 1, 1)
        self.label_options_landing_surface = QtWidgets.QLabel(self.tab_option)
        self.label_options_landing_surface.setObjectName("label_options_landing_surface")
        self.gridLayout_2.addWidget(self.label_options_landing_surface, 3, 0, 1, 1)
        self.comboBox_options_contact_direction = QtWidgets.QComboBox(self.tab_option)
        self.comboBox_options_contact_direction.setObjectName("comboBox_options_contact_direction")
        self.gridLayout_2.addWidget(self.comboBox_options_contact_direction, 2, 1, 1, 1)
        self.label_options_contact_direction = QtWidgets.QLabel(self.tab_option)
        self.label_options_contact_direction.setObjectName("label_options_contact_direction")
        self.gridLayout_2.addWidget(self.label_options_contact_direction, 2, 0, 1, 1)
        self.comboBox_options_liftout_joining_method = QtWidgets.QComboBox(self.tab_option)
        self.comboBox_options_liftout_joining_method.setObjectName("comboBox_options_liftout_joining_method")
        self.gridLayout_2.addWidget(self.comboBox_options_liftout_joining_method, 1, 1, 1, 1)
        self.comboBox_options_landing_surface = QtWidgets.QComboBox(self.tab_option)
        self.comboBox_options_landing_surface.setObjectName("comboBox_options_landing_surface")
        self.gridLayout_2.addWidget(self.comboBox_options_landing_surface, 3, 1, 1, 1)
        self.checkBox_options_confirm_next_stage = QtWidgets.QCheckBox(self.tab_option)
        self.checkBox_options_confirm_next_stage.setChecked(True)
        self.checkBox_options_confirm_next_stage.setObjectName("checkBox_options_confirm_next_stage")
        self.gridLayout_2.addWidget(self.checkBox_options_confirm_next_stage, 0, 1, 1, 1)
        self.checkBox_options_batch_mode = QtWidgets.QCheckBox(self.tab_option)
        self.checkBox_options_batch_mode.setChecked(True)
        self.checkBox_options_batch_mode.setObjectName("checkBox_options_batch_mode")
        self.gridLayout_2.addWidget(self.checkBox_options_batch_mode, 0, 0, 1, 1)
        self.label_options_landing_joining_method = QtWidgets.QLabel(self.tab_option)
        self.label_options_landing_joining_method.setObjectName("label_options_landing_joining_method")
        self.gridLayout_2.addWidget(self.label_options_landing_joining_method, 4, 0, 1, 1)
        self.comboBox_options_landing_joining_method = QtWidgets.QComboBox(self.tab_option)
        self.comboBox_options_landing_joining_method.setObjectName("comboBox_options_landing_joining_method")
        self.gridLayout_2.addWidget(self.comboBox_options_landing_joining_method, 4, 1, 1, 1)
        self.tabWidget.addTab(self.tab_option, "")
        self.tab_automation = QtWidgets.QWidget()
        self.tab_automation.setObjectName("tab_automation")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_automation)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.comboBox_auto_polishing = QtWidgets.QComboBox(self.tab_automation)
        self.comboBox_auto_polishing.setObjectName("comboBox_auto_polishing")
        self.gridLayout_3.addWidget(self.comboBox_auto_polishing, 6, 1, 1, 1)
        self.label_thinning = QtWidgets.QLabel(self.tab_automation)
        self.label_thinning.setObjectName("label_thinning")
        self.gridLayout_3.addWidget(self.label_thinning, 5, 0, 1, 1)
        self.label_trench = QtWidgets.QLabel(self.tab_automation)
        self.label_trench.setObjectName("label_trench")
        self.gridLayout_3.addWidget(self.label_trench, 0, 0, 1, 1)
        self.label_liftout = QtWidgets.QLabel(self.tab_automation)
        self.label_liftout.setObjectName("label_liftout")
        self.gridLayout_3.addWidget(self.label_liftout, 2, 0, 1, 1)
        self.label_polishing = QtWidgets.QLabel(self.tab_automation)
        self.label_polishing.setObjectName("label_polishing")
        self.gridLayout_3.addWidget(self.label_polishing, 6, 0, 1, 1)
        self.label_landing = QtWidgets.QLabel(self.tab_automation)
        self.label_landing.setObjectName("label_landing")
        self.gridLayout_3.addWidget(self.label_landing, 3, 0, 1, 1)
        self.comboBox_auto_landing = QtWidgets.QComboBox(self.tab_automation)
        self.comboBox_auto_landing.setObjectName("comboBox_auto_landing")
        self.gridLayout_3.addWidget(self.comboBox_auto_landing, 3, 1, 1, 1)
        self.comboBox_auto_mill_jcut = QtWidgets.QComboBox(self.tab_automation)
        self.comboBox_auto_mill_jcut.setObjectName("comboBox_auto_mill_jcut")
        self.gridLayout_3.addWidget(self.comboBox_auto_mill_jcut, 1, 1, 1, 1)
        self.comboBox_auto_liftout = QtWidgets.QComboBox(self.tab_automation)
        self.comboBox_auto_liftout.setObjectName("comboBox_auto_liftout")
        self.gridLayout_3.addWidget(self.comboBox_auto_liftout, 2, 1, 1, 1)
        self.label_jcut = QtWidgets.QLabel(self.tab_automation)
        self.label_jcut.setObjectName("label_jcut")
        self.gridLayout_3.addWidget(self.label_jcut, 1, 0, 1, 1)
        self.comboBox_auto_thinning = QtWidgets.QComboBox(self.tab_automation)
        self.comboBox_auto_thinning.setObjectName("comboBox_auto_thinning")
        self.gridLayout_3.addWidget(self.comboBox_auto_thinning, 5, 1, 1, 1)
        self.comboBox_auto_mill_trench = QtWidgets.QComboBox(self.tab_automation)
        self.comboBox_auto_mill_trench.setObjectName("comboBox_auto_mill_trench")
        self.gridLayout_3.addWidget(self.comboBox_auto_mill_trench, 0, 1, 1, 1)
        self.label_reset = QtWidgets.QLabel(self.tab_automation)
        self.label_reset.setObjectName("label_reset")
        self.gridLayout_3.addWidget(self.label_reset, 4, 0, 1, 1)
        self.comboBox_auto_reset = QtWidgets.QComboBox(self.tab_automation)
        self.comboBox_auto_reset.setObjectName("comboBox_auto_reset")
        self.gridLayout_3.addWidget(self.comboBox_auto_reset, 4, 1, 1, 1)
        self.tabWidget.addTab(self.tab_automation, "")
        self.tab_ml = QtWidgets.QWidget()
        self.tab_ml.setObjectName("tab_ml")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_ml)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.lineEdit_ml_weights = QtWidgets.QLineEdit(self.tab_ml)
        self.lineEdit_ml_weights.setObjectName("lineEdit_ml_weights")
        self.gridLayout_4.addWidget(self.lineEdit_ml_weights, 1, 1, 1, 1)
        self.spinBox_ml_num_classes = QtWidgets.QSpinBox(self.tab_ml)
        self.spinBox_ml_num_classes.setObjectName("spinBox_ml_num_classes")
        self.gridLayout_4.addWidget(self.spinBox_ml_num_classes, 2, 1, 1, 1)
        self.label_ml_encoder = QtWidgets.QLabel(self.tab_ml)
        self.label_ml_encoder.setObjectName("label_ml_encoder")
        self.gridLayout_4.addWidget(self.label_ml_encoder, 0, 0, 1, 1)
        self.label_ml_weights = QtWidgets.QLabel(self.tab_ml)
        self.label_ml_weights.setObjectName("label_ml_weights")
        self.gridLayout_4.addWidget(self.label_ml_weights, 1, 0, 1, 1)
        self.lineEdit_ml_encoder = QtWidgets.QLineEdit(self.tab_ml)
        self.lineEdit_ml_encoder.setObjectName("lineEdit_ml_encoder")
        self.gridLayout_4.addWidget(self.lineEdit_ml_encoder, 0, 1, 1, 1)
        self.label_ml_num_classes = QtWidgets.QLabel(self.tab_ml)
        self.label_ml_num_classes.setObjectName("label_ml_num_classes")
        self.gridLayout_4.addWidget(self.label_ml_num_classes, 2, 0, 1, 1)
        self.checkBox_ml_gpu = QtWidgets.QCheckBox(self.tab_ml)
        self.checkBox_ml_gpu.setObjectName("checkBox_ml_gpu")
        self.gridLayout_4.addWidget(self.checkBox_ml_gpu, 3, 0, 1, 2)
        self.tabWidget.addTab(self.tab_ml, "")
        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 2)
        self.label_title = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 2)
        self.pushButton_save = QtWidgets.QPushButton(Dialog)
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout.addWidget(self.pushButton_save, 2, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 2)

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton_exit.setText(_translate("Dialog", "Exit"))
        self.label_options_liftout_joining_method.setText(_translate("Dialog", "Liftout Joining Method"))
        self.label_options_landing_surface.setText(_translate("Dialog", "Landing Surface"))
        self.label_options_contact_direction.setText(_translate("Dialog", "Liftout Contact Direction"))
        self.checkBox_options_confirm_next_stage.setText(_translate("Dialog", "Confirm Next Stage"))
        self.checkBox_options_batch_mode.setText(_translate("Dialog", "Batch Mode"))
        self.label_options_landing_joining_method.setText(_translate("Dialog", "Landing Joining Method"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_option), _translate("Dialog", "Options"))
        self.label_thinning.setText(_translate("Dialog", "Thinning"))
        self.label_trench.setText(_translate("Dialog", "Mill Trench"))
        self.label_liftout.setText(_translate("Dialog", "Liftout"))
        self.label_polishing.setText(_translate("Dialog", "Polishing"))
        self.label_landing.setText(_translate("Dialog", "Landing"))
        self.label_jcut.setText(_translate("Dialog", "Mill JCut"))
        self.label_reset.setText(_translate("Dialog", "Reset Needle"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_automation), _translate("Dialog", "Automation"))
        self.label_ml_encoder.setText(_translate("Dialog", "Encoder"))
        self.label_ml_weights.setText(_translate("Dialog", "Weights"))
        self.label_ml_num_classes.setText(_translate("Dialog", "Num Classes"))
        self.checkBox_ml_gpu.setText(_translate("Dialog", "Use GPU"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_ml), _translate("Dialog", "Machine Learning"))
        self.label_title.setText(_translate("Dialog", "AutoLiftout Protocol"))
        self.pushButton_save.setText(_translate("Dialog", "Save"))
