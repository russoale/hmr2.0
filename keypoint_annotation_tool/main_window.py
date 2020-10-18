# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 5.15.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *

from widget.smpl_widget import SmplWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1024, 984)
        self.action_min_mesh = QAction(MainWindow)
        self.action_min_mesh.setObjectName(u"action_min_mesh")
        self.action_min_vertices = QAction(MainWindow)
        self.action_min_vertices.setObjectName(u"action_min_vertices")
        self.action_min_triangles = QAction(MainWindow)
        self.action_min_triangles.setObjectName(u"action_min_triangles")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setEnabled(True)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.openGLWidget = SmplWidget(self.centralwidget)
        self.openGLWidget.setObjectName(u"openGLWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.openGLWidget.sizePolicy().hasHeightForWidth())
        self.openGLWidget.setSizePolicy(sizePolicy)

        self.horizontalLayout.addWidget(self.openGLWidget)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 5, 0, 1, 1)

        self.convert_button = QPushButton(self.centralwidget)
        self.convert_button.setObjectName(u"convert_button")
        self.convert_button.setEnabled(False)

        self.gridLayout.addWidget(self.convert_button, 6, 1, 1, 1)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 7, 0, 1, 3)

        self.male_button = QRadioButton(self.centralwidget)
        self.male_button.setObjectName(u"male_button")
        self.male_button.setChecked(False)

        self.gridLayout.addWidget(self.male_button, 4, 2, 1, 1)

        self.poses_box = QComboBox(self.centralwidget)
        self.poses_box.setObjectName(u"poses_box")

        self.gridLayout.addWidget(self.poses_box, 2, 0, 1, 3)

        self.reset_button = QPushButton(self.centralwidget)
        self.reset_button.setObjectName(u"reset_button")

        self.gridLayout.addWidget(self.reset_button, 6, 2, 1, 1)

        self.regressor_name = QLineEdit(self.centralwidget)
        self.regressor_name.setObjectName(u"regressor_name")
        self.regressor_name.setMaxLength(32)

        self.gridLayout.addWidget(self.regressor_name, 6, 0, 1, 1)

        self.neutral_button = QRadioButton(self.centralwidget)
        self.neutral_button.setObjectName(u"neutral_button")
        self.neutral_button.setChecked(True)

        self.gridLayout.addWidget(self.neutral_button, 4, 0, 1, 1)

        self.female_button = QRadioButton(self.centralwidget)
        self.female_button.setObjectName(u"female_button")

        self.gridLayout.addWidget(self.female_button, 4, 1, 1, 1)


        self.horizontalLayout.addLayout(self.gridLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setEnabled(True)
        self.menubar.setGeometry(QRect(0, 0, 1024, 22))
        self.menuSmpl_Kepoint_Marker = QMenu(self.menubar)
        self.menuSmpl_Kepoint_Marker.setObjectName(u"menuSmpl_Kepoint_Marker")
        MainWindow.setMenuBar(self.menubar)

        self.menubar.addAction(self.menuSmpl_Kepoint_Marker.menuAction())
        self.menuSmpl_Kepoint_Marker.addAction(self.action_min_mesh)
        self.menuSmpl_Kepoint_Marker.addAction(self.action_min_vertices)
        self.menuSmpl_Kepoint_Marker.addAction(self.action_min_triangles)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.action_min_mesh.setText(QCoreApplication.translate("MainWindow", u"Minimum mesh", None))
        self.action_min_vertices.setText(QCoreApplication.translate("MainWindow", u"Minimum vertices", None))
        self.action_min_triangles.setText(QCoreApplication.translate("MainWindow", u"Minimum triangles", None))
        self.convert_button.setText(QCoreApplication.translate("MainWindow", u"Convert", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'.AppleSystemUIFont'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; text-decoration: underline;\">Commands</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600; text-decoration: underline;\">Note!!</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; color:#fc0107;\">In some cases smoothing changes the mesh topology!<br />This can cause the re"
                        "gressor generation to fail or <br />leading to completely wrong vertex indices!</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:'JetBrains Mono','monospace'; color:#a6aa43;\"><br /></p>\n"
"<table border=\"0\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px;\" cellspacing=\"2\" cellpadding=\"0\">\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">mouse drag</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">rotates the view</p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">ctrl/cmd + mouse dra"
                        "g</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">pans the view</p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">mouse wheel</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">zooms the view</p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">z</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">returns to the base view</p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-righ"
                        "t:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">v</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">toggles smoothing </p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">j</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">toggles joints</p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">r</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">toggles camera rays</p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p styl"
                        "e=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">w</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">toggles wireframe mode</p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">c</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">toggles backface culling</p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">f</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0"
                        "; text-indent:0px;\">toggles between fullscreen <br />and windowed mode</p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">m</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">maximizes the window</p></td></tr>\n"
"<tr>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">q</span></p></td>\n"
"<td>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">closes the window</p></td></tr></table></body></html>", None))
        self.male_button.setText(QCoreApplication.translate("MainWindow", u"Male", None))
        self.reset_button.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.regressor_name.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Set regressor name...", None))
        self.neutral_button.setText(QCoreApplication.translate("MainWindow", u"Neutral", None))
        self.female_button.setText(QCoreApplication.translate("MainWindow", u"Female", None))
        self.menuSmpl_Kepoint_Marker.setTitle(QCoreApplication.translate("MainWindow", u"Smpl Kepoint Marker", None))
    # retranslateUi

