# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SettingWidget.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGraphicsView,
    QHBoxLayout, QLabel, QLayout, QLineEdit,
    QPushButton, QSizePolicy, QSlider, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(929, 637)
        self.horizontalLayout_4 = QHBoxLayout(Form)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.heightLayout = QHBoxLayout()
        self.heightLayout.setSpacing(15)
        self.heightLayout.setObjectName(u"heightLayout")
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)

        self.heightLayout.addWidget(self.label_3)

        self.editHeight = QLineEdit(Form)
        self.editHeight.setObjectName(u"editHeight")

        self.heightLayout.addWidget(self.editHeight, 0, Qt.AlignmentFlag.AlignLeft)


        self.verticalLayout.addLayout(self.heightLayout)

        self.widthLayout = QHBoxLayout()
        self.widthLayout.setSpacing(10)
        self.widthLayout.setObjectName(u"widthLayout")
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)

        self.widthLayout.addWidget(self.label_2)

        self.editWidth = QLineEdit(Form)
        self.editWidth.setObjectName(u"editWidth")

        self.widthLayout.addWidget(self.editWidth, 0, Qt.AlignmentFlag.AlignLeft)


        self.verticalLayout.addLayout(self.widthLayout)

        self.inputDatLayout = QHBoxLayout()
        self.inputDatLayout.setSpacing(0)
        self.inputDatLayout.setObjectName(u"inputDatLayout")
        self.editInput = QLineEdit(Form)
        self.editInput.setObjectName(u"editInput")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.editInput.sizePolicy().hasHeightForWidth())
        self.editInput.setSizePolicy(sizePolicy1)
        self.editInput.setMaximumSize(QSize(16777215, 16777215))
        self.editInput.setReadOnly(True)

        self.inputDatLayout.addWidget(self.editInput)

        self.btnUpload = QPushButton(Form)
        self.btnUpload.setObjectName(u"btnUpload")
        sizePolicy.setHeightForWidth(self.btnUpload.sizePolicy().hasHeightForWidth())
        self.btnUpload.setSizePolicy(sizePolicy)
        self.btnUpload.setMaximumSize(QSize(30, 30))

        self.inputDatLayout.addWidget(self.btnUpload, 0, Qt.AlignmentFlag.AlignLeft)

        self.inputDatLayout.setStretch(0, 1)

        self.verticalLayout.addLayout(self.inputDatLayout)

        self.verticalSpacer_2 = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.verticalLayout.addItem(self.verticalSpacer_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.comboModels = QComboBox(Form)
        self.comboModels.setObjectName(u"comboModels")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.comboModels.sizePolicy().hasHeightForWidth())
        self.comboModels.setSizePolicy(sizePolicy2)

        self.horizontalLayout_2.addWidget(self.comboModels, 0, Qt.AlignmentFlag.AlignLeft)

        self.btnProcess = QPushButton(Form)
        self.btnProcess.setObjectName(u"btnProcess")
        sizePolicy1.setHeightForWidth(self.btnProcess.sizePolicy().hasHeightForWidth())
        self.btnProcess.setSizePolicy(sizePolicy1)
        self.btnProcess.setMaximumSize(QSize(100, 16777215))
        self.btnProcess.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.horizontalLayout_2.addWidget(self.btnProcess, 0, Qt.AlignmentFlag.AlignLeft)

        self.horizontalLayout_2.setStretch(1, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.verticalSpacer_4 = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.verticalLayout.addItem(self.verticalSpacer_4)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.comboSaveFormat = QComboBox(Form)
        self.comboSaveFormat.addItem("")
        self.comboSaveFormat.addItem("")
        self.comboSaveFormat.addItem("")
        self.comboSaveFormat.setObjectName(u"comboSaveFormat")

        self.horizontalLayout_3.addWidget(self.comboSaveFormat, 0, Qt.AlignmentFlag.AlignLeft)

        self.btnDownload = QPushButton(Form)
        self.btnDownload.setObjectName(u"btnDownload")
        sizePolicy1.setHeightForWidth(self.btnDownload.sizePolicy().hasHeightForWidth())
        self.btnDownload.setSizePolicy(sizePolicy1)
        self.btnDownload.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout_3.addWidget(self.btnDownload, 0, Qt.AlignmentFlag.AlignLeft)

        self.horizontalLayout_3.setStretch(1, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.downloadLayout = QVBoxLayout()
        self.downloadLayout.setSpacing(0)
        self.downloadLayout.setObjectName(u"downloadLayout")
        self.downloadLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.saveFormatLayout = QHBoxLayout()
        self.saveFormatLayout.setSpacing(10)
        self.saveFormatLayout.setObjectName(u"saveFormatLayout")

        self.downloadLayout.addLayout(self.saveFormatLayout)


        self.verticalLayout.addLayout(self.downloadLayout)


        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.verticalLayout_2.setStretch(1, 1)

        self.horizontalLayout_4.addLayout(self.verticalLayout_2)

        self.horizontalSpacer = QSpacerItem(20, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer)

        self.imagesSettingsLayout = QVBoxLayout()
        self.imagesSettingsLayout.setSpacing(10)
        self.imagesSettingsLayout.setObjectName(u"imagesSettingsLayout")
        self.checkShowMask = QCheckBox(Form)
        self.checkShowMask.setObjectName(u"checkShowMask")
        self.checkShowMask.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.checkShowMask.sizePolicy().hasHeightForWidth())
        self.checkShowMask.setSizePolicy(sizePolicy1)
        self.checkShowMask.setMinimumSize(QSize(0, 0))
        self.checkShowMask.setSizeIncrement(QSize(0, 0))
        self.checkShowMask.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.checkShowMask.setAutoFillBackground(False)
        self.checkShowMask.setTristate(False)

        self.imagesSettingsLayout.addWidget(self.checkShowMask, 0, Qt.AlignmentFlag.AlignRight)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setSpacing(10)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.graphicsView = QGraphicsView(Form)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setMinimumSize(QSize(1, 1))
        self.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.graphicsView.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphicsView.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.verticalLayout_3.addWidget(self.graphicsView)

        self.graphicsViewMask = QGraphicsView(Form)
        self.graphicsViewMask.setObjectName(u"graphicsViewMask")
        self.graphicsViewMask.setMinimumSize(QSize(1, 1))
        self.graphicsViewMask.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.graphicsViewMask.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphicsViewMask.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.verticalLayout_3.addWidget(self.graphicsViewMask)

        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 1)

        self.imagesSettingsLayout.addLayout(self.verticalLayout_3)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.opacityLayout = QHBoxLayout()
        self.opacityLayout.setSpacing(10)
        self.opacityLayout.setObjectName(u"opacityLayout")
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy2)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.opacityLayout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignHCenter)

        self.sliderOpacity = QSlider(Form)
        self.sliderOpacity.setObjectName(u"sliderOpacity")
        sizePolicy1.setHeightForWidth(self.sliderOpacity.sizePolicy().hasHeightForWidth())
        self.sliderOpacity.setSizePolicy(sizePolicy1)
        self.sliderOpacity.setMinimumSize(QSize(150, 0))
        self.sliderOpacity.setMaximumSize(QSize(16777215, 16777215))
        self.sliderOpacity.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.sliderOpacity.setMaximum(100)
        self.sliderOpacity.setValue(50)
        self.sliderOpacity.setOrientation(Qt.Orientation.Horizontal)

        self.opacityLayout.addWidget(self.sliderOpacity)

        self.opacityLayout.setStretch(0, 1)

        self.horizontalLayout.addLayout(self.opacityLayout)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(2, 1)

        self.imagesSettingsLayout.addLayout(self.horizontalLayout)

        self.imagesSettingsLayout.setStretch(1, 1)

        self.horizontalLayout_4.addLayout(self.imagesSettingsLayout)

        self.horizontalLayout_4.setStretch(2, 1)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"StratoSearch", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"\u0412\u044b\u0441\u043e\u0442\u0430", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"\u0428\u0438\u0440\u0438\u043d\u0430", None))
        self.editInput.setText(QCoreApplication.translate("Form", u"\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0444\u0430\u0439\u043b \u0434\u0430\u043d\u043d\u044b\u0445", None))
        self.btnUpload.setText(QCoreApplication.translate("Form", u"...", None))
        self.btnProcess.setText(QCoreApplication.translate("Form", u"Process", None))
        self.comboSaveFormat.setItemText(0, QCoreApplication.translate("Form", u"Image (*.png *.jpg *.jpeg)", None))
        self.comboSaveFormat.setItemText(1, QCoreApplication.translate("Form", u"Numpy (*.npy)", None))
        self.comboSaveFormat.setItemText(2, QCoreApplication.translate("Form", u"Data (*.dat)", None))

        self.btnDownload.setText(QCoreApplication.translate("Form", u"Download", None))
        self.checkShowMask.setText(QCoreApplication.translate("Form", u"\u041e\u0442\u043e\u0431\u0440\u0430\u0436\u0430\u0442\u044c \u041c\u0430\u0441\u043a\u0443", None))
        self.label.setText(QCoreApplication.translate("Form", u"\u041f\u0440\u043e\u0437\u0440\u0430\u0447\u043d\u043e\u0441\u0442\u044c \u043c\u0430\u0441\u043a\u0438", None))
    # retranslateUi

