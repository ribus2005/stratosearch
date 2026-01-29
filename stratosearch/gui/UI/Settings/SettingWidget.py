import sys
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import QWidget, QFileDialog, QMessageBox
from PySide6.QtCore import Qt, QThread

from stratosearch.gui.Controller import DataLoadWorker
from stratosearch.gui.Controller import InferenceWorker
from stratosearch.gui.Controller import ExportWorker
from .SettingWidget_ui import Ui_Form


class SettingWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Загружаем список моделей
        self.weights_dir = self.get_app_dir() / "weights"
        self.ensure_weights_dir()

        self.models_info = {}  # display_name → (weight_path, model_name)
        self.load_available_models()

        # Ресайз окна
        self.setWindowTitle("StratoSearch")
        self.resize(900, 600)

        # Подключаем кнопки
        self.ui.btnUpload.clicked.connect(self.upload_image)
        self.ui.btnProcess.clicked.connect(self.process_image)
        self.ui.btnDownload.clicked.connect(self.download_image)

        # Слайдер прозрачности
        self.ui.sliderOpacity.valueChanged.connect(self.update_overlay)

        # Отображение маски
        self.ui.checkShowMask.stateChanged.connect(self.update_mask_visibility)

        self.original_pixmap = None
        self.mask_pixmap = None
        self.result_pixmap = None
        self.mask_array = None
        self.input_array = None

    @staticmethod
    def get_app_dir():
        if getattr(sys, 'frozen', False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parents[2]

    def ensure_weights_dir(self):
        """
        Создаёт папку weights, если она отсутствует
        """
        try:
            self.weights_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось создать папку весов:\n{self.weights_dir}\n\n{e}"
            )

    # ---------------- ЧТЕНИЕ ВЕСОВ МОДЕЛЕЙ ----------------
    def load_available_models(self):
        self.ui.comboModels.clear()
        self.models_info.clear()

        model_files = list(self.weights_dir.glob("*.pth"))

        if not model_files:
            self.ui.comboModels.addItem("Нет доступных моделей")
            self.ui.comboModels.setEnabled(False)
            return

        self.ui.comboModels.setEnabled(True)

        for file in model_files:
            name = file.stem
            if "_" not in name:
                continue

            model_name, version = name.rsplit("_", 1)
            display = f"{model_name} ({version})"

            self.models_info[display] = (str(file), model_name)
            self.ui.comboModels.addItem(display)

    # ---------------- ЗАГРУЗКА ИЗОБРАЖЕНИЯ ----------------
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
            "Выберите .dat файл",
            "",
            "Data files (*.dat)"
        )
        if not file_path:
            return

        try:
            width = int(self.ui.editWidth.text())
            height = int(self.ui.editHeight.text())
            dtype_str = self.ui.comboDtype.currentText()
            rotation = int(self.ui.comboRotation.currentText())
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите корректные Width и Height")
            return

        dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "int16": np.int16,
            "int32": np.int32,
        }

        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            QMessageBox.warning(self, "Ошибка", "Неподдерживаемый тип данных")
            return

        self.ui.editInput.setText(file_path)

        self.thread = QThread()
        self.worker = DataLoadWorker(file_path, width, height, dtype, rotation)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_data_loaded)
        self.worker.error.connect(self.show_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_data_loaded(self, input_array, rgb_img):
        self.input_array = input_array
        self.original_pixmap = self.numpy_to_pixmap(rgb_img)

        self.mask_pixmap = None
        self.result_pixmap = None

        self.show_pixmap(self.original_pixmap, self.ui.labelInputImage)
        self.ui.labelMaskImage.clear()
        self.ui.checkShowMask.setChecked(False)

    # ---------------- СЕГМЕНТАЦИЯ ----------------
    def process_image(self):
        if self.input_array is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные")
            return

        display_name = self.ui.comboModels.currentText()
        if display_name not in self.models_info:
            QMessageBox.warning(self, "Ошибка", "Модель не выбрана")
            return

        weight_path, model_name = self.models_info[display_name]

        self.thread = QThread()
        self.worker = InferenceWorker(self.input_array, weight_path, model_name)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_inference_done)
        self.worker.error.connect(self.show_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_inference_done(self, mask_array, rgb_mask):
        self.mask_array = mask_array
        self.mask_pixmap = self.numpy_to_pixmap(rgb_mask)

        self.update_mask_visibility()
        self.update_overlay()

    # ---------------- НАЛОЖЕНИЕ МАСКИ ----------------
    def update_mask_visibility(self):
        """Показывает или скрывает маску в правом QLabel"""
        if not self.ui.checkShowMask.isChecked():
            self.ui.labelMaskImage.clear()
            return

        if self.mask_pixmap is None:
            return

        self.show_pixmap(self.mask_pixmap, self.ui.labelMaskImage)

    def update_overlay(self):
        if not self.original_pixmap:
            return

        self.result_pixmap = QPixmap(self.original_pixmap.size())
        self.result_pixmap.fill(Qt.transparent)

        painter = QPainter(self.result_pixmap)
        painter.drawPixmap(0, 0, self.original_pixmap)

        # ---------- МАСКА ----------
        if self.mask_pixmap:
            scaled_mask = self.mask_pixmap.scaled(
                self.original_pixmap.size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            opacity = self.ui.sliderOpacity.value() / 100.0
            painter.setOpacity(opacity)
            painter.drawPixmap(0, 0, scaled_mask)
            painter.setOpacity(1.0)

        painter.end()

        self.show_pixmap(self.result_pixmap, self.ui.labelInputImage)

    # ---------------- ОТОБРАЖЕНИЕ ----------------
    @staticmethod
    def show_pixmap(pixmap, label):
        if pixmap is None or label.width() < 10 or label.height() < 10:
            return

        scaled = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Левое изображение (оригинал или результат)
        if self.result_pixmap:
            self.show_pixmap(self.result_pixmap, self.ui.labelInputImage)
        elif self.original_pixmap:
            self.show_pixmap(self.original_pixmap, self.ui.labelInputImage)

        # Правое изображение (маска)
        if self.ui.checkShowMask.isChecked() and self.mask_pixmap:
            self.show_pixmap(self.mask_pixmap, self.ui.labelMaskImage)

    # ---------------- СОХРАНЕНИЕ ----------------
    def download_image(self):
        if self.result_pixmap is None and self.mask_array is None:
            QMessageBox.warning(self, "Ошибка", "Нет результата для сохранения")
            return

        format_text = self.ui.comboSaveFormat.currentText()

        if format_text.startswith("Image"):
            file_filter = "Images (*.png *.jpg *.jpeg)"
        elif format_text.startswith("Numpy"):
            file_filter = "Numpy (*.npy)"
        elif format_text.startswith("Data"):
            file_filter = "Data (*.dat)"
        else:
            QMessageBox.warning(self, "Ошибка", "Неподдерживаемый формат")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", "result", file_filter)
        if not file_path:
            return

        pixmap_to_save = self.result_pixmap or self.original_pixmap

        self.thread = QThread()
        self.worker = ExportWorker(file_path, format_text, pixmap_to_save, self.mask_array)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.error.connect(self.show_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def show_error(self, message):
        QMessageBox.critical(self, "Ошибка", message)

    @staticmethod
    def numpy_to_pixmap(arr):
        """
        Конвертирует numpy-массив (H×W или H×W×3) в QPixmap без сохранения на диск
        """
        arr = np.ascontiguousarray(arr, dtype=np.uint8)  # гарантируем непрерывную память

        if arr.ndim == 2:  # Grayscale
            h, w = arr.shape
            qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
        elif arr.ndim == 3 and arr.shape[2] == 3:  # RGB
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        else:
            raise ValueError("Неподдерживаемая форма массива для изображения")

        return QPixmap.fromImage(qimg)
