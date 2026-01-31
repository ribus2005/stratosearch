import numpy as np
from matplotlib.colors import to_rgb
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter
from PySide6.QtWidgets import (
    QWidget, QFileDialog, QMessageBox,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsView,
    QGraphicsPathItem)
from PySide6.QtCore import Qt, QThread, Signal, QPointF

from stratosearch.gui.Controller import DataLoadWorker
from stratosearch.gui.Controller import InferenceWorker
from stratosearch.gui.Controller import ExportWorker
from stratosearch.gui.Controller import SplineWorker
from .EditableSpline import EditableSpline, SplineHandle
from .SettingWidget_ui import Ui_Form


class ClickableGraphicsView(QGraphicsView):
    # Сигнал с координатами клика в системе сцены
    clicked = Signal(float, float)

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())
        item = self.scene().itemAt(pos, self.transform())

        if isinstance(item, (SplineHandle, QGraphicsPathItem)):
            super().mousePressEvent(event)
            return

        self.clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(event)


class SettingWidget(QWidget):
    def __init__(self, app_dir):
        super().__init__()

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Цвета
        self.class_colors_codes = [
            '#000066',  # тёмно-синий   → 0
            '#0000cc',  # синий         → 1
            '#4444ff',  # светло-синий  → 2
            '#ff4444',  # светло-красный→ 3
            '#cc0000',  # красный       → 4
            '#660000'  # тёмно-красный → 5
        ]

        self.class_colors = []
        for color_code in self.class_colors_codes:
            r, g, b = to_rgb(color_code)  # получаем float 0–1
            self.class_colors.append(QColor(int(r * 255), int(g * 255), int(b * 255)))
        self.class_palette = np.array([(c.red(), c.green(), c.blue()) for c in self.class_colors], dtype=np.uint8)

        # Сцена для основного изображения
        self.scene_main = QGraphicsScene(self)
        self.ui.graphicsView.__class__ = ClickableGraphicsView
        self.ui.graphicsView.clicked.connect(self.on_image_clicked)
        self.ui.graphicsView.setScene(self.scene_main)

        self.original_image_item = QGraphicsPixmapItem()
        self.original_image_item.setZValue(0)
        self.image_scene_rect = None

        self.mask_item_main = QGraphicsPixmapItem()
        self.mask_item_main.setZValue(1)

        self.scene_main.addItem(self.original_image_item)
        self.scene_main.addItem(self.mask_item_main)

        # Сцена только для маски
        self.scene_mask = QGraphicsScene(self)
        self.ui.graphicsViewMask.setScene(self.scene_mask)

        self.mask_item_preview = QGraphicsPixmapItem()
        self.scene_mask.addItem(self.mask_item_preview)

        # Сплайны
        self.current_spline = None
        self.spline_building = False

        # Загружаем список моделей
        self.weights_dir = app_dir / "weights"
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

        # Дефолтные значения размеров
        self.ui.editWidth.setText("701")
        self.ui.editHeight.setText("255")

        # Тип данных для загрузки из .dat
        self.dat_dtype = np.float32

        # Интерактив с маской
        self.ui.sliderOpacity.valueChanged.connect(self.update_mask_display)
        self.ui.checkShowMask.stateChanged.connect(self.update_mask_display)

        self.original_pixmap = None
        self.mask_pixmap = None
        self.mask_array = None
        self.input_array = None

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

        model_files = [
            f for f in self.weights_dir.iterdir()
            if f.suffix.lower() in {".pth", ".pt"}
        ]

        if not model_files:
            self.ui.comboModels.addItem("Нет доступных моделей")
            self.ui.comboModels.setEnabled(False)
            return

        self.ui.comboModels.setEnabled(True)

        for file in model_files:
            model_name = file.stem
            self.models_info[model_name] = (str(file), model_name)
            self.ui.comboModels.addItem(model_name)

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
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите корректные Width и Height")
            return

        self.ui.editInput.setText(file_path)

        self.thread = QThread()
        self.worker = DataLoadWorker(file_path, width, height, self.dat_dtype)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_data_loaded)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.error.connect(self.show_error)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.start()

    def on_data_loaded(self, input_array, rgb_img):
        # --- Новые входные данные ---
        self.input_array = input_array
        self.original_pixmap = self.numpy_to_pixmap(rgb_img)

        # --- ПОЛНЫЙ СБРОС СТАРОЙ МАСКИ ---
        self.mask_array = None
        self.mask_pixmap = None

        # Убираем маску со сцен
        self.mask_item_main.setPixmap(QPixmap())
        self.mask_item_main.hide()

        self.mask_item_preview.setPixmap(QPixmap())
        self.mask_item_preview.hide()

        # --- УДАЛЕНИЕ СТАРОГО СПЛАЙНА ---
        if self.current_spline:
            self.current_spline.remove()
            self.current_spline = None

        self.spline_building = False

        # --- УСТАНОВКА НОВОГО ИЗОБРАЖЕНИЯ ---
        self.original_image_item.setPixmap(self.original_pixmap)

        self.image_scene_rect = self.original_image_item.mapToScene(
            self.original_image_item.boundingRect()
        ).boundingRect()

        self.scene_main.setSceneRect(self.original_image_item.boundingRect())
        self.ui.graphicsView.fitInView(self.original_image_item, Qt.KeepAspectRatio)

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
        self.worker = InferenceWorker(self.input_array, weight_path, model_name, self.class_palette)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_inference_done)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.error.connect(self.show_error)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.start()

    def on_inference_done(self, mask_array, rgb_mask):
        self.mask_array = mask_array
        self.mask_pixmap = self.numpy_to_pixmap(rgb_mask)

        self.update_mask_display()

        self.scene_mask.setSceneRect(self.mask_item_preview.boundingRect())
        self.ui.graphicsViewMask.fitInView(self.mask_item_preview, Qt.KeepAspectRatio)

    # ---------------- НАЛОЖЕНИЕ МАСКИ ----------------
    def update_mask_display(self):
        if self.mask_pixmap is None:
            self.mask_item_main.hide()
            self.mask_item_preview.hide()
            return

        visible = self.ui.checkShowMask.isChecked()
        opacity = self.ui.sliderOpacity.value() / 100.0

        self.mask_item_main.setPixmap(self.mask_pixmap)
        self.mask_item_main.setOpacity(opacity)
        self.mask_item_main.setVisible(True)

        self.mask_item_preview.setPixmap(self.mask_pixmap)
        self.mask_item_preview.setVisible(visible)

    # ---------------- ОБРАБОТКА КЛИКА ПОЛЬЗОВАТЕЛЯ ----------------
    def on_image_clicked(self, x, y):
        if self.spline_building:
            return

        if self.current_spline and self.current_spline.dragging:
            return

        if self.mask_pixmap is None or self.mask_array is None:
            return

        pos_scene = QPointF(x, y)
        pos_item = self.original_image_item.mapFromScene(pos_scene)

        ix = int(pos_item.x())
        iy = int(pos_item.y())

        h, w = self.mask_array.shape
        if not (0 <= ix < w and 0 <= iy < h):
            return

        self.spline_building = True

        self.thread = QThread()
        self.worker = SplineWorker(self.mask_array, ix, iy)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_splines_done)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.error.connect(self.show_error)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.start()

    def on_splines_done(self, points, class_id):
        self.spline_building = False

        if not points:
            return

        if self.current_spline:
            self.current_spline.remove()
            self.current_spline = None

        color = self.class_colors[class_id]
        self.current_spline = EditableSpline(self.scene_main,
                                             points,
                                             color,
                                             0.3,
                                             self.image_scene_rect)

    # ---------------- СОХРАНЕНИЕ ----------------
    def compose_result_pixmap(self):
        if self.original_pixmap is None:
            return None

        result = QPixmap(self.original_pixmap.size())
        result.fill(Qt.transparent)

        painter = QPainter(result)
        painter.drawPixmap(0, 0, self.original_pixmap)

        if self.mask_pixmap:
            opacity = self.ui.sliderOpacity.value() / 100.0
            painter.setOpacity(opacity)
            painter.drawPixmap(0, 0, self.mask_pixmap)

        painter.end()
        return result

    def download_image(self):
        if self.mask_array is None:
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

        if format_text.startswith("Image"):
            pixmap_to_save = self.compose_result_pixmap()
        else:
            pixmap_to_save = None

        self.thread = QThread()
        self.worker = ExportWorker(file_path, format_text, pixmap_to_save, self.mask_array)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.error.connect(self.show_error)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.start()

    def show_error(self, message):
        self.spline_building = False
        QMessageBox.critical(self, "Ошибка", message)

    @staticmethod
    def numpy_to_pixmap(arr):
        """
        Конвертирует numpy-массив (H×W или H×W×3) в QPixmap без сохранения на диск
        """
        arr = np.ascontiguousarray(arr, dtype=np.uint8)

        if arr.ndim == 2:
            h, w = arr.shape
            qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
        elif arr.ndim == 3 and arr.shape[2] == 3:  # RGB
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        else:
            raise ValueError("Неподдерживаемая форма массива для изображения")

        return QPixmap.fromImage(qimg.copy())
