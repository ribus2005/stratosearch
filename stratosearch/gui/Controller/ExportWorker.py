from PySide6.QtCore import QObject, Signal, Slot
import numpy as np


class ExportWorker(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, file_path, format_text, pixmap, mask_array):
        super().__init__()
        self.file_path = file_path
        self.format_text = format_text
        self.pixmap = pixmap
        self.mask_array = mask_array

    @Slot()
    def run(self):
        try:
            if self.format_text.startswith("Image"):
                self.pixmap.save(self.file_path)
            elif self.format_text.startswith("Numpy"):
                np.save(self.file_path, self.mask_array)
            elif self.format_text.startswith("Data"):
                self.mask_array.tofile(self.file_path)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
