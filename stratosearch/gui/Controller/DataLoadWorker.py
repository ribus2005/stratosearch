from PySide6.QtCore import QObject, Signal, Slot
import numpy as np
import matplotlib.cm as cm


class DataLoadWorker(QObject):
    finished = Signal(object, object)  # input_array, rgb_image
    error = Signal(str)

    def __init__(self, file_path, width, height, dtype):
        super().__init__()
        self.file_path = file_path
        self.width = width
        self.height = height
        self.dtype = dtype

    @Slot()
    def run(self):
        try:
            data = np.fromfile(self.file_path, dtype=self.dtype)
            data = data.reshape((self.height, self.width), order="F")

            rgb = self.apply_seismic_colormap(data)

            self.finished.emit(data, rgb)
        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def apply_seismic_colormap(arr):
        arr = arr.astype(np.float32)
        min_val, max_val = arr.min(), arr.max()

        if max_val - min_val == 0:
            norm = np.zeros_like(arr, dtype=np.float32)
        else:
            norm = (arr - min_val) / (max_val - min_val)

        cmap = cm.get_cmap('seismic')
        colored = cmap(norm)[:, :, :3]
        return (colored * 255).astype(np.uint8)
