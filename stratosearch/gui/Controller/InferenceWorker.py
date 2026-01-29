import matplotlib.cm as cm
import numpy as np
import torch

from PySide6.QtCore import QObject, Signal, Slot

from stratosearch.calculate.calculate import POSTPROCESSORS, default_postprocess
from stratosearch.calculate.calculate import PREPROCESSORS, default_preprocess
from stratosearch.calculate.models import MODELS


class InferenceWorker(QObject):
    finished = Signal(object, object)  # mask_array, rgb_mask
    error = Signal(str)

    def __init__(self, input_array, weight_path, model_name):
        super().__init__()
        self.input_array = input_array
        self.model_name = model_name

        self.model = MODELS.get(model_name)()
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        self.model.eval()

        self.mask_array = None
        self.rgb_mask = None

    @Slot()
    def run(self):
        try:
            self.run_segmentation()
            self.finished.emit(self.mask_array, self.rgb_mask)
        except Exception as e:
            self.error.emit(str(e))

    # ---------------- СЕГМЕНТАЦИЯ ---------------- #

    def run_segmentation(self):
        preprocess_fn = PREPROCESSORS.get(self.model_name, default_preprocess)
        preprocessed_input_tensor = preprocess_fn(self.input_array)

        with torch.no_grad():
            output = self.model.forward(preprocessed_input_tensor)

        postprocess_fn = POSTPROCESSORS.get(self.model_name, default_postprocess)
        self.mask_array = postprocess_fn(output)
        self.rgb_mask = self.colorize_classes(self.mask_array)

    # ---------------- ВИЗУАЛИЗАЦИЯ ---------------- #

    @staticmethod
    def colorize_classes(mask_array):
        mask_array = mask_array.astype(np.float32)

        min_val, max_val = mask_array.min(), mask_array.max()
        if max_val - min_val == 0:
            norm = np.zeros_like(mask_array)
        else:
            norm = (mask_array - min_val) / (max_val - min_val)

        cmap = cm.get_cmap("viridis")
        colored = cmap(norm)[..., :3]
        return (colored * 255).astype(np.uint8)
