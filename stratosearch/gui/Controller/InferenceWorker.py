import numpy as np
import torch

from PySide6.QtCore import QObject, Signal, Slot

from stratosearch.calculate.calculate import POSTPROCESSORS, default_postprocess
from stratosearch.calculate.calculate import PREPROCESSORS, default_preprocess
from stratosearch.calculate.models import UNet, UNET_FEATURES


class InferenceWorker(QObject):
    finished = Signal(object, object)  # mask_array, rgb_mask
    error = Signal(str)

    def __init__(self, input_array, weight_path, model_name, class_palette):
        super().__init__()
        self.input_array = input_array
        self.model_name = model_name
        self.class_palette = class_palette
        self.model = None

        if model_name == "UNet":
            ckpt = torch.load(weight_path, map_location=torch.device('cpu'))
            self.model = UNet(in_ch=2, out_ch=6, features=UNET_FEATURES)
            self.model.load_state_dict(ckpt["model_state"])
        elif model_name == "DPT":
            self.model = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=False)

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

        clipped = np.clip(self.mask_array, 0, len(self.class_palette) - 1)
        self.rgb_mask = self.class_palette[clipped]
