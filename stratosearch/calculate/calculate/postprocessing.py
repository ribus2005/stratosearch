import numpy as np
import torch


def default_postprocess(output: torch.Tensor) -> np.ndarray:
    """
    output: (C, H, W) или (1, C, H, W)
    """
    if output.ndim == 4:
        output = output.squeeze()
    output_arr = output.numpy()
    class_map = np.argmax(output_arr, axis=0)  # (H,W)
    return class_map.astype(np.uint8)


POSTPROCESSORS = {
    "DINO": default_postprocess,
}
