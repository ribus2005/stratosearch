import numpy as np


def multiclass_postprocess(output):
    """
    output: (C, H, W) или (1, C, H, W)
    """
    if output.ndim == 4:
        output = output[0]
    class_map = np.argmax(output, axis=0)  # (H,W)
    return class_map.astype(np.uint8)


def default_postprocess(output):
    """
    output: (C, H, W) или (1, C, H, W)
    """
    if output.ndim == 4:
        output = output[0]
    class_map = np.argmax(output, axis=0)  # (H,W)
    return class_map.astype(np.uint8)


POSTPROCESSORS = {
    "unet": multiclass_postprocess,
}
