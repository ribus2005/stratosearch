import numpy as np
import torch
import cv2


def default_postprocess(output: torch.Tensor) -> np.ndarray:
    """
    output: (C, H, W) или (1, C, H, W)
    """
    if output.ndim == 4:
        output = output.squeeze()
    output_arr = output.numpy()
    class_map = np.argmax(output_arr, axis=0)  # (H,W)
    return class_map.astype(np.uint8)


def DPT_postprocess(output: torch.Tensor, shape = (701, 255)) -> np.ndarray:
    pred = output.predicted_depth.argmax(dim=1).squeeze()

    pred_np = pred.detach().cpu().numpy()

    resized = cv2.resize(pred_np, shape, interpolation=cv2.INTER_NEAREST)

    return resized


POSTPROCESSORS = {
    'DPT': DPT_postprocess
}
