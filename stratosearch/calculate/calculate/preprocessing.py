import numpy as np
import torch


def default_preprocess(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float()


def unet_preprocess(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float()[None, None, ...]


PREPROCESSORS = {
    'unet': unet_preprocess,
}
