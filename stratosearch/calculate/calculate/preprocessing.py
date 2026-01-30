import numpy as np
import torch
import torchvision.transforms as transforms


def default_preprocess(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float()


def unet_preprocess(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float()[None, None, ...]


def DPT_preprocess(img: np.ndarray) -> torch.Tensor:
    img_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                ])
    input = img_transform(img)

    return input.unsqueeze(0)


PREPROCESSORS = {
    'unet': unet_preprocess,
    'DPT': DPT_preprocess
}
