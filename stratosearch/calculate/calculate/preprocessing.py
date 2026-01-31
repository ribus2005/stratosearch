import numpy as np
import torch
import torchvision.transforms as transforms


def default_preprocess(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float()


def unet_preprocess(arr: np.ndarray, add_grad_channel=True) -> torch.Tensor:
    x = arr.astype(np.float32)

    if add_grad_channel:
        gy, gx = np.gradient(x)
        grad = np.sqrt(gx*gx + gy*gy)
        if grad.max() > 0:
            grad /= (grad.max() + 1e-8)
        x = np.stack([x, grad], axis=0)  # (C,H,W)
    else:
        x = x[np.newaxis, ...]  # (1,H,W)

    for ci in range(x.shape[0]):
        ch = x[ci]
        mean = ch.mean()
        std = ch.std()
        if std > 1e-6:
            x[ci] = (ch - mean) / (std + 1e-8)
        else:
            x[ci] = ch - mean

    x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1,C,H,W)
    return x_tensor


def DPT_preprocess(img: np.ndarray) -> torch.Tensor:
    img_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                ])
    input = img_transform(img)

    return input.unsqueeze(0)


PREPROCESSORS = {
    'UNet': unet_preprocess,
    'DPT': DPT_preprocess
}
