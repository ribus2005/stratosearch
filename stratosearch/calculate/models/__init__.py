from .unet import UNet

MODELS = {
    "unet": UNet,
}

__all__ = [
    "MODELS"
]