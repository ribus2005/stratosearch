from math import ceil
from typing import Dict

import torch
import torchvision
import matplotlib.pyplot as plt

import data_preprocess as dp


torch.set_default_device(dp.DEVICE)

def compute_confusion(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, int]:
    """
    Вычисляет TP, FP, TN, FN для бинарной маски
    """
    logits = torch.nn.functional.sigmoid(logits)
    logits = (logits > threshold).float()
    targets = targets.float()

    tp = ((logits == 1) & (targets == 1)).sum().item()
    fp = ((logits == 1) & (targets == 0)).sum().item()
    tn = ((logits == 0) & (targets == 0)).sum().item()
    fn = ((logits == 0) & (targets == 1)).sum().item()

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

def calculate_metrics(metrics: Dict):
    tp = metrics["TP"]
    fp = metrics["FP"]
    # tn = metrics["TN"]
    fn = metrics["FN"]

    recall = tp / (tp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    f1 = 2 * recall * precision / (recall + precision + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return {"Rec": recall, "Prec": precision, "F1": f1, "IoU": iou}

class BCEDICELoss(torch.nn.Module):
    '''
    My DICE-BCE loss
    '''
    def __init__(self):
        super(BCEDICELoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, logits, target):
        bce_res = self.bce(logits, target)
        preds = torch.nn.functional.sigmoid(logits)
        numen = (preds * target).sum()
        denom = preds.sum() + target.sum()
        dice_res = 2 * numen / (denom + 1e-7)
        return (bce_res + 1 - dice_res) / 2


def metrics_plot(metrics: Dict):
    length = len(metrics)
    ncols = 2
    nrows = ceil(length / 2)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 16), constrained_layout=True)
    for i, (name, val) in enumerate(metrics.items()):
        ax[i // ncols, i % ncols].plot(val)
        ax[i // ncols, i % ncols].set_title(name)
    fig.suptitle("Metrics")
    return fig


def show_tensors(images: Dict):
    length = len(images)
    ncols = 2
    nrows = ceil(length / 2)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 16), constrained_layout=True)
    to_np = torchvision.transforms.ToPILImage()
    for i, (name, val) in enumerate(images.items()):
        ax[i // ncols, i % ncols].imshow(to_np(val.to("cpu")))
        ax[i // ncols, i % ncols].set_title(name)
    fig.suptitle("Images")
    return fig
