import os

import torch
from tqdm import tqdm

import useful_funcs as fuc
import data_preprocess as dp


def train_loop(model: torch.nn.Module, criterion, optimizer, dl: torch.utils.data.DataLoader, threshold=0.5):
    model.train()
    running_loss = 0.0
    epoch_confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for data in tqdm(dl):
        inputs = data[0].to(dp.DEVICE)
        labels = data[1].to(dp.DEVICE)

        logits:torch.Tensor = model(inputs)["out"]
        logits = logits.view_as(labels)

        # if i < 5 or i > (len(dl) - 5):
        #     # print("Input:", inputs.shape, inputs[0].mean(), inputs[0].std())
        #     # print("Label:", labels.shape, labels[0].mean(), labels[0].std())
        #     # print("Logit:", logits.shape, logits[0].mean(), logits[0].std())
        #     plot = fuc.show_tensors({"Input": inputs[0], "Label": labels[0], "Logit": logits[0], "Prediction": (logits[0] > 0.5).float()})
        #     plot.savefig(os.path.join("plots", f"train_plot_{i}.png"))
        #     # plot.clear()

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        conf = fuc.compute_confusion(logits, labels, threshold)
        for key in epoch_confusion:
            epoch_confusion[key] += conf[key]
    running_loss /= len(dl)
    return running_loss, epoch_confusion

def val_loop(model: torch.nn.Module, criterion, dl: torch.utils.data.DataLoader, threshold=0.5):
    model.eval()
    running_loss = 0.0
    epoch_confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    with torch.no_grad():
        for data in tqdm(dl):
            inputs = data[0].to(dp.DEVICE)
            labels = data[1].to(dp.DEVICE)

            logits:torch.Tensor = model(inputs)["out"]
            logits = logits.view_as(labels)

            # if i < 5 or i > (len(dl) - 5):
            #     # print("Input:", inputs.shape, inputs[0].mean(), inputs[0].std())
            #     # print("Label:", labels.shape, labels[0].mean(), labels[0].std())
            #     # print("Logit:", logits.shape, logits[0].mean(), logits[0].std())
            #     plot = fuc.show_tensors({"Input": inputs[0], "Label": labels[0], "Logit": logits[0], "Prediction": (logits[0] > 0.5).float()})
            #     plot.savefig(os.path.join("plots", f"val_plot_{i}.png"))
            #     # plot.clear()

            loss = criterion(logits, labels)

            running_loss += loss.item()
            
            conf = fuc.compute_confusion(logits, labels, threshold)
            # print(conf)
            for key in epoch_confusion:
                epoch_confusion[key] += conf[key]
    running_loss /= len(dl)
    return running_loss, epoch_confusion


def test_loop(model: torch.nn.Module, criterion, dl: torch.utils.data.DataLoader, threshold=0.5):
    model.eval()
    running_loss = 0.0
    epoch_confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    dir = os.path.join("test_plots_sigm", f"plots_{threshold}")
    os.makedirs(dir, exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(tqdm(dl)):
            inputs = data[0].to(dp.DEVICE)
            labels = data[1].to(dp.DEVICE)

            logits:torch.Tensor = model(inputs)["out"]
            logits = logits.view_as(labels)

            if i in [0, len(dl) - 1]:
                print(logits.mean(), logits.std())
                sigmified = torch.nn.functional.sigmoid(logits[0])
                print(sigmified.mean(), sigmified.std())
                plot = fuc.show_tensors({"Input": inputs[0], "Label": labels[0], 
                                         "Logit": sigmified, 
                                         "Prediction": (sigmified > threshold).float()})
                plot.savefig(os.path.join(dir, f"plot_{i}.png"))
                plot.clear()

            loss = criterion(logits, labels)

            running_loss += loss.item()
            
            conf = fuc.compute_confusion(logits, labels, threshold)
            for key in epoch_confusion:
                epoch_confusion[key] += conf[key]
    running_loss /= len(dl)
    return running_loss, epoch_confusion
