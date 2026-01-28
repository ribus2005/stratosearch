import os

import torch
import torchvision
from torch.utils.data import DataLoader

import data_preprocess as dp
import useful_funcs as fuc
import loops as ls


PATH = os.path.join("saves", "save.pth")
BS = 8
EPOCHES = 30
LR = 1e-3
LRS = [1e-4, 1e-5, 1e-5, 1e-6]
# THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
# THRESHOLDS = [0.7, 0.9]
THRESHOLDS = [0.5,]
torch.set_default_device(dp.DEVICE)

CRITER = fuc.BCEDICELoss()


def training(model, train_dl, val_dl, criterion, start_lr, epoches=10, threshold=0.5):
    result = {"Train loss": [], "Val loss": [], "Rec": [], "Prec": [], "F1": [], "IoU": []}
    maxiou = 0
    curr_lr = start_lr
    saves_dir = f"saves_{threshold}"
    plots_dir = f"plots_{threshold}"
    os.makedirs(saves_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr)
    for epoch in range(epoches):
        

        print("Epoch:", epoch)
        # print("LR:", curr_lr)
        # print("Threshold:", threshold)
        # if curr_lr > 5e-9:
        #     curr_lr *= 0.8

        tr_loss, train_confs = ls.train_loop(model, criterion, optimizer, train_dl, threshold)
        result["Train loss"].append(tr_loss)
        print(f"\tTrain loss: {tr_loss:.4f}")

        val_loss, val_confs = ls.val_loop(model, criterion, val_dl, threshold)
        result["Val loss"].append(val_loss)
        print(f"Val loss: {val_loss:.4f}", end="")
        
        val_mets = fuc.calculate_metrics(val_confs)
        for key in val_mets:
            result[key].append(val_mets[key])
            print(f", {key}: {val_mets[key]:.4f}", end="")
        print()
        
        if val_mets["IoU"] > maxiou:
            if maxiou != 0:
                os.system("rm -f " + os.path.join(saves_dir, f"maxIoU_{(maxiou * 100):.0f}.pth"))
            maxiou = val_mets["IoU"]
            torch.save(model.state_dict(), os.path.join(saves_dir, f"maxIoU_{(maxiou * 100):.0f}.pth"))

        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join(saves_dir, f"epoch_{epoch}.pth"))
            try:
                plot = fuc.metrics_plot(result)
                plot.savefig(os.path.join(plots_dir, f"MetricsDeepLabv3_{epoch}Epoche.png"))
            except Exception as e:
                print(f"Caught exception during plotting: {e}")

    return result, maxiou

if __name__ == "__main__":
    if dp.DEVICE != torch.device("cpu"):
        torch.cuda.empty_cache()

    train_ds = dp.RoadDataset(dp.x_train_dir, dp.y_train_dir,
                              transform=None, bs=BS)
    val_ds = dp.RoadDataset(dp.x_valid_dir, dp.y_valid_dir,
                            transform=None, bs=1)
    test_ds = dp.RoadDataset(dp.x_test_dir, dp.y_test_dir,
                             transform=None, bs=1)
    train_dl = DataLoader(train_ds, BS, shuffle=True, generator=torch.Generator(device=dp.DEVICE))
    val_dl = DataLoader(val_ds, 1, generator=torch.Generator(device=dp.DEVICE))
    test_dl = DataLoader(test_ds, 1, generator=torch.Generator(device=dp.DEVICE))

    # model = torchvision.models.segmentation.deeplabv3_resnet50(
    #     weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
    #     weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT
    # )
    # model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    # model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    # model = torch.nn.DataParallel(model)
    # model.to(dp.DEVICE)
    ious = []

    # is_load = int(input("Load model? (0/1) "))
    is_load = 0
    if is_load:
        # model.load_state_dict(torch.load(os.path.join("saves", "maxIoU_64.pth")))
        # test_loss, test_conf = ls.val_loop(model, CRITER, test_dl)
        # test_metrics = fuc.calculate_metrics(test_conf)
        # print("Test loss:", test_loss)
        # for key in test_metrics:
        #     print(f"{key}: {test_metrics[key]}")    
        pass
    else:
        for threshold in THRESHOLDS:
            # model = torchvision.models.segmentation.deeplabv3_resnet50(
            #     weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
            #     weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT
            # )
            model = torchvision.models.segmentation.deeplabv3_resnet50()
            model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
            model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
            # model = torch.nn.DataParallel(model)
            model.to(dp.DEVICE)

            print("Threshold:", threshold)
            train_res, maxiou = training(model, train_dl, val_dl, 
                                         criterion=CRITER, start_lr=LR, 
                                         epoches=EPOCHES, threshold=threshold)
            ious.append(maxiou)
            torch.save(model.state_dict(), os.path.join(f"saves_{threshold}", "save.pth"))
            try:
                plot = fuc.metrics_plot(train_res)
                plot.savefig(os.path.join(f"plots_{threshold}", f"MetricsDeepLabv3_{EPOCHES}Epoche.png"))
            except Exception as e:
                print(f"Caught exception during plotting: {e}")
            # plot.show()
            print("\n\n")

    print()
    print(list(zip(THRESHOLDS, ious)))

    torch.cuda.empty_cache()
