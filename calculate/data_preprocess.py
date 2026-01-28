import os
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(DEVICE)
DATA_DIR = "../yolo_dataset"
TEST = False

x_train_dir = os.path.join(DATA_DIR, "images", "train")
y_train_dir = os.path.join(DATA_DIR, "labels", "train")

x_valid_dir = os.path.join(DATA_DIR, "images", "val")
y_valid_dir = os.path.join(DATA_DIR, "labels", "val")

x_test_dir = os.path.join(DATA_DIR, "images", "test")
y_test_dir = os.path.join(DATA_DIR, "labels", "test")

# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(18,6))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

TRAIN_CROP_SIZE = 1024

# TRAIN_TRANSFORM = album.Compose([
#         # album.RandomCrop(height=TRAIN_CROP_SIZE, width=TRAIN_CROP_SIZE, p = 1),
#         album.PadIfNeeded(min_height=TRAIN_CROP_SIZE, min_width=TRAIN_CROP_SIZE, border_mode=cv2.BORDER_CONSTANT),
#         album.ToTensorV2()
#     ]
# )
# VALID_TRANSFORM = album.Compose([
#     album.PadIfNeeded(min_height=TRAIN_CROP_SIZE, min_width=TRAIN_CROP_SIZE, border_mode=cv2.BORDER_CONSTANT),
#     album.ToTensorV2()
# ])

class RoadDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            transform=None,
            bs=8
    ):
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        length = len(self.image_paths)
        if length % bs != 0:
            length = (length // bs) * bs
        self.image_paths = self.image_paths[:length]
        self.mask_paths = self.mask_paths[:length]

        if TEST and bs != 1:
            length = int((length / bs) * 0.05) * bs
            self.image_paths = self.image_paths[:length]
            self.mask_paths = self.mask_paths[:length]

        self.transform = transform
        # print(self.image_paths[0])
        # print(self.mask_paths[0])

    def __getitem__(self, i):
        # print(self.image_paths[i])
        # print(self.mask_paths[i])
        image = cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
        # print(image.shape, mask.shape)
        # print("Image before:", image.mean(), image.std())
        # print("Mask before:", mask.mean(), mask.std())

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image'].float()
            mask = augmented['mask'].float()
        else:
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
        # print("Image after:", image.mean(), image.std())
        # print("Mask after:", mask.mean(), mask.std())
        # print(image.shape, mask.shape)
        return image.unsqueeze(0), mask.unsqueeze(0) 

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    dataset = RoadDataset(x_train_dir, y_train_dir)
    random_idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[random_idx]
    print(image.shape, mask.shape)

    visualize(
        original_image = torchvision.transforms.ToPILImage()(image),
        mask = mask.numpy(force=True).astype(np.int32)
    )
