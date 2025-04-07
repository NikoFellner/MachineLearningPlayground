# This is a sample Python script.
import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import v2
from CustomDataLoaderCarRacing import CustomDataLoaderCarRacing
import matplotlib.pyplot as plt
from PIL import Image
import random
label_names = {0: "left",
               1: "right",
               2: "accelerate",
               3: "deccelerate"}

def plot_different_methods():
    car_racing_dataset = CustomDataLoaderCarRacing(transforms=False)
    print(f"{car_racing_dataset.__len__()}")
    fig_1, ax_1 = plt.subplots(1,4)
    data, label = car_racing_dataset.__getitem__(500)

    ax_1[0].set_title(f"Original: {label_names[np.argmax(label)]}")
    ax_1[0].axis("off")
    data = torch.Tensor.numpy(torch.permute((data), (1, 2, 0)))
    ax_1[0].imshow(data, interpolation='nearest')

    data_copy = np.copy(data)
    for n in range(96):
        for nn in range(96):
            if np.sum(np.abs(np.subtract(data_copy[n,nn,:], [102,102,102])))<10:
                data_copy[n,nn,:] = [121, 85, 60]

    ax_1[1].set_title(f"First Color change: {label_names[np.argmax(label)]}")
    ax_1[1].axis("off")
    ax_1[1].imshow(data_copy)

    data_copy = np.copy(data).astype(dtype="float32")
    brightness = 0.8
    contrast = 1
    saturation = 1
    hue = -0.5
    t2 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.ToPILImage(),
                                            torchvision.transforms.ColorJitter(brightness=(brightness,brightness),contrast=(contrast,contrast), saturation=(saturation,saturation), hue=(hue,hue)),
                                            torchvision.transforms.ToTensor()])
    transformedImage = torch.Tensor.numpy(torch.permute(t2(data_copy), (1,2,0)))

    ax_1[2].set_title(f"Jitter: {label_names[np.argmax(label)]}")
    ax_1[2].axis("off")
    ax_1[2].imshow(transformedImage)

    data_copy_brown = np.copy(data)
    data_copy_brown[np.where((387 >= np.sum(data_copy_brown[0:84, :, :], axis=2)) & (np.sum(data_copy_brown[0:84, :, :], axis=2) >= 300))]=[155, 102, 60]

    ax_1[3].imshow(data_copy_brown)
    ax_1[3].set_title(f"Mask: {label_names[np.argmax(label)]}")
    ax_1[3].axis("off")
    plt.show()

def plot_images():
    car_racing_dataset_with_transform = CustomDataLoaderCarRacing(street_color=(121, 85, 60))
    car_racing_dataset_with_transform.set_augmentation(vertical=True, horizontal=False,street_color=True)
    idx = random.randint(0,999)
    NUM_ROWS = 5
    NUM_COLS = 6
    fig, ax = plt.subplots(NUM_ROWS,NUM_COLS,layout="constrained", figsize=(3 * NUM_COLS, 3 * NUM_ROWS))
    for rows in range(NUM_ROWS):
        for cols in range(NUM_COLS):
            if rows==0:
                car_racing_dataset_with_transform.set_augmentation(vertical=True, horizontal=False, street_color=False)
                idx = random.randint(0, 999)
            if rows==1:
                car_racing_dataset_with_transform.set_augmentation(vertical=True, horizontal=False, street_color=False)
                idx = random.randint(0, 999)
            elif rows==2:
                car_racing_dataset_with_transform.set_augmentation(vertical=True, horizontal=False, street_color=False)
                idx = random.randint(0, 999)
            elif rows==3:
                car_racing_dataset_with_transform.set_augmentation(vertical=True, horizontal=False, street_color=True)
                idx = random.randint(0, 999)
            elif rows==4:
                car_racing_dataset_with_transform.set_augmentation(vertical=True, horizontal=False, street_color=True)
                idx = random.randint(0, 999)

            data, label = car_racing_dataset_with_transform.__getitem__(idx)
            ax[rows,cols].set_title(f"{label_names[np.argmax(label)]}", fontsize=30)
            ax[rows,cols].axis("off")
            data = torch.Tensor.numpy(torch.permute((data), (1, 2, 0)))
            ax[rows,cols].imshow(data)
    fig.tight_layout(pad=3)
    plt.show()
    fig.savefig("carImagesWithAugmentation")

plot_different_methods()
plot_images()