import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import random


LABELS_DICT = {
    0: "Left",
    1: "Right",
    2: "Acc.",
    3: "Decc."}

def plot_different_methods(data_loader):
    car_racing_dataset = data_loader(transforms=False)
    print(f"{car_racing_dataset.__len__()}")
    fig_1, ax_1 = plt.subplots(1,4)
    data, label = car_racing_dataset.__getitem__(500)

    ax_1[0].set_title(f"Original: {label}")
    ax_1[0].axis("off")
    data = torch.Tensor.numpy(torch.permute((data), (1, 2, 0)))
    ax_1[0].imshow(data, interpolation='nearest')

    data_copy = np.copy(data)
    for n in range(96):
        for nn in range(96):
            if np.sum(np.abs(np.subtract(data_copy[n,nn,:], [102,102,102])))<10:
                data_copy[n,nn,:] = [121, 85, 60]

    ax_1[1].set_title(f"First Color change: {label}")
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

    ax_1[2].set_title(f"Jitter: {label}")
    ax_1[2].axis("off")
    ax_1[2].imshow(transformedImage)

    data_copy_brown = np.copy(data)
    data_copy_brown[np.where((387 >= np.sum(data_copy_brown[0:84, :, :], axis=2)) & (np.sum(data_copy_brown[0:84, :, :], axis=2) >= 300))]=[155, 102, 60]

    ax_1[3].imshow(data_copy_brown)
    ax_1[3].set_title(f"Mask: {label}")
    ax_1[3].axis("off")
    plt.show()

def plot_images(samples,label, num_figures,num_rows, num_columns):
    i = 0
    for figs in range(num_figures):
        fig, ax = plt.subplots(num_rows,num_columns,layout="constrained", figsize=(3 * num_columns, 3 * num_rows))
        for rows in range(num_rows):
            for cols in range(num_columns):
                ax[rows,cols].set_title(f"{LABELS_DICT[label[i].item()]}", fontsize=30)
                ax[rows,cols].axis("off")
                data = torch.Tensor.numpy(torch.permute((samples[i]), (1, 2, 0)))
                ax[rows,cols].imshow(data)
                i+=1

        fig.tight_layout(pad=3)
        plt.show()


