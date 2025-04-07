# This is a sample Python script.
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from NeuralNetwork import NeuralNetwork, NeuralNetwork_2
from torch.utils.data import Dataset, DataLoader
from CustomDataLoaderCarRacing import CustomDataLoaderCarRacing
from car_racing import CarRacing
import torchvision.transforms
import torchvision.transforms.functional
import numpy as np

LABELS_DICT = {
    0: np.array([-1, 0, 0]),
    1: np.array([1, 0, 0]),
    2: np.array([0, 1, 0]),
    3: np.array([0, 0, 0.8])}

DATASET_SIZE = 5000
BATCH_SIZE = 30
SPLIT_RATIO = (60, 20, 20)
NUM_EPOCHS = 15
CROP_SIZE = 90
IMG_SIZE = 96

def model_without_augmentation(model=1):
    custom_car_racing_train_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=DATASET_SIZE,
                                                                data_split="train", split_ratio= SPLIT_RATIO,
                                                                crop_size=CROP_SIZE,image_size=IMG_SIZE)
    custom_car_racing_train_dataloader = DataLoader(custom_car_racing_train_dataset, shuffle=False, batch_size=BATCH_SIZE)
    custom_car_racing_val_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=DATASET_SIZE,
                                                              data_split="validate", split_ratio=SPLIT_RATIO,
                                                              crop_size=CROP_SIZE,image_size=IMG_SIZE)
    custom_car_racing_val_dataloader = DataLoader(custom_car_racing_val_dataset, shuffle=False, batch_size=BATCH_SIZE)
    custom_car_racing_test_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=DATASET_SIZE,
                                                               data_split="test", split_ratio=SPLIT_RATIO,
                                                               crop_size=CROP_SIZE, image_size=IMG_SIZE)
    custom_car_racing_test_dataloader = DataLoader(custom_car_racing_test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    if model == 1:
        model = NeuralNetwork(IMG_SIZE=IMG_SIZE, NUM_EPOCHS=NUM_EPOCHS)
    elif model == 2:
        model = NeuralNetwork_2(IMG_SIZE=IMG_SIZE, NUM_EPOCHS=NUM_EPOCHS)
    if torch.cuda.is_available():
        model.to("cuda")
    optimizer = optim.Adam(model.parameters())
    loss_fct = nn.CrossEntropyLoss()
    model.assignFurther(train_loader=custom_car_racing_train_dataloader,
                        val_loader=custom_car_racing_val_dataloader,
                        test_loader=custom_car_racing_test_dataloader,
                        optimizer=optimizer, loss_fct=loss_fct)
    model.train_model(model)
    model.test_Model(model)
    model.show_figures(model)
    return model


def model_with_augmentation(model=1):
    custom_car_racing_train_dataset = CustomDataLoaderCarRacing(transforms=True, dataset_size=DATASET_SIZE,
                                                                data_split="train", split_ratio=SPLIT_RATIO,
                                                                crop_size=CROP_SIZE,image_size=IMG_SIZE)
    custom_car_racing_train_dataloader = DataLoader(custom_car_racing_train_dataset, shuffle=False,
                                                    batch_size=BATCH_SIZE)
    custom_car_racing_val_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=DATASET_SIZE,
                                                              data_split="validate", split_ratio=SPLIT_RATIO,
                                                              crop_size=CROP_SIZE,image_size=IMG_SIZE)
    custom_car_racing_val_dataloader = DataLoader(custom_car_racing_val_dataset, shuffle=False, batch_size=BATCH_SIZE)
    custom_car_racing_test_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=DATASET_SIZE,
                                                               data_split="test", split_ratio=SPLIT_RATIO,
                                                               crop_size=CROP_SIZE,image_size=IMG_SIZE)
    custom_car_racing_test_dataloader = DataLoader(custom_car_racing_test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    if model == 1:
        model = NeuralNetwork(IMG_SIZE=IMG_SIZE, NUM_EPOCHS=NUM_EPOCHS)
    elif model == 2:
        model = NeuralNetwork_2(IMG_SIZE=IMG_SIZE, NUM_EPOCHS=NUM_EPOCHS)
    if torch.cuda.is_available():
        model.to("cuda")

    optimizer = optim.Adam(model.parameters())
    loss_fct = nn.CrossEntropyLoss()
    model.assignFurther(train_loader=custom_car_racing_train_dataloader,
                        val_loader=custom_car_racing_val_dataloader,
                        test_loader=custom_car_racing_test_dataloader,
                        optimizer=optimizer, loss_fct=loss_fct)
    model.train_model(model)
    model.test_Model(model)
    model.show_figures(model)
    return model

def model_grayscale(model=1, transform=True):
    custom_car_racing_train_dataset = CustomDataLoaderCarRacing(transforms=transform, dataset_size=DATASET_SIZE,
                                                                data_split="train", split_ratio=SPLIT_RATIO,
                                                                crop_size=CROP_SIZE,image_size=IMG_SIZE,
                                                                input_channels=1)
    custom_car_racing_train_dataloader = DataLoader(custom_car_racing_train_dataset, shuffle=False,
                                                    batch_size=BATCH_SIZE)
    custom_car_racing_val_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=DATASET_SIZE,
                                                              data_split="validate", split_ratio=SPLIT_RATIO,
                                                              crop_size=CROP_SIZE,image_size=IMG_SIZE, input_channels=1)
    custom_car_racing_val_dataloader = DataLoader(custom_car_racing_val_dataset, shuffle=False, batch_size=BATCH_SIZE)
    custom_car_racing_test_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=DATASET_SIZE,
                                                               data_split="test", split_ratio=SPLIT_RATIO,
                                                               crop_size=CROP_SIZE,image_size=IMG_SIZE, input_channels=1)
    custom_car_racing_test_dataloader = DataLoader(custom_car_racing_test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    if model == 1:
        model = NeuralNetwork(IMG_SIZE=IMG_SIZE, NUM_EPOCHS=NUM_EPOCHS, INPUT_CHANNELS=1)
    elif model == 2:
        model = NeuralNetwork_2(IMG_SIZE=IMG_SIZE, NUM_EPOCHS=NUM_EPOCHS,INPUT_CHANNELS=1)
    if torch.cuda.is_available():
        model.to("cuda")
    optimizer = optim.Adam(model.parameters())
    loss_fct = nn.CrossEntropyLoss()
    model.assignFurther(train_loader=custom_car_racing_train_dataloader,
                        val_loader=custom_car_racing_val_dataloader,
                        test_loader=custom_car_racing_test_dataloader,
                        optimizer=optimizer, loss_fct=loss_fct)
    model.train_model(model)
    model.test_Model(model)
    model.show_figures(model)
    return model


def show_figures_augvsnoaug(model1, model2):
    f, axarr = plt.subplots(1, 2)
    f.suptitle("Model loss and accuracy - augment & no augment")
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, model1.num_epochs + 1), model1.average_loss_train, label="Train-Loss no-augment", color="blue")
    plt.plot(np.arange(1, model1.num_epochs + 1), model1.average_loss_val, label="Validation-Loss no-augment",
             color="blue", linestyle="dashed")
    plt.plot(np.arange(1, model2.num_epochs + 1), model2.average_loss_train, label="Train-Loss augment", color="red")
    plt.plot(np.arange(1, model2.num_epochs + 1), model2.average_loss_val, label="Validation-Loss augment",
             color="red", linestyle="dashed")

    plt.title("Loss")
    plt.legend(bbox_to_anchor =(0.5,-0.27), loc='upper center')
    plt.tight_layout()
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, model1.num_epochs + 1), model1.acc_train, label="Train-ACC no-augment", color="blue")
    plt.plot(np.arange(1, model1.num_epochs + 1), model1.acc_val, label="Validation-ACC no-augment", color="blue",
             linestyle="dashed")
    plt.plot(np.arange(1, model2.num_epochs + 1), model2.acc_train, label="Train-ACC augment", color="red")
    plt.plot(np.arange(1, model2.num_epochs + 1), model2.acc_val, label="Validation-ACC augment", color="red",
             linestyle="dashed")
    plt.legend(bbox_to_anchor =(0.5,-0.27), loc='upper center')
    plt.tight_layout()
    plt.title("Accuracy")
    plt.show()

def show_figures_rgbvsgray(model1, model2):
    f, axarr = plt.subplots(1, 2)
    f.suptitle("Model loss and accuracy - rgb & grayscale")
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, model1.num_epochs + 1), model1.average_loss_train, label="Train-Loss rgb",
             color="blue")
    plt.plot(np.arange(1, model1.num_epochs + 1), model1.average_loss_val, label="Validation-Loss rgb",
             color="blue", linestyle="dashed")
    plt.plot(np.arange(1, model2.num_epochs + 1), model2.average_loss_train, label="Train-Loss gray",
             color="red")
    plt.plot(np.arange(1, model2.num_epochs + 1), model2.average_loss_val, label="Validation-Loss gray",
             color="red", linestyle="dashed")

    plt.title("Loss")
    plt.legend(bbox_to_anchor =(0.5,-0.27), loc='upper center')
    plt.tight_layout()
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, model1.num_epochs + 1), model1.acc_train, label="Train-ACC rgb", color="blue")
    plt.plot(np.arange(1, model1.num_epochs + 1), model1.acc_val, label="Validation-ACC rgb", color="blue",
             linestyle="dashed")
    plt.plot(np.arange(1, model2.num_epochs + 1), model2.acc_train, label="Train-ACC gray", color="red")
    plt.plot(np.arange(1, model2.num_epochs + 1), model2.acc_val, label="Validation-ACC gray", color="red",
             linestyle="dashed")
    plt.tight_layout()
    plt.legend(bbox_to_anchor =(0.5,-0.27), loc='upper center')
    plt.tight_layout()
    plt.title("Accuracy")
    plt.show()


print("----------Model without augmentation--------")
model_no_augment = model_without_augmentation(model=2)
print("----------Model with augmentation--------")
model_augment = model_with_augmentation(model=2)
print("----------Model grayscale--------")
model_grayscale_without_augment =model_grayscale(model=2, transform=False)

show_figures_augvsnoaug(model_no_augment, model_augment)
show_figures_rgbvsgray(model_no_augment, model_grayscale_without_augment)
print("----------DONE--------")