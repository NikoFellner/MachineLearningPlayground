from CustomDataLoaderCarRacing import CustomDataLoaderCarRacing
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from VideoCreatorTestCases import VideoCreatorTestCases
from NeuralNetwork import NeuralNetwork_2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

DATASET_SIZE = 5000
BATCH_SIZE = 10
SPLIT_RATIO = (60, 20, 20)
NUM_EPOCHS = 15
CROP_SIZE = 90
IMG_SIZE = 96

def create_the_videos():
    vid_creator = VideoCreatorTestCases()
    #vid_creator.create_video(videoname="flipud.avi")
    vid_creator.create_video(videoname="flipud.mp4")

    vid_creator.create_dataloader_for_case(test_case="fliplr")
    vid_creator.create_video(videoname="fliplr.mp4")

    vid_creator.create_dataloader_for_case(test_case="streetcolor")
    vid_creator.create_video(videoname="streetcolor.mp4")

#create_the_videos()

model = NeuralNetwork_2(IMG_SIZE=96, NUM_EPOCHS=NUM_EPOCHS)
if torch.cuda.is_available():
    model.to("cuda")

optimizer = optim.Adam(model.parameters())
loss_fct = nn.CrossEntropyLoss()

custom_car_racing_train_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=DATASET_SIZE,
                                                                data_split="train", split_ratio= SPLIT_RATIO,
                                                                crop_size=CROP_SIZE,image_size=IMG_SIZE)

custom_car_racing_train_dataloader = DataLoader(custom_car_racing_train_dataset, shuffle=True, batch_size=BATCH_SIZE)

custom_car_racing_val_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=DATASET_SIZE,
                                                                data_split="validate", split_ratio= SPLIT_RATIO,
                                                                crop_size=CROP_SIZE,image_size=IMG_SIZE)

custom_car_racing_val_dataloader = DataLoader(custom_car_racing_val_dataset, shuffle=False, batch_size=BATCH_SIZE)

custom_car_racing_test_dataset_fliplr = CustomDataLoaderCarRacing(street_color=(155,102,60), transforms="fliplr",
                                                    dataset_size=1000, data_split="test",
                                               split_ratio=(60, 20, 20), crop_size=72, image_size=96, input_channels=3)

custom_car_racing_test_dataloader_fliplr = DataLoader(custom_car_racing_test_dataset_fliplr, shuffle=False,
                                                      batch_size=BATCH_SIZE)

custom_car_racing_test_dataset_flipud = CustomDataLoaderCarRacing(street_color=(155,102,60), transforms="flipud",
                                                    dataset_size=1000, data_split="test",
                                               split_ratio=(60, 20, 20), crop_size=72, image_size=96, input_channels=3)

custom_car_racing_test_dataloader_flipud = DataLoader(custom_car_racing_test_dataset_flipud, shuffle=False,
                                                      batch_size=BATCH_SIZE)

custom_car_racing_test_dataset_streetcolor = CustomDataLoaderCarRacing(street_color=(155,102,60), transforms="streetcolor",
                                                    dataset_size=1000, data_split="test",
                                               split_ratio=(60, 20, 20), crop_size=72, image_size=96, input_channels=3)

custom_car_racing_test_dataloader_streetcolor = DataLoader(custom_car_racing_test_dataset_streetcolor, shuffle=False,
                                                      batch_size=BATCH_SIZE)

model.assignFurther(train_loader=custom_car_racing_train_dataloader,
                        val_loader=custom_car_racing_val_dataloader,
                        test_loader=custom_car_racing_test_dataloader_fliplr,
                        optimizer=optimizer, loss_fct=loss_fct)

model.train_model(model)

#test fliplr
model.test_Model(model)
fliplr_test_loss = model.average_loss_test
fliplr_test_acc = model.acc_test
#test flipud
model.testLoader = custom_car_racing_test_dataloader_flipud
model.test_Model(model)
flipud_test_loss = model.average_loss_test
flipud_test_acc = model.acc_test

#test streetcolor
model.testLoader = custom_car_racing_test_dataloader_streetcolor
model.test_Model(model)
streetcolor_test_loss = model.average_loss_test
streetcolor_test_acc = model.acc_test

model.show_figures_task1(model,fliplr_test_loss, fliplr_test_acc,flipud_test_loss,flipud_test_acc,streetcolor_test_loss,streetcolor_test_acc)