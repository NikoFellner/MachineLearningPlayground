from torch.utils.data import Dataset, DataLoader
import numpy as np
import os.path
import pygame
from car_racing import CarRacing
import torchvision.transforms
import torchvision.transforms.functional
import random
import torch
import matplotlib.pyplot as plt



class CustomDataLoaderCarRacing(Dataset):

    def __init__(self, street_color=(155, 102, 60), transforms=True, dataset_size=1000, data_split="train",
                 split_ratio=(60, 20, 20), crop_size=72, image_size=96, input_channels=3):
        super(CustomDataLoaderCarRacing, self).__init__()
        self.split = data_split
        self.street_color = street_color

        if not os.path.isfile("data/car_Racing_Actions.npz") & os.path.isfile("data/car_Racing_Images.npz"):
            self.create_data(img_to_save=dataset_size, filename_img="data/car_Racing_Images.npz",
                             filename_action="data/car_Racing_Actions.npz")
        if data_split == "train":
            dict_data = np.load("data/car_Racing_Images.npz")
            end_data_point = int(dataset_size * split_ratio[0]/100)
            self.data = dict_data.f.arr_0[:end_data_point,:,:,:]
            dict_data = np.load("data/car_Racing_Actions.npz")
            self.label = dict_data.f.arr_0[:end_data_point,:]
        elif data_split == "validate":
            start_data_point = int(dataset_size * split_ratio[0]/100)
            end_data_point = int(start_data_point + dataset_size * split_ratio[1]/100)
            dict_data = np.load("data/car_Racing_Images.npz")
            self.data = dict_data.f.arr_0[start_data_point:end_data_point,:,:,:]
            dict_data = np.load("data/car_Racing_Actions.npz")
            self.label = dict_data.f.arr_0[start_data_point:end_data_point,:]
        elif data_split == "test":
            start_data_point = int(dataset_size * split_ratio[0]/100 + dataset_size * split_ratio[1]/100)
            end_data_point = int(start_data_point + dataset_size * split_ratio[2]/100)
            dict_data = np.load("data/car_Racing_Images.npz")
            self.data = dict_data.f.arr_0[start_data_point:end_data_point, :, :, :]
            dict_data = np.load("data/car_Racing_Actions.npz")
            self.label = dict_data.f.arr_0[start_data_point:end_data_point, :]

        if self.__len__()<dataset_size:
            if not os.path.isfile("data/car_Racing_Actions.npz") & os.path.isfile("data/car_Racing_Images.npz"):
                self.create_data(img_to_save=dataset_size, filename_img="data/car_Racing_Images.npz",
                                 filename_action="data/car_Racing_Actions.npz")


        self.transforms = transforms
        self.input_channels = input_channels
        self.transform_hor = torchvision.transforms.RandomHorizontalFlip(p=1)
        self.transform_tensor = torchvision.transforms.ToTensor()
        self.transform_crop_rotation = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=(crop_size,crop_size)),
                                                                       torchvision.transforms.Resize(size=(image_size,image_size)),
                                                                       torchvision.transforms.RandomRotation(degrees=(-10, 10))])



    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        if (self.split == "train") | (self.split == "validate"):
            if self.transforms:
                """horizontal flipping, random crop and rotation (up to 20 degrees)"""
                horizontal_augment = random.randint(1, 2)
                label = self.label[idx]
                #data = torchvision.transforms.functional.resized_crop(img=self.transform_tensor(self.data[idx]),
                #                                                      top=0, left=0, height=80, width=96, antialias=True,
                #                                                      size=(96, 96))
                data = self.transform_tensor(self.data[idx])
                data = self.transform_crop_rotation(data)
                if (horizontal_augment == 1):
                    data = self.transform_hor(data)
                    if label[0] == 1:
                        label += [-1, 1, 0, 0]
                    elif label[1] == 1:
                        label += [1, -1, 0, 0]
                if self.input_channels == 1:
                    data = torchvision.transforms.functional.rgb_to_grayscale(data)
                return data, torch.argmax(torch.from_numpy(label))
            else:
                #data = torchvision.transforms.functional.resized_crop(img=self.transform_tensor(self.data[idx]), top=0, left=0,
                #                                               height=80, width=96,antialias=True, size=(96,96))

                data = self.transform_tensor(self.data[idx])
                if self.input_channels == 1:
                    data = torchvision.transforms.functional.rgb_to_grayscale(data)
                return data, torch.argmax(torch.from_numpy(self.label[idx]))
        elif self.split == "test":
            flip_left_right = torchvision.transforms.RandomHorizontalFlip(p=1)
            if self.transforms == "streetcolor":
                #print("change_streetcolor")
                img = self.transform_tensor(self.street_color_filter(self.data[idx]))
                return img, torch.argmax(torch.from_numpy(self.label[idx]))
            elif self.transforms == "fliplr":
                #print("flip left right")
                label = self.label[idx]
                if label[0] == 1:
                    label += [-1, 1, 0, 0]
                elif label[1] == 1:
                    label += [1, -1, 0, 0]
                img = self.transform_tensor(self.data[idx])
                img = flip_left_right(img)
                return img, torch.argmax(torch.from_numpy(label))
            elif self.transforms == "flipud":
                #print("flip upside down")
                label = self.label[idx]
                if label[0] == 1:
                    label += [-1, 1, 0, 0]
                elif label[1] == 1:
                    label += [1, -1, 0, 0]
                img = self.transform_tensor(self.data[idx])
                img = torchvision.transforms.functional.rotate(img, 180)
                img = flip_left_right(img)
                return img, torch.argmax(torch.from_numpy(label))
            else:
                #print("return normal image")
                return self.transform_tensor(self.data[idx]), torch.argmax(torch.from_numpy(self.label[idx]))



    def street_color_filter(self, ndarray_image):
        ndarray_image[np.where((387 >= np.sum(ndarray_image[0:84, :, :], axis=2)) & (
                 np.sum(ndarray_image[0:84, :, :], axis=2) >= 300))] = self.street_color
        return ndarray_image

    def create_data(self, img_to_save, filename_img, filename_action):
        env = CarRacing(render_mode="human")
        action = np.array([0.0, 0.0, 0.0])
        action_separated_leftright = np.array([0.0, 0.0, 0.0, 0.0])

        def register_input():
            global quit, restart
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action[0] = -1.0
                        action_separated_leftright[0] = 1.0
                    if event.key == pygame.K_RIGHT:
                        action[0] = +1.0
                        action_separated_leftright[1] = 1.0
                    if event.key == pygame.K_UP:
                        action[1] = +1.0
                        action_separated_leftright[2] = 1.0
                    if event.key == pygame.K_DOWN:
                        action[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                        action_separated_leftright[3] = 1.0
                    if event.key == pygame.K_RETURN:
                        restart = True
                    if event.key == pygame.K_ESCAPE:
                        quit = True

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        action[0] = 0
                        action_separated_leftright[0] = 0.0
                    if event.key == pygame.K_RIGHT:
                        action[0] = 0
                        action_separated_leftright[1] = 0.0
                    if event.key == pygame.K_UP:
                        action[1] = 0
                        action_separated_leftright[2] = 0.0
                    if event.key == pygame.K_DOWN:
                        action[2] = 0
                        action_separated_leftright[3] = 0.0

                if event.type == pygame.QUIT:
                    quit = True


        NUM_IMAGES_TO_SAVE = img_to_save
        images_to_save = np.random.randint(0, 255, size=(NUM_IMAGES_TO_SAVE, 96, 96, 3))
        actions_to_save = np.random.randint(0, 255, size=(NUM_IMAGES_TO_SAVE, 4))
        #actions_to_save = np.random.randint(0, 255, size=(NUM_IMAGES_TO_SAVE, 1))
        data_saved = 0
        quit = False
        while not quit:
            env.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                register_input()
                state, reward, terminated, truncated, info = env.step(action)
                if any(action != 0):
                    images_to_save[data_saved] = state
                    actions_to_save[data_saved] = action_separated_leftright
                    data_saved += 1
                    if data_saved == NUM_IMAGES_TO_SAVE:
                        np.savez_compressed(filename_img, images_to_save)
                        np.savez_compressed(filename_action, actions_to_save)
                        quit = True
                total_reward += reward
                if steps % 200 == 0 or terminated or truncated:
                    print("\naction " + str([f"{x:+0.2f}" for x in action]))
                    print(f"step {steps} total_reward {total_reward:+0.2f}")
                steps += 1
                if terminated or truncated or restart or quit:
                    break
        env.close()
