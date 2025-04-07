from torch.utils.data import Dataset, DataLoader
import numpy as np
import os.path
import pygame
from car_racing import CarRacing
import torchvision.transforms
import random


class CustomDataLoaderCarRacing(Dataset):

    def __init__(self, street_color=(155, 102, 60), transforms=True, dataset_size=1000):
        super(CustomDataLoaderCarRacing, self).__init__()
        if not os.path.isfile("car_Racing_Actions.npz") & os.path.isfile("car_Racing_Images.npz"):
            self.create_data(img_to_save=dataset_size, filename_img="car_Racing_Images.npz",
                             filename_action="car_Racing_Actions.npz")
        dict = np.load("car_Racing_Images.npz")
        self.data = dict.f.arr_0
        dict = np.load("car_Racing_Actions.npz")
        self.label = dict.f.arr_0
        if self.__len__()<dataset_size:
            if not os.path.isfile("car_Racing_Actions.npz") & os.path.isfile("car_Racing_Images.npz"):
                self.create_data(img_to_save=dataset_size, filename_img="car_Racing_Images.npz",
                                 filename_action="car_Racing_Actions.npz")
        self.street_color = street_color
        self.set_augmentation()
        self.transforms = transforms
        self.transform_vert = torchvision.transforms.RandomVerticalFlip(p=1)
        self.transform_hor = torchvision.transforms.RandomHorizontalFlip(p=1)
        self.transform_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.transforms:
            street_color = random.randint(1, 2)
            horizontal_augment = random.randint(1, 2)
            vertical_augment = random.randint(1, 2)
            data = self.data[idx]
            label = self.label[idx]
            if (street_color == 1) & self.augment_streetcolor:
                data = self.street_color_filter(data)

            data = self.transform_tensor(data)
            if (horizontal_augment == 1) & self.augment_horizontal:
                data = self.transform_hor(data)
                if label[0] == 1:
                    label += [-1, 1, 0, 0]
                elif label[1] == 1:
                    label += [1, -1, 0, 0]

            if (vertical_augment == 1) & self.augment_vertical:
                data = self.transform_vert(data)
            return data, label
        else:
            return self.transform_tensor(self.data[idx]), self.label[idx]

    def street_color_filter(self, ndarray_image):
        ndarray_image[np.where((387 >= np.sum(ndarray_image[0:84, :, :], axis=2)) & (
                 np.sum(ndarray_image[0:84, :, :], axis=2) >= 300))] = self.street_color
        #ndarray_image[np.where(np.sum(ndarray_image[0:84, :, :], axis=2)%3==0)] = self.street_color

        return ndarray_image

    def set_augmentation(self, vertical=True, horizontal=True, street_color=True):
        self.augment_vertical = vertical
        self.augment_horizontal = horizontal
        self.augment_streetcolor = street_color
        if (self.augment_vertical == False) & (self.augment_horizontal == False) & (self.augment_streetcolor == False):
            self.transforms = False
        else:
            self.transforms = True



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
        # actions_to_save = np.random.randint(0, 255, size=(NUM_IMAGES_TO_SAVE, 1))
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
                    if np.sum(action_separated_leftright) == 1:
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
