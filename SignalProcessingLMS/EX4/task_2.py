import torch.cuda
from torch.utils.data import DataLoader
from CustomDataLoaderCarRacing import CustomDataLoaderCarRacing
from NeuralNetwork import NeuralNetwork_2, NeuralNetwork
import torch.optim as optim
import torch.nn as nn
from car_racing import CarRacing
import numpy as np
import torchvision.transforms
import torchvision.transforms.functional

DATASET_SIZE = 5000
BATCH_SIZE = 32
SPLIT_RATIO = (60, 20, 20)
NUM_EPOCHS = 10
CROP_SIZE = 90
IMG_SIZE = 96
TOTENSOR = torchvision.transforms.ToTensor()

model = NeuralNetwork_2(IMG_SIZE=96, NUM_EPOCHS=NUM_EPOCHS)

if torch.cuda.is_available():
    model.to("cuda")

optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_crit = nn.CrossEntropyLoss()

custom_car_racing_train_dataset = CustomDataLoaderCarRacing(transforms=True, dataset_size=DATASET_SIZE,
                                                                data_split="train", split_ratio= SPLIT_RATIO,
                                                                crop_size=CROP_SIZE,image_size=IMG_SIZE)

custom_car_racing_train_dataloader = DataLoader(custom_car_racing_train_dataset, shuffle=True, batch_size=BATCH_SIZE)

custom_car_racing_val_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=DATASET_SIZE,
                                                                data_split="validate", split_ratio= SPLIT_RATIO,
                                                                crop_size=CROP_SIZE,image_size=IMG_SIZE)

custom_car_racing_val_dataloader = DataLoader(custom_car_racing_val_dataset, shuffle=False, batch_size=BATCH_SIZE)

model.assignFurther(train_loader=custom_car_racing_train_dataloader, val_loader=custom_car_racing_val_dataloader,
                    optimizer=optimizer,loss_fct=loss_crit)

model.train_model(model)

car_racing_env = CarRacing()


env = CarRacing(render_mode="human")
action = np.array([0.0, 0.0, 0.0])
_s_model = torch.zeros((1, 3, 96, 96), device="cuda")

model.eval()

quit = False
while not quit:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:

        s, r, terminated, truncated, info = env.step(action)
        #s = torchvision.transforms.functional.resized_crop(img=TOTENSOR(s), top=0, left=0,
         #                                                     height=80, width=96, antialias=True, size=(96, 96))
        s = TOTENSOR(s)
        s = s.to("cuda")

        _s_model[0,:,:,:] = s
        pred = torch.argmax(model(_s_model)).item()

        if pred==0:
            action = np.array([1.0, 0.0, 0.0])
        elif pred == 1:
            action = np.array([-1.0, 0.0, 0.0])
        elif pred == 2:
            action = np.array([0.0, 1.0, 0.0])
        elif pred == 3:
            action = np.array([0.0, 0.0, 0.8])

        total_reward += r
        if steps % 200 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in action]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if steps == 1000:
            quit = True

        if terminated or truncated or restart or quit:
            break
env.close()