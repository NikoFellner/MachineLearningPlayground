import numpy
import numpy as np
import matplotlib.pyplot as plt
from CustomDataLoaderCarRacing import CustomDataLoaderCarRacing
from torch.utils.data import DataLoader
from plot_images import plot_images

DATASET_SIZE = 5000
BATCH_SIZE = 30
SPLIT_RATIO = (60, 20, 20)
NUM_EPOCHS = 15
CROP_SIZE = 92
IMG_SIZE = 96
NUM_FIGURES = 2
NUM_ROWS = 5
NUM_COLUMNS = int(BATCH_SIZE / (NUM_ROWS*NUM_FIGURES))
print(f"number of calculated images = {NUM_ROWS*NUM_COLUMNS*NUM_FIGURES} should be batch-size {BATCH_SIZE}\n")
dict_data = np.load("data\car_Racing_Images.npz")
data = dict_data.f.arr_0
dict_data = np.load("data\car_Racing_Actions.npz")
label = dict_data.f.arr_0
label_int = np.argmax(label)


#data_dataset_train = CustomDataLoaderCarRacing(transforms=True, dataset_size=5000, data_split="train", split_ratio=(60, 20, 20),crop_size=88)
#plt.hist(np.argmax(data_dataset_train.label, axis=1))
#plt.show()
#data_dataset_val = CustomDataLoaderCarRacing(transforms=False, dataset_size=5000, data_split="validate", split_ratio=(60, 20, 20))
#plt.hist(np.argmax(data_dataset_val.label, axis=1))
#plt.show()
#data_dataset_test = CustomDataLoaderCarRacing(transforms=False, dataset_size=5000, data_split="test", split_ratio=(60, 20, 20))
#plt.hist(np.argmax(data_dataset_test.label, axis=1))
#plt.show()


data_dataset_train = CustomDataLoaderCarRacing(transforms=True, dataset_size=DATASET_SIZE,
                                                            data_split="train", split_ratio=SPLIT_RATIO,
                                                            crop_size=CROP_SIZE, image_size=IMG_SIZE, input_channels=3)
data_loader = DataLoader(data_dataset_train, shuffle=False, batch_size=BATCH_SIZE)


for i, data in enumerate(data_loader):
    samples, labels = data
    plot_images(samples, labels,NUM_FIGURES, NUM_ROWS, NUM_COLUMNS)