import csv
import os
import torch as t
from torch import nn, optim
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules

csv_path = ''
for root, _, files in os.walk('.'):
        for name in files:
                if name == 'data.csv':
                        csv_path = os.path.join(root, name)
tab = pd.read_csv(csv_path, sep=';')
data = pd.read_csv(csv_path, sep=";")

train, valid = train_test_split(data,test_size=0.2)


# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects

train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)
data_train = t.utils.data.DataLoader(ChallengeDataset(train, mode="train"), batch_size=100, shuffle=True)
data_validation = t.utils.data.DataLoader(ChallengeDataset(valid, mode="val"), batch_size=20, shuffle=True)

# create an instance of our ResNet model
model_resnet = model.ResNet()


# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion

lossFunction = nn.BCELoss()
optimizer = optim.Adam(params=model_resnet.ResNet.parameters())
#optimizer = optim.SGD(params=model_resnet.ResNet.parameters(), lr=0.1, momentum=0.9)

trainer = Trainer(model=model_resnet, crit=lossFunction, optim=optimizer, train_dl=data_train, val_test_dl=data_validation, cuda=True, early_stopping_patience=5)


# go, go, go... call fit on trainer

res = trainer.fit(10)

#TODO
print(res)
# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()

trainer.restore_checkpoint(trainer.sweet_spot)
trainer.save_onnx(fn="Model1.onnx")