import urllib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from CustomClassification import CustomClassification
from NeuralNetwork import NeuralNetwork
import random
import gzip
import numpy as np
import os
from sklearn.model_selection import train_test_split
# Dataloader MNIST

classesFashionMnist = {
    0: "T-Shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

fashionmnistDatasset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
fashionmnistDataloader = DataLoader(fashionmnistDatasset, batch_size=6, shuffle=True)

img, label = next(iter(fashionmnistDataloader))
f, axarr = plt.subplots(2, 3)
f.suptitle("FashionMNIST Data examples - Pytorch Dataset Class - for one Batch")
n = 0
for i in range(2):
    for ii in range(3):
        axarr[i, ii].imshow(img[n][0], cmap='gray')
        axarr[i, ii].set_title(f"label: {label[n]} = {classesFashionMnist[label[n].item()]}")

        n += 1
plt.show()



"""
mnist_trainData = CustomClassification(library="MNIST", train=True)
mnist_TrainDataLoader = DataLoader(mnist_trainData, batch_size=60, shuffle=True)
print(mnist_trainData.__len__())
# random sample
f, axarr = plt.subplots(1,3)
f.suptitle("MNIST Data examples - CustomClassification Class")
for i in range(3):
    idx = random.randint(1, mnist_trainData.__len__())
    sample, label = mnist_trainData.__getitem__(idx=idx)
    axarr[i].imshow(sample, cmap='gray')
    axarr[i].set_title(f"label: {label}")
plt.show()


fashionmnist_trainData = CustomClassification(library="FashionMNIST", train=True)
fashionmnist_DataLoader = DataLoader(fashionmnist_trainData, batch_size=60, shuffle=True)
print(fashionmnist_trainData.__len__())
f, axarr = plt.subplots(1,3)
f.suptitle("FashionMNIST Data examples - CustomClassification Class")
for i in range(3):
    idx = random.randint(1, fashionmnist_trainData.__len__())
    sample, label = fashionmnist_trainData.__getitem__(idx=idx)
    axarr[i].imshow(sample, cmap='gray')
    axarr[i].set_title(f"label: {label}  = {classesFashionMnist[label]}")
plt.show()
"""
"""
#MNIST Dataloader using Pytorch prebuild
mnistDataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnistTestset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
mnist_TrainSet, mnist_ValSet = torch.utils.data.random_split(mnistDataset, [50000, 10000])
mnist_trainLoader = DataLoader(mnist_TrainSet, batch_size=6, shuffle=True)
mnist_valLoader = DataLoader(mnist_ValSet,batch_size=10, shuffle=True)
mnist_testLoader = DataLoader(mnistTestset, batch_size=10)

f, axarr = plt.subplots(1,3)
f.suptitle("MNIST Data examples - Pytorch Dataset Class")
for i in range(3):
    idx = random.randint(1, mnistDataset.__len__())
    sample, label = mnistDataset.__getitem__(idx)
    axarr[i].imshow(sample[0], cmap='gray')
    axarr[i].set_title(f"label: {label}")
plt.show()

# Dataloader Fashion-MNIST
fashionmnistDataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
fashionmnistTestset= datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
fashionmnist_TrainSet, fashionmnist_ValSet = torch.utils.data.random_split(fashionmnistDataset, [50000, 10000])
fashionmnist_trainLoader = DataLoader(fashionmnist_TrainSet)
fashionmnist_valLoader = DataLoader(fashionmnist_ValSet)
fashionmnist_testLoader = DataLoader(fashionmnistTestset)
f, axarr = plt.subplots(1,3)
f.suptitle("FashionMNIST Data examples - Pytorch Dataset Class")
for i in range(3):
    idx = random.randint(1, fashionmnistDataset.__len__())
    sample, label = fashionmnistDataset.__getitem__(idx)
    axarr[i].imshow(sample[0], cmap='gray')
    axarr[i].set_title(f"label: {label} = {classesFashionMnist[label]}")
plt.show()
"""

"""
# Train MNIST Dataset
lossFctMNIST = nn.CrossEntropyLoss()
mnistModel = NeuralNetwork(IMG_SIZE=24, NUM_EPOCHS=3)
optimizerMNIST = optim.SGD(mnistModel.parameters(), lr=0.01)

mnistModel.assignFurther(trainLoader=mnist_trainLoader, valLoader=mnist_valLoader, testLoader=mnist_testLoader, optim=optimizerMNIST, lossFct=lossFctMNIST)

for epoch in range(mnistModel.numEpochs):
    print('EPOCH {}:'.format(epoch+1))
    mnistModel.train(True)
    mnistModel.average_LossTrain[epoch], mnistModel.accTrain[epoch] = mnistModel.train_one_epoch(mnistModel)
    mnistModel.train(False)
    mnistModel.average_LossVal[epoch], mnistModel.accVal[epoch] = mnistModel.validate_epoch(mnistModel)
    print('LOSS train {} val {}'.format(mnistModel.average_LossTrain[epoch],mnistModel.average_LossVal[epoch]))
    print('ACC train {} % val {} %'.format(mnistModel.accTrain[epoch], mnistModel.accVal[epoch]))

# Test
mnistModel.train(False)
mnistModel.average_LossTest, mnistModel.accTest = mnistModel.test_Model(mnistModel)
print('Loss test {}, ACC test {} %'.format(mnistModel.average_LossTest, mnistModel.accTest))
# Show figures
f, axarr = plt.subplots(1,2)
f.suptitle("MNIST Model loss and accuracy")
plt.subplot(1,2,1)
plt.plot(np.arange(1, mnistModel.numEpochs+1), mnistModel.average_LossTrain, label="Train-Loss")
plt.plot(np.arange(1, mnistModel.numEpochs+1), mnistModel.average_LossVal, label="Validation-Loss")
plt.title("Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(np.arange(1, mnistModel.numEpochs+1), mnistModel.accTrain, label="Train-ACC")
plt.plot(np.arange(1, mnistModel.numEpochs+1), mnistModel.accVal, label="Validation-ACC")
plt.legend()
plt.title("Accuracy")
plt.show()

# Train FashionMNIST Dataset
lossFctFashionMNIST = nn.CrossEntropyLoss()
fashionMnistModel = NeuralNetwork(IMG_SIZE=24, NUM_EPOCHS=3)
optimizerFashionMNIST = optim.SGD(fashionMnistModel.parameters(), lr=0.01)

fashionMnistModel.assignFurther(trainLoader=fashionmnist_trainLoader, valLoader=fashionmnist_valLoader, testLoader=fashionmnist_testLoader, optim=optimizerFashionMNIST, lossFct=lossFctFashionMNIST)

for epoch in range(fashionMnistModel.numEpochs):
    print('EPOCH {}:'.format(epoch+1))
    fashionMnistModel.train(True)
    fashionMnistModel.average_LossTrain[epoch], fashionMnistModel.accTrain[epoch] = fashionMnistModel.train_one_epoch(fashionMnistModel)
    fashionMnistModel.train(False)
    fashionMnistModel.average_LossVal[epoch], fashionMnistModel.accVal[epoch] = fashionMnistModel.validate_epoch(fashionMnistModel)
    print('LOSS train {} val {}'.format(fashionMnistModel.average_LossTrain[epoch],fashionMnistModel.average_LossVal[epoch]))
    print('ACC train {} % val {} %'.format(fashionMnistModel.accTrain[epoch], fashionMnistModel.accVal[epoch]))

# Test
fashionMnistModel.train(False)
fashionMnistModel.average_LossTest, fashionMnistModel.accTest = fashionMnistModel.test_Model(fashionMnistModel)
print('Loss test {}, ACC test {} %'.format(fashionMnistModel.average_LossTest, fashionMnistModel.accTest))
# Show figures
f, axarr = plt.subplots(1,2)
f.suptitle("FashionMNIST Model loss and accuracy")
plt.subplot(1,2,1)
plt.plot(np.arange(1, fashionMnistModel.numEpochs+1), fashionMnistModel.average_LossTrain, label="Train-Loss")
plt.plot(np.arange(1, fashionMnistModel.numEpochs+1), fashionMnistModel.average_LossVal, label="Validation-Loss")
plt.title("Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(np.arange(1, fashionMnistModel.numEpochs+1), fashionMnistModel.accTrain, label="Train-ACC")
plt.plot(np.arange(1, fashionMnistModel.numEpochs+1), fashionMnistModel.accVal, label="Validation-ACC")
plt.legend()
plt.title("Accuracy")
plt.show()
"""
"""
mnistTrainData_augment = CustomClassification(library="MNIST", train=True, transform=True)
mnistTTestData_augment = CustomClassification(library="MNIST", train=False, transform=True)
mnist_TrainSet_augment, mnist_ValSet_augment = torch.utils.data.random_split(mnistTrainData_augment, [50000, 10000])
mnist_TrainDataLoader_augment = DataLoader(mnist_TrainSet_augment, batch_size=50, shuffle=True)
mnist_ValDataLoader_augment = DataLoader(mnist_ValSet_augment, batch_size=10, shuffle=True)
mnist_TestDataLoader_augment = DataLoader(mnist_ValSet_augment, batch_size=10, shuffle=True)


lossFctMNIST = nn.CrossEntropyLoss()
mnistModel = NeuralNetwork(IMG_SIZE=24, NUM_EPOCHS=10)
optimizerMNIST = optim.SGD(mnistModel.parameters(), lr=0.01)

mnistModel.assignFurther(trainLoader=mnist_TrainDataLoader_augment, valLoader=mnist_ValDataLoader_augment, testLoader=mnist_TestDataLoader_augment, optim=optimizerMNIST, lossFct=lossFctMNIST)
for epoch in range(mnistModel.numEpochs):
    print('EPOCH {}:'.format(epoch+1))
    mnistModel.train(True)
    mnistModel.average_LossTrain[epoch], mnistModel.accTrain[epoch] = mnistModel.train_one_epoch(mnistModel)
    mnistModel.train(False)
    mnistModel.average_LossVal[epoch], mnistModel.accVal[epoch] = mnistModel.validate_epoch(mnistModel)
    print('LOSS train {} val {}'.format(mnistModel.average_LossTrain[epoch],mnistModel.average_LossVal[epoch]))
    print('ACC train {} % val {} %'.format(mnistModel.accTrain[epoch], mnistModel.accVal[epoch]))


f, axarr = plt.subplots(1,2)
f.suptitle("FashionMNIST Model loss and accuracy")
plt.subplot(1,2,1)
plt.plot(np.arange(1, mnistModel.numEpochs+1), mnistModel.average_LossTrain, label="Train-Loss")
plt.plot(np.arange(1, mnistModel.numEpochs+1), mnistModel.average_LossVal, label="Validation-Loss")
plt.title("Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(np.arange(1, mnistModel.numEpochs+1), mnistModel.accTrain, label="Train-ACC")
plt.plot(np.arange(1, mnistModel.numEpochs+1), mnistModel.accVal, label="Validation-ACC")
plt.legend()
plt.title("Accuracy")
plt.show()

"""

#Mnist DataLoaders with augmentation
trans = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomCrop(size=(24,24)),
                torchvision.transforms.RandomVerticalFlip(p=0.5)
            ])


mnistTrainData_augment = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
mnistTTestData_augment = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
mnist_TrainSet_augment, mnist_ValSet_augment = torch.utils.data.random_split(mnistTrainData_augment, [50000, 10000])
mnist_trainLoader_augment = DataLoader(mnist_TrainSet_augment, batch_size=6, shuffle=True)
mnist_valLoader_augment = DataLoader(mnist_ValSet_augment,batch_size=10, shuffle=True)
mnist_testLoader_augment = DataLoader(mnistTTestData_augment, batch_size=10)

# Implement your network (MNIST) with augmentation
lossFctMNIST_augment = nn.CrossEntropyLoss()
mnistModel_augment = NeuralNetwork(IMG_SIZE=24, NUM_EPOCHS=10)
optimizerMNIST_augment = optim.SGD(mnistModel_augment.parameters(), lr=0.01)

mnistModel_augment.assignFurther(trainLoader=mnist_trainLoader_augment, valLoader=mnist_valLoader_augment, testLoader=mnist_testLoader_augment, optim=optimizerMNIST_augment, lossFct=lossFctMNIST_augment)

for epoch in range(mnistModel_augment.numEpochs):
    print('EPOCH {}:'.format(epoch+1))
    mnistModel_augment.train(True)
    mnistModel_augment.average_LossTrain[epoch], mnistModel_augment.accTrain[epoch] = mnistModel_augment.train_one_epoch(mnistModel_augment)
    mnistModel_augment.average_LossVal[epoch], mnistModel_augment.accVal[epoch] = mnistModel_augment.validate_epoch(mnistModel_augment)
    print('LOSS train {} val {}'.format(mnistModel_augment.average_LossTrain[epoch],mnistModel_augment.average_LossVal[epoch]))
    print('ACC train {} % val {} %'.format(mnistModel_augment.accTrain[epoch], mnistModel_augment.accVal[epoch]))
