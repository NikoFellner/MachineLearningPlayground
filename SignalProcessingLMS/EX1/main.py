import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from NeuralNetwork import NeuralNetwork
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


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

fashionmnistDataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(24), transforms.RandomVerticalFlip(0.5)]))
fashionmnistTestset= datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(24), transforms.RandomVerticalFlip(0.5)]))
fashionmnist_TrainSet, fashionmnist_ValSet = torch.utils.data.random_split(fashionmnistDataset, [50000, 10000])
fashionmnist_trainLoader = DataLoader(fashionmnist_TrainSet, batch_size=50)
fashionmnist_valLoader = DataLoader(fashionmnist_ValSet, batch_size=10)
fashionmnist_testLoader = DataLoader(fashionmnistTestset, batch_size=10)

lossFct = nn.CrossEntropyLoss()
model = NeuralNetwork(IMG_SIZE=24, NUM_EPOCHS=10)
if torch.cuda.is_available():
    model.to("cuda")
optimizerFashionMNIST = optim.Adam(model.parameters(), lr=0.01)

model.assignFurther(fashionmnist_trainLoader, fashionmnist_valLoader, fashionmnist_testLoader, optimizerFashionMNIST, lossFct)

for epoch in range(model.numEpochs):
    print('EPOCH {}:'.format(epoch+1))
    model.train(True)
    model.average_LossTrain[epoch], model.accTrain[epoch] = model.train_one_epoch(model)
    model.train(False)
    model.average_LossVal[epoch], model.accVal[epoch] = model.validate_epoch(model)
    print('LOSS train {} val {}'.format(model.average_LossTrain[epoch],model.average_LossVal[epoch]))
    print('ACC train {} % val {} %'.format(model.accTrain[epoch], model.accVal[epoch]))

    # Test
model.train(False)
model.average_LossTest, model.accTest = model.test_Model(model)
print('Loss test {}, ACC test {} %'.format(model.average_LossTest, model.accTest))
# Show figures
f, axarr = plt.subplots(1,2)
f.suptitle("FashionMNIST Model loss and accuracy")
plt.subplot(1,2,1)
plt.plot(np.arange(1, model.numEpochs+1), model.average_LossTrain, label="Train-Loss")
plt.plot(np.arange(1, model.numEpochs+1), model.average_LossVal, label="Validation-Loss")
plt.title("Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(np.arange(1, model.numEpochs+1), model.accTrain, label="Train-ACC")
plt.plot(np.arange(1, model.numEpochs+1), model.accVal, label="Validation-ACC")
plt.legend()
plt.title("Accuracy")
plt.show()
