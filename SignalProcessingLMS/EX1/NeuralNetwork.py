# Implement your network
import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(nn.Module):

    def __init__(self, IMG_SIZE, NUM_EPOCHS):
        super(NeuralNetwork,self).__init__()
        self.numEpochs = NUM_EPOCHS
        self.actualEpochNum = 1
        self.average_LossTrain = np.zeros(NUM_EPOCHS)
        self.accTrain = np.zeros(NUM_EPOCHS)
        self.average_LossVal = np.zeros(NUM_EPOCHS)
        self.accVal = np.zeros(NUM_EPOCHS)
        self.average_LossTest = 0
        self.accTest = 0

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding="same")
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding="same")
        self.conv3 = nn.Conv2d(32, 64, (5, 5), padding="same")

        self.fc1 = nn.Linear(64*IMG_SIZE*IMG_SIZE, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.Softmax = nn.Softmax()


    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x



    def train_one_epoch(self, model):
        running_loss = 0
        numCorrect = 0
        Batch = 1
        for i, data in enumerate(self.trainLoader):
            samples, labels = data
            self.optimizer.zero_grad()
            samples = samples.to(device="cuda")
            labels = labels.to(device="cuda")
            output = model(samples)
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
            correct = prediction == labels
            numCorrect += correct.sum()
            loss = self.lossFct(output, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            Batch +=1
        last_loss = running_loss / self.trainLoader.__len__()  # loss per batch
        acc = numCorrect/(self.trainLoader.__len__()*self.trainLoader.batch_size) *100
        return last_loss, acc

    def validate_epoch(self, model):
        running_loss = 0
        numCorrect =0
        for i, data in enumerate(self.valLoader):
            samples, labels = data
            samples = samples.to(device="cuda")
            labels = labels.to(device="cuda")
            self.optimizer.zero_grad()
            output = model(samples)
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
            correct = prediction == labels
            numCorrect += correct.sum()
            loss = self.lossFct(output, labels)
            running_loss += loss.item()
        last_loss = running_loss / self.valLoader.__len__()  # loss per batch
        acc = numCorrect / (self.valLoader.__len__()*self.valLoader.batch_size) * 100
        return last_loss, acc

    def test_Model(self, model):
        running_loss = 0
        numCorrect = 0
        for i, data in enumerate(self.testLoader):
            samples, labels = data
            samples = samples.to(device="cuda")
            labels = labels.to(device="cuda")
            self.optimizer.zero_grad()
            output = model(samples)
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
            correct = prediction == labels
            numCorrect += correct.sum()
            loss = self.lossFct(output, labels)
            running_loss += loss.item()
        last_loss = running_loss / self.testLoader.__len__()
        acc = numCorrect / (self.testLoader.__len__()*self.testLoader.batch_size) * 100
        return last_loss, acc

    def assignFurther(self, trainLoader, valLoader, testLoader, optim, lossFct):
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.testLoader = testLoader
        self.optimizer = optim
        self.lossFct = lossFct
