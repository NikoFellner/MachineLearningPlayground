# Implement your network
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):

    def __init__(self, IMG_SIZE, NUM_EPOCHS, INPUT_CHANNELS=3):
        super(NeuralNetwork,self).__init__()
        self.num_epochs = NUM_EPOCHS
        self.actualEpochNum = 1
        self.average_loss_train = np.zeros(NUM_EPOCHS)
        self.acc_train = np.zeros(NUM_EPOCHS)
        self.average_loss_val = np.zeros(NUM_EPOCHS)
        self.acc_val = np.zeros(NUM_EPOCHS)
        self.average_loss_test = 0
        self.acc_test = 0
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 16, (3, 3), padding="same")
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding="same")
        self.conv3 = nn.Conv2d(32, 64, (5, 5), padding="same")
        self.pool_3 = nn.AvgPool2d((3, 3), stride=1, padding=(0,0))
        self.pool_5 = nn.AvgPool2d((5, 5), stride=1, padding=(0,0))
        #IMG_SIZE = IMG_SIZE - 2 * (1 + 1 + 2 + 1 + 1 + 2)
        #IMG_SIZE = IMG_SIZE - 2 * (1 + 1 + 2 )
        self.fc1 = nn.Linear(64*IMG_SIZE*IMG_SIZE, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 4)

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

    def train_model(self, model):
        for epoch in range(self.num_epochs):
            print('\n--------------EPOCH {}:------------------'.format(epoch + 1))
            model.train(True)
            model.average_loss_train[epoch], model.acc_train[epoch] = model.train_one_epoch(model)
            print('---------------TRAIN---------------')
            print('LOSS train {}'.format(model.average_loss_train[epoch]))
            print('ACC train {} %'.format(model.acc_train[epoch]))
            model.train(False)
            model.average_loss_val[epoch], model.acc_val[epoch] = model.validate_epoch(model)
            print('-------------VALIDATE-------------')
            print('LOSS validate {}'.format(model.average_loss_val[epoch]))
            print('ACC validate {} %'.format(model.acc_val[epoch]))


    def train_one_epoch(self, model):
        running_loss = 0
        numCorrect = 0
        Batch = 1
        for i, data in enumerate(self.trainLoader):
            samples, labels = data
            self.optimizer.zero_grad()
            samples = samples.type(torch.FloatTensor)
            labels = labels.type(torch.IntTensor)
            samples = samples.to(device="cuda")
            labels = labels.type(torch.LongTensor).to(device="cuda")
            output = model(samples)
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
            correct = prediction == labels
            #prediction = self.one_hot_vector(torch.argmax(torch.softmax(output, dim=1), dim=1))
            #correct = prediction == labels
            #correct = torch.argmax(prediction, dim=1) == torch.argmax(labels, dim=1)

            numCorrect += correct.sum()
            loss = self.lossFct(output, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            Batch +=1
        last_loss = running_loss / self.trainLoader.__len__()  # loss per batch
        acc = numCorrect/(self.trainLoader.__len__()*self.trainLoader.batch_size) *100
        return last_loss, acc

    def one_hot_vector(self, pred):
        one_hot_pred = torch.zeros(len(pred),4)
        for row in range(len(pred)):
            one_hot_pred[row,pred[row]] = 1
        return one_hot_pred.to(device="cuda")

    def validate_epoch(self, model):
        running_loss = 0
        numCorrect =0
        for i, data in enumerate(self.valLoader):
            samples, labels = data
            samples = samples.type(torch.FloatTensor)
            labels = labels.type(torch.IntTensor)
            samples = samples.to(device="cuda")
            labels = labels.type(torch.LongTensor).to(device="cuda")
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
        model.train(False)
        for i, data in enumerate(self.testLoader):
            samples, labels = data
            samples = samples.type(torch.FloatTensor)
            labels = labels.type(torch.IntTensor)
            samples = samples.to(device="cuda")
            labels = labels.type(torch.LongTensor).to(device="cuda")
            self.optimizer.zero_grad()
            output = model(samples)
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
            correct = prediction == labels
            numCorrect += correct.sum()
            loss = self.lossFct(output, labels)
            running_loss += loss.item()
        self.average_loss_test = running_loss / self.testLoader.__len__()
        self.acc_test = numCorrect / (self.testLoader.__len__()*self.testLoader.batch_size) * 100
        print('---------------TEST---------------')
        print('LOSS validate {}'.format(model.average_loss_test))
        print('ACC validate {} %'.format(model.acc_test))
        return



    def assignFurther(self, train_loader=None, val_loader=None, test_loader=None, optimizer=None, loss_fct=None):
        self.trainLoader = train_loader
        self.valLoader = val_loader
        self.testLoader = test_loader
        self.optimizer = optimizer
        self.lossFct = loss_fct


    def show_figures(self, model):
        f, axarr = plt.subplots(1,2)
        f.suptitle("Model loss and accuracy")
        plt.subplot(1,2,1)
        plt.plot(np.arange(1, model.num_epochs+1), model.average_loss_train, label="Train-Loss")
        plt.plot(np.arange(1, model.num_epochs+1), model.average_loss_val, label="Validation-Loss")
        plt.title("Loss")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(np.arange(1, model.num_epochs+1), model.acc_train, label="Train-ACC")
        plt.plot(np.arange(1, model.num_epochs+1), model.acc_val, label="Validation-ACC")
        plt.legend()
        plt.title("Accuracy")
        plt.show()

class NeuralNetwork_2(nn.Module):

    def __init__(self, IMG_SIZE, NUM_EPOCHS, INPUT_CHANNELS=3):
        super(NeuralNetwork_2,self).__init__()
        self.num_epochs = NUM_EPOCHS
        self.actualEpochNum = 1
        self.average_loss_train = np.zeros(NUM_EPOCHS)
        self.acc_train = np.zeros(NUM_EPOCHS)
        self.average_loss_val = np.zeros(NUM_EPOCHS)
        self.acc_val = np.zeros(NUM_EPOCHS)
        self.average_loss_test = 0
        self.acc_test = 0
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 16, (3, 3), padding="same")
        self.batch_normalization1 = nn.BatchNorm2d(16, affine=True)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding="same")
        self.batch_normalization2 = nn.BatchNorm2d(32, affine=True)
        self.conv3 = nn.Conv2d(32, 64, (5, 5), padding="same")
        self.fc1 = nn.Linear(64*IMG_SIZE*IMG_SIZE, 512)
        self.batch_normalization3 = nn.BatchNorm1d(512, affine=True)
        self.fc2 = nn.Linear(512, 4)

        self.Softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_normalization1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.batch_normalization2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.batch_normalization3(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.Softmax(x)
        return x

    def train_model(self, model):
        for epoch in range(self.num_epochs):
            print('\n--------------EPOCH {}:------------------'.format(epoch + 1))
            model.train(True)
            model.average_loss_train[epoch], model.acc_train[epoch] = model.train_one_epoch(model)
            print('---------------TRAIN---------------')
            print('LOSS train {}'.format(model.average_loss_train[epoch]))
            print('ACC train {} %'.format(model.acc_train[epoch]))
            model.train(False)
            model.average_loss_val[epoch], model.acc_val[epoch] = model.validate_epoch(model)
            print('-------------VALIDATE-------------')
            print('LOSS validate {}'.format(model.average_loss_val[epoch]))
            print('ACC validate {} %'.format(model.acc_val[epoch]))


    def train_one_epoch(self, model):
        running_loss = 0
        numCorrect = 0
        Batch = 1
        for i, data in enumerate(self.trainLoader):
            samples, labels = data
            self.optimizer.zero_grad()
            samples = samples.type(torch.FloatTensor)
            labels = labels.type(torch.IntTensor)
            samples = samples.to(device="cuda")
            labels = labels.type(torch.LongTensor).to(device="cuda")
            output = model(samples)
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
            correct = prediction == labels
            #prediction = self.one_hot_vector(torch.argmax(torch.softmax(output, dim=1), dim=1))
            #correct = prediction == labels
            #correct = torch.argmax(prediction, dim=1) == torch.argmax(labels, dim=1)

            numCorrect += correct.sum()
            loss = self.lossFct(output, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            Batch +=1
        last_loss = running_loss / self.trainLoader.__len__()  # loss per batch
        acc = numCorrect/(self.trainLoader.__len__()*self.trainLoader.batch_size) *100
        return last_loss, acc

    def one_hot_vector(self, pred):
        one_hot_pred = torch.zeros(len(pred),4)
        for row in range(len(pred)):
            one_hot_pred[row,pred[row]] = 1
        return one_hot_pred.to(device="cuda")

    def validate_epoch(self, model):
        running_loss = 0
        numCorrect =0
        for i, data in enumerate(self.valLoader):
            samples, labels = data
            samples = samples.type(torch.FloatTensor)
            labels = labels.type(torch.IntTensor)
            samples = samples.to(device="cuda")
            labels = labels.type(torch.LongTensor).to(device="cuda")
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
        model.train(False)
        for i, data in enumerate(self.testLoader):
            samples, labels = data
            samples = samples.type(torch.FloatTensor)
            labels = labels.type(torch.IntTensor)
            samples = samples.to(device="cuda")
            labels = labels.type(torch.LongTensor).to(device="cuda")
            self.optimizer.zero_grad()
            output = model(samples)
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
            correct = prediction == labels
            numCorrect += correct.sum()
            loss = self.lossFct(output, labels)
            running_loss += loss.item()
        self.average_loss_test = running_loss / self.testLoader.__len__()
        self.acc_test = numCorrect / (self.testLoader.__len__()*self.testLoader.batch_size) * 100
        print('---------------TEST---------------')
        print('LOSS validate {}'.format(model.average_loss_test))
        print('ACC validate {} %'.format(model.acc_test))
        return



    def assignFurther(self, train_loader=None, val_loader=None, test_loader=None, optimizer=None, loss_fct=None):
        self.trainLoader = train_loader
        self.valLoader = val_loader
        self.testLoader = test_loader
        self.optimizer = optimizer
        self.lossFct = loss_fct


    def show_figures(self, model):
        f, axarr = plt.subplots(1,2)
        f.suptitle("Model loss and accuracy")
        plt.subplot(1,2,1)
        plt.plot(np.arange(1, model.num_epochs+1), model.average_loss_train, label="Train-Loss")
        plt.plot(np.arange(1, model.num_epochs+1), model.average_loss_val, label="Validation-Loss")
        plt.title("Loss")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(np.arange(1, model.num_epochs+1), model.acc_train, label="Train-ACC")
        plt.plot(np.arange(1, model.num_epochs+1), model.acc_val, label="Validation-ACC")
        plt.legend()
        plt.title("Accuracy")
        plt.show()