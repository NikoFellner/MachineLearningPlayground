import sklearn.metrics
import torch
import torch as t
import torchmetrics
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        self.sweet_spot = None
        self.epoch_counter = 0

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called.
        # This behavior is not required here, so you need to ensure that all the gradients are zero before calling
        # the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        self._optim.zero_grad()
        output = self._model.forward(x)
        loss = self._crit(output, y)
        loss.backward()
        self._optim.step()
        return loss
    
    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO

        prediction = self._model.forward(x)
        loss = self._crit(prediction, y)
        prediction = torch.gt(prediction, 0.5)*1

        return loss, prediction
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        loss_epoch = []
        for x, y in self._train_dl:
            if self._cuda == True:
                x = x.to(device="cuda")
                y = y.to(device="cuda")
            loss = self.train_step(x, y)
            loss_epoch.append(loss)

        loss_average = sum(loss_epoch)/len(loss_epoch)
        #print("train loss: ", float(loss_average))
        return loss_average


    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model.eval()
        with torch.no_grad():
            label_predictions = None
            loss_epoch = []
            y_epoch = None
            for x, y in self._val_test_dl:
                if self._cuda == True:
                    x = x.to(device="cuda")
                    y = y.to(device="cuda")

                loss, output = self.val_test_step(x, y)
                loss_epoch.append(loss)

                if label_predictions == None:
                    label_predictions = output
                    y_epoch = y
                else:
                    label_predictions = torch.cat([label_predictions, output], dim=0)
                    y_epoch = torch.cat([y_epoch, y], dim=0)
                #t.cuda.empty_cache()

        f1_crack = f1_score(label_predictions[:, 0].cpu(), y_epoch[:,0].cpu(), average='micro')
        f1_inactive = f1_score(label_predictions[:, 1].cpu(), y_epoch[:, 1].cpu(), average='micro')
        loss_average = sum(loss_epoch)/len(loss_epoch)
        #print("validation loss: ", float(loss_average))
        return loss_average, (f1_crack, f1_inactive)

    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        running = True
        counter = 0
        train_loss = []
        validation_loss = []
        self.epoch_counter = 0
        flag = False
        patience = 0
        while running:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            loss_epoch = self.train_epoch()
            train_loss.append(float(loss_epoch))

            validation_loss_epoch, f1 = self.val_test()
            validation_loss.append(float(validation_loss_epoch))

            if self.epoch_counter > 0:
                if (self.val_loss_min < validation_loss_epoch):
                    patience += 1
                    if patience > self._early_stopping_patience:
                        running = False
                if self.val_loss_min > validation_loss_epoch:
                    #self.sweet_spot = self.epoch_counter
                    self.sweet_spot = 1
                    self.save_checkpoint(epoch=self.sweet_spot)
                    self.val_loss_min = validation_loss_epoch
                    if patience > 0:
                        patience = 0
            else:
                self.val_loss_min = validation_loss_epoch


            self.epoch_counter += 1
            if self.epoch_counter >= epochs:
                running = False

            print("Epoch: ", self.epoch_counter)
            if patience != 0:
                print("patience", patience,"/",self._early_stopping_patience)
            print("F1_crack: ", float(f1[0]), "       ", " F1_inactive: ", float(f1[1]))
            print("train_loss: ", float(loss_epoch), "     ", "validation loss: ", float(validation_loss_epoch))
        return train_loss, validation_loss


        
        
        
