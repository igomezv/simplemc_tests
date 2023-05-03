"""neural networks to neuralike.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: April 2022
"""
import sys

import numpy as np
# np.random.seed(0)
import matplotlib.pyplot as plt
from time import time
import math
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchinfo import summary
from torch_optimizer import AdaBound
from .pytorchtools import EarlyStopping
import torchbnn as bnn
from .nnogada.Nnogada import Nnogada
from .nnogada.Hyperparameter import Hyperparameter

# if torch.cuda.is_available():
# 	dev = "cuda:0"
# else:
# 	dev = "cpu"

device = torch.device("cpu")


class NeuralNet:
    def __init__(self, n_train, X, Y, n_input, n_output, hidden_layers_neurons=200,
                 load=False, model_path=None, dropout=0.5,
                 valid_loss=0.5, hyp_tunning='manual', **kwargs):
        """
        Read the network params
        Parameters
        -----------
        load: bool
            if True, then use an existing model
        X, Y: numpy array
            Data to train

        """
        hyp_tunning = 'manual'
        self.valid_loss = valid_loss
        self.n_train = n_train
        self.load = load
        self.model_path = model_path
        self.hidden_layers_neurons = hidden_layers_neurons
        self.dims = n_input
        self.n_output = n_output
        self.dropout = dropout
        self.epochs = kwargs.pop('epochs', 50)
        self.learning_rate = kwargs.pop('learning_rate', 5e-4)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.patience = kwargs.pop('patience', 5)
        self.deep = kwargs.pop('deep', 3)
        psplit = kwargs.pop('psplit', 0.7)
        # self.bayesian = True
        t = time()
        if load:
            self.model = self.load_model()
            self.model.summary()
        else:
            X_train, X_val, Y_train, Y_val = self.load_data(X, Y)
            if hyp_tunning == 'auto' and n_train == 0:
                population_size = 5  # max of individuals per generation
                max_generations = 2  # number of generations
                gene_length = 4  # lenght of the gene, depends on how many hiperparameters are tested
                k = 1  # num. of finralist individuals

                # Define the hyperparameters for the search
                #
                hyperparams = {'batch_size': [4, 8], 'deep': [2, 3], 'learning_rate': [0.0005, 0.001], 'num_units': [50, 100]}

                # generate a Nnogada instance
                epochs = Hyperparameter("epochs", None, self.epochs, vary=False)
                net_fit = Nnogada(hyp_to_find=hyperparams, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                                  neural_library='torch', epochs=epochs, dropout=self.dropout)
                # Set the possible values of hyperparameters and not use the default values from hyperparameters.py
                net_fit.set_hyperparameters()

                # Find best solutions
                net_fit.ga_with_elitism(population_size, max_generations, gene_length, k)
                # best solution
                self.hidden_layers_neurons = int(net_fit.best['num_units'])
                self.batch_size = int(net_fit.best['batch_size'])
                self.learning_rate = float(net_fit.best['learning_rate'])
                self.deep = int(net_fit.best['deep'])
                # print("best individual", net_fit.best)
                print("Best number of nodes:", net_fit.best['num_units'], type(self.hidden_layers_neurons))
                print("Best number of learning rate:", net_fit.best['learning_rate'], type(self.learning_rate))
                print("Best number of deep:", net_fit.best['deep'], type(self.batch_size))
                # print("Total elapsed time:", (time() - t) / 60, "minutes")
            # Initialize the MLP

            self.model = MLP(ncols=self.dims, noutput=self.n_output, hidden_layers_neurons=self.hidden_layers_neurons, nlayers=self.deep, dropout=self.dropout)
            self.model.apply(self.model.init_weights)
            self.model.float()
            print("Total elapsed time:", (time() - t) / 60, "minutes")
   
    def train(self, X, Y, n_train=0):
        if n_train >= 1:
            if self.learning_rate > 0.00001:
                self.learning_rate = self.learning_rate*0.5
        X_train, X_val, Y_train, Y_val = self.load_data(X, Y)
        dataset_train = LoadDataSet(X_train, Y_train)
        dataset_val = LoadDataSet(X_val, Y_val)
        # counter_valid = 0

        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=1)
        validloader = torch.utils.data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=True, num_workers=1)

        # Define the loss function and optimizer
        # loss_function = nn.L1Loss()
        loss_function = nn.MSELoss()
        # if self.bayesian:
        #     kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        #     kl_weight = 0.01
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-5)
        # optimizer = AdaBound(self.model.parameters(), lr=self.learning_rate, final_lr=0.01, weight_decay=1e-10, gamma=0.1)
        # optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
        #                                 lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=self.epochs//4, verbose=True)
        # it needs pytorch utilities
        summary(self.model)
        # t0 = time()
        # Run the training loop
        history_train = np.empty((1,))
        history_val = np.empty((1,))
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=False)
        for epoch in range(0, self.epochs):
            # Set current loss value
            current_loss = 0.0
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))
                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()

                if i % 10 == 0:
                    # print('Loss after mini-batch %5d: %.3f' %
                    #       #                 (i + 1, current_loss / 500))
                    #       (i + 1, loss.item()), end='\r')
                    current_loss = 0.0
            history_train = np.append(history_train, loss.item())

            valid_loss = 0.0
            self.model.eval()  # Optional when not using Model Specific layer
            for i, data in enumerate(validloader, 0):
                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))

                output_val = self.model(inputs)
                valid_loss = loss_function(output_val, targets)
                valid_loss += loss.item()
                # scheduler.step(valid_loss)

            history_val = np.append(history_val, valid_loss.item())
            print('Epoch: {}/{} | Training Loss: {:.5f} | Validation Loss:'
                  '{:.5f}'.format(epoch+1, self.epochs, loss.item(), valid_loss.item()), end='\r')
            # if valid_loss <= self.valid_loss and loss.item() <= self.valid_loss and epoch >= 500:
            #     counter_valid += 1
            # else:
            #     counter_valid = 0
            # if counter_valid >= 5:
            #     break
        # Process is complete.
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        # tf = time() - t0
        # print('\nTraining process has finished in {:.3f} minutes.'.format(tf/60))
        self.history = {'loss': history_train, 'val_loss': history_val}
        self.loss_val = history_val[-5:]
        self.loss_train = history_train[-5:]
        return self.history

    def predict(self, x):
        x = torch.from_numpy(x).float()
        prediction = self.model.forward(x)
        return prediction.detach().numpy()

    def plot(self, save=False, figname=False, ylogscale=False, show=False):
        plt.plot(self.history['loss'], label='training set')
        plt.plot(self.history['val_loss'], label='validation set')
        if ylogscale:
            plt.yscale('log')
        plt.title('MSE train: {:.4f} | MSE val: {:.4f} | '.format(self.loss_train[-1],
                                                            self.loss_val[-1]))
        plt.ylabel('loss function')
        plt.xlabel('epoch')
        # plt.ylim(0, 10)
        # plt.xlim(0, self.epochs)
        plt.legend(['train', 'val'], loc='upper left')
        if save and figname:
            plt.savefig(figname)
        if show:
            plt.show()
        return True

    def load_data(self, X, Y, psplit=0.8):
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=psplit, random_state = 42)
        print("\nNeuralike: Shape of X dataset: {} | Shape of Y dataset: {}".format(X_train.shape, Y_train.shape))
        print("Neuralike: Shape of X_val dataset: {} | Shape of Y_val dataset: {}".format(X_val.shape,
                                                                                          Y_val.shape))
        return X_train, X_val, Y_train, Y_val

    # def test_mse(self):
    #     y_pred = self.predict(self.X_test)
    #     mse = ((self.Y_test - y_pred) ** 2).mean()
    #     return mse

    # def tunning(self):


class LoadDataSet:
    def __init__(self, X, y, scale_data=False):
        """
        Prepare the dataset for regression
        """
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # # Apply scaling if necessary
            # if scale_data:
            #     X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, ncols, noutput, hidden_layers_neurons=200, dropout=0.5, nlayers=3):
        """
            Multilayer Perceptron for regression.
        """
        super().__init__()
        ncols = int(ncols)
        hidden_layers_neurons = int(hidden_layers_neurons)
        noutput = int(noutput)

        l_input = nn.Linear(ncols, hidden_layers_neurons)
        a_input = nn.ReLU()

        l_hidden = nn.Linear(hidden_layers_neurons, hidden_layers_neurons)
        a_hidden = nn.ReLU()
        # drop_hidden = nn.Dropout(dropout)

        l_output = nn.Linear(hidden_layers_neurons, noutput)

        # l = [l_input, a_input, drop_hidden]
        l = [l_input, a_input]
        for _ in range(nlayers):
            l.append(l_hidden)
            l.append(a_hidden)
            # l.append(drop_hidden)
        l.append(l_output)
        self.module_list = nn.ModuleList(l)

    def forward(self, x):
        for f in self.module_list:
            x = f(x)
        return x
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)


class L1(torch.nn.Module):
    def __init__(self, module, weight_decay=1e-5):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)