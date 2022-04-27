"""neural networks to neuralike.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: April 2022
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from time import time

import torch
from torch import nn
from torchinfo import summary


class NeuralNet:
    def __init__(self, load=False, model_path=None, X=None, Y=None, topology=None, **kwargs):
        """
        Read the network params
        Parameters
        -----------
        load: bool
            if True, then use an existing model
        X, Y: numpy array
            Data to train

        """
        self.load = load
        self.model_path = model_path
        self.topology = topology
        self.epochs = kwargs.pop('epochs', 50)
        self.learning_rate = kwargs.pop('learning_rate', 5e-4)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.patience = kwargs.pop('patience', 5)
        psplit = kwargs.pop('psplit', 0.8)
        # self.valid_delta_mse = kwargs.pop('valid_delta_ms', 0.1)
        if load:
            self.model = self.load_model()
            self.model.summary()
        else:
            ntrain = int(psplit * len(X))
            indx = [ntrain]
            shuffle = np.random.permutation(len(X))
            X = X[shuffle]
            Y = Y[shuffle]
            self.X_train, self.X_test = np.split(X, indx)
            self.Y_train, self.Y_test = np.split(Y, indx)
            # Initialize the MLP
            self.model = MLP()
            # mlp = self.model()
            self.model.float()

        # self.mse = np.min(self.history.history['val_loss'])

    def train(self):
        dataset_train = LoadDataSet(self.X_train, self.Y_train)
        dataset_val = LoadDataSet(self.X_test, self.Y_test)

        # X_test = torch.from_numpy(X_test).float()
        # y_test = torch.from_numpy(y_test).float()

        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=1)
        validloader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=1)


        # Define the loss function and optimizer
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # try:
        summary(self.model)
        t0 = time()
        # Run the training loop
        history_train = np.empty((1,))
        history_val = np.empty((1,))
        nepochs = 50
        for epoch in range(0, nepochs):  # 5 epochs at maximum
            # Print epoch
            print(f'Starting epoch {epoch + 1}')

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
                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 10 == 0:
                    print('Loss after mini-batch %5d: %.3f' %
                          #                 (i + 1, current_loss / 500))
                          (i + 1, loss.item()), end='\r')
                    current_loss = 0.0
            history_train = np.append(history_train, current_loss)

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
            history_val = np.append(history_val, valid_loss.item())
            print('Training Loss: {:.3f} \t\t Validation Loss:' \
                  '{:.3f}'.format(loss.item(), valid_loss.item()))
        # Process is complete.
        print('Training process has finished.')
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
        plt.title('MSE: {:.4f}'.format(np.min(self.loss_val)))
        plt.ylabel('loss function')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if save and figname:
            plt.savefig(figname)
        if show:
            plt.show()
        return True

    def delta_loss(self):
        delta_loss = np.abs(self.loss_val - self.loss_train)
        return np.mean(delta_loss)
        # if np.all(delta_mse <= self.valid_delta_mse):
        #     return True
        # else:
        #     return False
    #

class LoadDataSet:
    '''
    Prepare the dataset for regression
    '''

    def __init__(self, X, y, scale_data=False):
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
    '''
    Multilayer Perceptron for regression.
    '''

    def __init__(self):
        super().__init__()
        ncols = 3
        self.layers = nn.Sequential(
            nn.Linear(ncols, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)
# """neural networks to neuralike.
# Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
# Date: Dec 2021
# """
#
# import numpy as np
# import matplotlib.pyplot as plt
# from time import time
#
#
# class NeuralNet:
#
#     def __init__(self, load=False, model_path=None, X=None, Y=None, topology=None, **kwargs):
#         """
#         Read the network params
#         Parameters
#         -----------
#         load: bool
#             if True, then use an existing model
#         X, Y: numpy array
#             Data to train
#
#         """
#
#         self.load = load
#         self.model_path = model_path
#         self.topology = topology
#         self.epochs = kwargs.pop('epochs', 50)
#         self.learning_rate = kwargs.pop('learning_rate', 5e-4)
#         self.batch_size = kwargs.pop('batch_size', 32)
#         self.patience = kwargs.pop('patience', 5)
#         psplit = kwargs.pop('psplit', 0.8)
#         # self.valid_delta_mse = kwargs.pop('valid_delta_ms', 0.1)
#
#         if load:
#             self.model = self.load_model()
#             self.model.summary()
#         else:
#             ntrain = int(psplit * len(X))
#             indx = [ntrain]
#             shuffle = np.random.permutation(len(X))
#             X = X[shuffle]
#             Y = Y[shuffle]
#             self.X_train, self.X_test = np.split(X, indx)
#             self.Y_train, self.Y_test = np.split(Y, indx)
#             self.model = self.model()
#
#         # self.mse = np.min(self.history.history['val_loss'])
#
#     def model(self):
#         try:
#             import tensorflow as tf
#             import tensorflow.keras as K
#         except:
#             import warnings
#             warnings.warn("Please install tensorflow library if you want to use neural networks")
#         # Red neuronal
#         model = K.models.Sequential()
#         # Hidden layers
#
#         for i, nodes in enumerate(self.topology):
#             if i == 0:
#                 model.add(K.layers.Dense(self.topology[1], input_dim=self.topology[0], activation='relu'))
#             elif 1 < i < len(self.topology) - 1:
#                 model.add(K.layers.Dense(self.topology[i], activation='relu'))
#             elif i == len(self.topology) - 1:
#                 model.add(K.layers.Dense(self.topology[i], activation='linear'))
#         # Adam recommendations from arxiv:1412.6980
#         optimizer = K.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-3)
#         # optimizer = K.optimizers.RMSprop(learning_rate=self.learning_rate)
#         model.compile(optimizer=optimizer, loss='mean_squared_error')
#
#         return model
#
#     def train(self):
#         try:
#             import tensorflow as tf
#             import tensorflow.keras as K
#         except:
#             import warnings
#             warnings.warn("Please install tensorflow library if you want to use neural networks")
#
#         print("Training neural network...")
#         callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
#                                                       min_delta=0.0,
#                                                       patience=self.patience,
#                                                       restore_best_weights=True)]
#         t0 = time()
#         self.history = self.model.fit(self.X_train,
#                                       self.Y_train,
#                                       validation_data=(self.X_test,
#                                                        self.Y_test),
#                                       epochs=self.epochs,
#                                       batch_size=self.batch_size,
#                                       shuffle=True,
#                                       verbose=1,
#                                       callbacks=callbacks,
#                                       use_multiprocessing=True)
#         tt = time() - t0
#         nepochs = len(self.history.history['val_loss'])
#         self.mse_val = np.array(self.history.history['val_loss'][-nepochs//5:])
#         self.mse_train = np.array(self.history.history['loss'][-nepochs//5:])
#         print("Training complete! Time training: {:.3f} min".format(tt/60.))
#         return self.history
#
#     def get_w_and_b(self, nlayer):
#         weights, biases = self.model.layers[nlayer].get_weights()
#         return weights, biases
#
#     def save_model(self, filename):
#         self.model.save(filename)
#         print('Neural net model {} saved!'.format(filename))
#
#     def load_model(self):
#         try:
#             import tensorflow as tf
#             import tensorflow.keras as K
#         except:
#             import warnings
#             warnings.warn("Please install tensorflow library if you want to use neural networks")
#         neural_model = tf.keras.models.load_model('{}'.format(self.model_path))
#         self.history = neural_model.history
#         return neural_model
#
#     def predict(self, x):
#         if type(x) == type([1]):
#             x = np.array(x)
#         elif type(x) == type(1):
#             x = np.array([x])
#         prediction = self.model.predict(x)
#         return prediction
#
#     def delta_mse(self):
#         delta_mse = np.abs(self.mse_val - self.mse_train)
#         return np.mean(delta_mse)
#         # if np.all(delta_mse <= self.valid_delta_mse):
#         #     return True
#         # else:
#         #     return False
#
#     def plot(self, save=False, figname=False, ylogscale=False, show=False):
#         plt.plot(self.history.history['loss'], label='training set')
#         plt.plot(self.history.history['val_loss'], label='validation set')
#         if ylogscale:
#             plt.yscale('log')
#         plt.title('MSE: {:.4f}'.format(np.min(self.mse_val)))
#         plt.ylabel('loss function')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'val'], loc='upper left')
#         if save and figname:
#             plt.savefig(figname)
#         if show:
#             plt.show()