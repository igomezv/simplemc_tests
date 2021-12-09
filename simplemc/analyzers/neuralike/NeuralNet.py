"""neural networks to neuralike.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: Dec 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time


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
        self.patience = kwargs.pop('patience', 100)
        psplit = kwargs.pop('psplit', 0.8)

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
            self.model = self.model()

        # self.mse = np.min(self.history.history['val_loss'])

    def model(self):
        try:
            import tensorflow as tf
            import tensorflow.keras as K
        except:
            import warnings
            warnings.warn("Please install tensorflow library if you want to use neural networks")
        # Red neuronal
        model = K.models.Sequential()
        # Hidden layers

        for i, nodes in enumerate(self.topology):
            if i == 0:
                model.add(K.layers.Dense(self.topology[1], input_dim=self.topology[0], activation='relu'))
            elif 1 < i < len(self.topology) - 1:
                model.add(K.layers.Dense(self.topology[i], activation='relu'))
            elif i == len(self.topology) - 1:
                model.add(K.layers.Dense(self.topology[i], activation='linear'))
        # Adam recommendations from arxiv:1412.6980
        optimizer = K.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-3)
        # optimizer = K.optimizers.RMSprop(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    def train(self):
        try:
            import tensorflow as tf
            import tensorflow.keras as K
        except:
            import warnings
            warnings.warn("Please install tensorflow library if you want to use neural networks")

        print("Training neural network...")
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                      min_delta=0.0,
                                                      patience=self.patience,
                                                      restore_best_weights=True)]
        t0 = time()
        self.history = self.model.fit(self.X_train,
                                      self.Y_train,
                                      validation_data=(self.X_test,
                                                       self.Y_test),
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      verbose=1,
                                      callbacks=callbacks)
        tt = time() - t0
        self.mse_val = np.array(self.history.history['val_loss'][-int(self.patience):])
        self.mse_train = np.array(self.history.history['loss'][-int(self.patience):])
        print("Training complete! Time training: {:.3f} min".format(tt/60.))
        return self.history

    def get_w_and_b(self, nlayer):
        weights, biases = self.model.layers[nlayer].get_weights()
        return weights, biases

    def save_model(self, filename):
        self.model.save(filename)
        print('Neural net model {} saved!'.format(filename))

    def load_model(self):
        try:
            import tensorflow as tf
            import tensorflow.keras as K
        except:
            import warnings
            warnings.warn("Please install tensorflow library if you want to use neural networks")
        neural_model = tf.keras.models.load_model('{}'.format(self.model_path))
        self.history = neural_model.history
        return neural_model

    def predict(self, x):
        if type(x) == type([1]):
            x = np.array(x)
        if type(x) == type(1):
            x = np.array([x])

        prediction = self.model.predict(x)

        return prediction

    def plot(self, save=False, figname=False, ylogscale=False, show=False):
        plt.plot(self.history.history['loss'], label='training set')
        plt.plot(self.history.history['val_loss'], label='validation set')
        if ylogscale:
            plt.yscale('log')
        plt.title('MSE: {:.4f}'.format(np.min(self.mse_val)))
        plt.ylabel('loss function')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if save and figname:
            plt.savefig(figname)
        if show:
            plt.show()

