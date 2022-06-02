"""neuralike management object.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: Dec 2021
"""
import sys

from .NeuralNet import NeuralNet
from .RandomSampling import RandomSampling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import multiprocessing as mp

import os


class NeuralManager:
    """
        Manager for neural networks to learn likelihood function over a grid
        Parameters
        -----------
        loglikelihood
        pars_bounds
        rootname
        ndivsgrid
        hidden_layers_neurons
        epochs
        plot
    """

    # def __init__(self, loglikelihood, samples, likes,
    #              rootname='neural', neuralike_settings=None):
    def __init__(self, loglikelihood, rootname='neural', neuralike_settings=None):
        self.loglikelihood_fn = loglikelihood
        self.valid = False
        self.fig_path = '{}.png'.format(rootname)
        if neuralike_settings:
            self.learning_rate = neuralike_settings['learning_rate']
            self.batch_size = neuralike_settings['batch_size']
            self.epochs = neuralike_settings['epochs']
            self.patience = neuralike_settings['patience']
            self.psplit = neuralike_settings['psplit']
            self.hidden_layers_neurons = neuralike_settings['hidden_layers_neurons']
            self.plot = neuralike_settings['plot']
            self.valid_loss = neuralike_settings['valid_loss']
            self.nrand = neuralike_settings['nrand']
        else:
            self.learning_rate = 5e-4
            self.batch_size = 32
            self.epochs = 100
            self.patience = self.epochs//2
            self.psplit = 0.8
            self.hidden_layers_neurons = [100, 100, 100]
            self.plot = True
            self.valid_loss = 0.5
            self.nrand = 5
        # self.model_path = '{}.h5'.format(rootname)
        # if not self.modelChecker():
        #     self.training()
        # else:
        #     self.neural_model = self.load()

    def training(self, samples, likes):
        _, self.dims = np.shape(samples)
        self.topology = [self.dims] + self.hidden_layers_neurons + [1]

        ml_idx = np.argmax(likes)
        means = samples[ml_idx, :]

        rsampling = RandomSampling(self.loglikelihood_fn, means=means,
                                   cov=np.cov(samples.T),
                                   nrand=self.nrand)

        rsamples, rlikes = rsampling.make_dataset()

        likes = np.append(rlikes, likes)
        samples = np.append(rsamples, samples, axis=0)

        rsampling_test = RandomSampling(self.loglikelihood_fn, means=means,
                                        cov=np.cov(samples.T),
                                        nrand=int(0.1*len(likes)))
        rsamples_test, rlikes_test = rsampling_test.make_dataset()
        # # create scaler
        # # self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler(feature_range=(0.5, 1))
        # self.likes_scaler = StandardScaler()
        self.likes_scaler = MinMaxScaler(feature_range=(0, 1))
        #
        # self.scaler.fit(samples)
        self.likes_scaler.fit(likes.reshape(-1, 1))
        # # apply transforms
        # sc_samples = self.scaler.transform(samples)
        sc_likes = self.likes_scaler.transform(likes.reshape(-1, 1))

        self.neural_model = NeuralNet(X=samples, Y=sc_likes, topology=self.topology,
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      learrning_rate=self.learning_rate,
                                      patience=self.patience,
                                      minsample=np.min(np.abs(samples)))

        self.neural_model.train()
        # neural_model.save_model('{}'.format(self.model_path))
        if self.plot:
            self.neural_model.plot(save=True, figname='{}'.format(self.fig_path), show=False)

        # delta_loss = np.abs(self.neural_model.loss_val - self.neural_model.loss_train)
        lastval = self.neural_model.loss_val[-1]
        lasttrain = self.neural_model.loss_train[-1]
        test_set_test = self.test_neural(rsamples_test, rlikes_test)
        if lastval < self.valid_loss and lasttrain < self.valid_loss and test_set_test:
            self.valid = True
            print("\nValid Neural net | Train loss: {:.4f} | "
                  "Val loss: {:.4f}\n".format(lasttrain, lastval))
        else:
            self.valid = False
            print("\nNOT valid neural net | Train loss:{:.4f} | Val loss: {:.4f}\n".format(lasttrain, lastval))

    def load(self):
        neural_model = NeuralNet(load=True, model_path=self.model_path)
        return neural_model

    def modelChecker(self):
        """
        This method checks if the name of the model.h5 exists, if it already exists use it, otherwise train a
        new neural net in order to generate a new model.
        """
        if os.path.isfile('{}'.format(self.model_path)):
            return True
        else:
            return False

    def neuralike(self, params):
        # loglikelihood only can work if trainning was executed
        # sc_params = self.scaler.transform(np.array(params).reshape(len(params), self.dims))
        # pred = self.neural_model.predict(sc_params)
        pred = self.neural_model.predict(params)
        likes = self.likes_scaler.inverse_transform(pred)
        likes = np.array(likes)
        # likes = self.neural_model.predict(params)
        return likes


    def test_neural(self, x_test, y_test):
        # sc_x = self.scaler.transform(np.array(x_test).reshape(len(y_test), self.dims))
        # pred = self.neural_model.predict(sc_x)
        pred = self.neural_model.predict(x_test)
        y_pred_test = self.likes_scaler.inverse_transform(pred)
        # y_pred_test = self.neural_model.predict(x_test)
        mse = ((y_pred_test - y_test) ** 2).mean(axis=1)
        diff = np.abs(y_pred_test - y_test)
        maxlike_test = np.max(y_test)
        print("MSE in test set", mse)
        print("diff in test set", diff)
        if np.all(diff < np.abs(maxlike_test/10)):
            return True
        else:
            return False
