"""neuralike management object.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: Dec 2021
"""

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

    def __init__(self, loglikelihood, samples, likes,
                 rootname='neural', neuralike_settings=None):

        if neuralike_settings:
            self.learning_rate = neuralike_settings['learning_rate']
            self.batch_size = neuralike_settings['batch_size']
            self.epochs = neuralike_settings['epochs']
            self.patience = neuralike_settings['patience']
            self.psplit = neuralike_settings['psplit']
            hidden_layers_neurons = neuralike_settings['hidden_layers_neurons']
            self.plot = neuralike_settings['plot']
            self.valid_delta_loss = neuralike_settings['valid_delta_loss']
            self.nrand = neuralike_settings['nrand']
        else:
            self.learning_rate = 5e-4
            self.batch_size = 32
            self.epochs = 100
            self.patience = self.epochs//2
            self.psplit = 0.8
            hidden_layers_neurons = [100, 100, 100]
            self.plot = True
            self.valid_delta_loss = 0.05
            self.nrand = 5

        self.loglikelihood_fn = loglikelihood
        _, self.dims = np.shape(samples)
        ml_idx = np.argmax(likes)
        means = samples[ml_idx, :]

        self.model_path = '{}.h5'.format(rootname)
        self.fig_path = '{}.png'.format(rootname)

        rsampling = RandomSampling(self.loglikelihood_fn, means=means,
                                   cov=np.cov(samples.T),
                                   nrand=self.nrand)
                                   # files_path=self.model_path)
        rsamples, rlikes = rsampling.make_dataset()

        self.original_likes = likes
        self.likes = np.append(rlikes, likes)
        self.samples = np.append(rsamples, samples, axis=0)
        self.valid = False
        # self.likes = likes
        # self.samples = samples
        self.maxl = np.max(self.original_likes)
        self.minl = np.min(self.original_likes)
        self.dev = np.std(self.original_likes[-len(self.original_likes)//10:])



        self.topology = [self.dims] + hidden_layers_neurons + [1]

        if not self.modelChecker():
            self.training()
        else:
            self.neural_model = self.load()

    def training(self):
        # create scaler
        scaler = StandardScaler()
        # fit scaler on data
        scaler.fit(self.samples)
        # apply transform
        self.samples = scaler.transform(self.samples)
        likes_scaler = StandardScaler()
        likes_scaler.fit(self.likes.reshape(-1, 1))
        sc_likes = likes_scaler.transform(self.likes.reshape(-1, 1))

        self.neural_model = NeuralNet(X=self.samples, Y=sc_likes, topology=self.topology,
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      learrning_rate=self.learning_rate,
                                      patience=self.patience,
                                      valid_delta_loss=self.valid_delta_loss)

        self.neural_model.train()
        # neural_model.save_model('{}'.format(self.model_path))
        if self.plot:
            self.neural_model.plot(save=True, figname='{}'.format(self.fig_path), show=False)

        delta_loss = np.abs(self.neural_model.loss_val - self.neural_model.loss_train)
        if self.neural_model.delta_loss() < self.valid_delta_loss:
            self.valid = True
            print("\nValid Neural net: loss_val={}, "
                  "loss_train={}".format(np.min(self.neural_model.loss_val),
                                        np.min(self.neural_model.loss_train)))
        else:
            self.valid = False
            print("\nNOT valid neural net. Delta_mse: {}".format(delta_loss))


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

    def loglikelihood(self, params):
        likes_scaler = StandardScaler()
        likes_scaler.fit(self.likes.reshape(-1, 1))
        pred = self.neural_model.predict(np.array(params).reshape(1, -1))

        likes = likes_scaler.inverse_transform(pred)
        likes = np.array(likes)
        if self.like_valid(likes):
            return likes
        else:
            print("Using original like", end='\r')
            self.valid = False
            return self.loglikelihood_fn(params)

    def like_valid(self, loglike):
        # first_cond = (loglike < (self.maxl + self.neural_model.delta_mse()))
        # second_cond = (loglike > (self.minl - self.neural_model.delta_mse()))
        first_cond = (loglike < (10*self.maxl))
        second_cond = (loglike > (self.minl/10))
        if first_cond and second_cond:
            return True
        else:
            return False