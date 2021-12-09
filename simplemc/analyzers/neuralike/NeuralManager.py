"""neuralike management object.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: Dec 2021
"""

from .NeuralNet import NeuralNet
from .RandomSampling import RandomSampling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

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
                 nrand=5, rootname='neural', neural_options=None):

        self.loglikelihood_fn = loglikelihood

        # self.pars_info = pars_info
        _, self.dims = np.shape(samples)
        ml_idx = np.argmax(likes)
        means = samples[ml_idx, :]

        self.nrand = nrand
        # self.epochs = epochs
        # self.plot = plot
        self.model_path = '{}.h5'.format(rootname)
        self.fig_path = '{}.png'.format(rootname)
        # self.nrand = nrand
        #
        rsampling = RandomSampling(self.loglikelihood_fn, means=means,
                                   cov=np.diag(np.ones(len(means)))*1e-6,
                                   nrand=self.nrand)
                                   # files_path=self.model_path)
        rsamples, rlikes = rsampling.make_dataset()

        self.likes = np.append(rlikes, likes)
        self.samples = np.append(rsamples, samples, axis=0)
        self.valid = False
        self.original_likes = likes
        # self.likes = likes
        # self.samples = samples

        if neural_options:
            self.learning_rate = neural_options['learning_rate']
            self.batch_size = neural_options['batch_size']
            self.epochs = neural_options['epochs']
            self.patience = neural_options['patience']
            self.psplit = neural_options['psplit']
            hidden_layers_neurons = neural_options['hidden_layers_neurons']
            self.plot = neural_options['plot']
            self.valid_delta_mse = 0.1
        else:
            self.learning_rate = 5e-4
            self.batch_size = 32
            self.epochs = 100
            self.patience = self.epochs//2
            self.psplit = 0.8
            hidden_layers_neurons = [100, 100, 100]
            self.plot = True
            self.valid_delta_mse = 0.1

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
                                      patience=self.patience)

        self.neural_model.train()
        # neural_model.save_model('{}'.format(self.model_path))
        if self.plot:
            self.neural_model.plot(save=True, figname='{}'.format(self.fig_path), show=False)

        delta_mse = np.abs(self.neural_model.mse_val - self.neural_model.mse_train)
        if np.all(delta_mse < self.valid_delta_mse):
            self.valid = True
            print("\nValid Neural net: mse_val={}, "
                  "mse_train={}".format(np.min(self.neural_model.mse_val),
                                        np.min(self.neural_model.mse_train)))
        else:
            self.valid = False
            print("\nNOT valid neural net. Delta_mse: {}".format(delta_mse))


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
        likes = np.array(likes_scaler.inverse_transform(self.neural_model.predict(np.array(params).reshape(1,-1))))
        # likes = np.array(self.neural_model.predict(np.array(params).reshape(1, -1)))
        if self.like_valid(likes):
            return likes
        else:
            print("Using original like")
            self.valid = False
            return self.loglikelihood_fn(params)

    def like_valid(self, like):
        maxl = np.max(self.original_likes)
        minl = np.min(self.original_likes)
        dev = np.std(self.original_likes[-100:])
        first_cond = (like < (maxl + dev))
        second_cond = (like > (minl - dev))
        # print("like: {}, maxl: {}, minl:{}".format(like, maxl, minl))
        if first_cond and second_cond:
            return True
        else:
            return False
