
## TODO: usar modelChecker once it is trained, and flag for overige

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
                 nrand=10, rootname='neural', neural_options=None):

        self.loglikelihood_fn = loglikelihood

        # self.pars_info = pars_info
        _, self.dims = np.shape(samples)
        ml_idx = np.argmax(likes)
        means = samples[ml_idx, :]

        # self.nrand = nrand
        # self.epochs = epochs
        # self.plot = plot
        self.model_path = '{}.h5'.format(rootname)
        self.fig_path = '{}.png'.format(rootname)
        self.nrand = nrand

        rsampling = RandomSampling(self.loglikelihood_fn, means=means,
                                   cov=np.diag(np.ones(len(means)))*1e-6,
                                   nrand=self.nrand)
                                   # files_path=self.model_path)
        rsamples, rlikes = rsampling.make_dataset()

        self.likes = np.append(rlikes, likes)
        self.samples = np.append(rsamples, samples, axis=0)

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
        else:
            self.learning_rate = 5e-4
            self.batch_size = 32
            self.epochs = 100
            self.patience = self.epochs//2
            self.psplit = 0.8
            hidden_layers_neurons = [100, 100, 100]
            self.plot = True

        self.topology = [self.dims] + hidden_layers_neurons + [1]

        if not self.modelChecker():
            self.training()
        else:
            self.neural_model = self.load()


    def training(self):
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
        if self.neural_model.mse_val < 0.5 and self.neural_model.mse_train < 0.5:
            return True
        else:
            return False

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
        return likes

