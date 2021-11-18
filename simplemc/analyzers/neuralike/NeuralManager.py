
## TODO: usar modelChecker once it is trained, and flag for overige

from .NeuralNet import NeuralNet
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

    def __init__(self, loglikelihood, pars_info, rootname, nrand=500, hidden_layers_neurons=None,
                 epochs=100, patience=None, plot=True, **kwargs):
        if hidden_layers_neurons is None:
            hidden_layers_neurons = [100, 100, 100]
        self.loglikelihood = loglikelihood

        self.pars_info = pars_info
        self.dims = len([None for _ in pars_info])

        self.nrand = nrand
        self.epochs = epochs
        self.plot = plot
        # self.grid_path = 'simplemc/analyzers/neuralike/neural_models/{}'.format(rootname)
        self.model_path = 'simplemc/analyzers/neuralike/neural_models/{}.h5'.format(rootname)
        self.fig_path = 'simplemc/analyzers/neuralike/neural_models/{}.png'.format(rootname)

        self.learning_rate = kwargs.pop('learning_rate', 5e-4)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.patience = kwargs.pop('patience', epochs)
        self.psplit = kwargs.pop('psplit', 0.8)
        self.topology = [self.dims] + hidden_layers_neurons + [1]

        if not self.modelChecker():
            self.training()
        self.neural_model = self.load()


    def training(self, samples, likes):
        # rsampling = RandomSampling(self.loglikelihood, pars_info=self.pars_info, nrand=self.nrand, files_path=self.grid_path)
        # samples, likes = rsampling.make_dataset()
        # # samples_scaler = StandardScaler()
        # samples_scaler = MinMaxScaler(feature_range=(-1, 1))
        # # fit scaler on data
        # samples_scaler.fit(samples.reshape(-1, 1))
        # # apply transform
        # sc_samples = samples_scaler.transform(samples.reshape(-1, 1))
        #
        # likes_scaler = MinMaxScaler(feature_range=(-1, 1))
        likes_scaler = StandardScaler()
        likes_scaler.fit(likes.reshape(-1, 1))
        sc_likes = likes_scaler.transform(likes.reshape(-1, 1))
        print("sc_likes\n", sc_likes)

        neural_model = NeuralNet(X=samples, Y=sc_likes, topology=self.topology, epochs=self.epochs,
                                 batch_size=self.batch_size, learrning_rate=self.learning_rate, patience=self.patience)

        neural_model.train()
        neural_model.save_model('{}'.format(self.model_path))
        if self.plot:
            neural_model.plot(save=True, figname='{}'.format(self.fig_path), show=False)

        return True

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
        rsampling = RandomSampling(self.loglikelihood, pars_info=self.pars_info, nrand=self.nrand,
                                   files_path=self.grid_path)
        samples, likes = rsampling.make_dataset()
        # # samples_scaler = StandardScaler()
        # params_scaler = MinMaxScaler(feature_range=(-1, 1))
        # # fit scaler on data
        # params_scaler.fit(samples.reshape(-1, 1))
        # # apply transform
        # sc_params = params_scaler.transform(samples.reshape(-1, 1))
        # likes_scaler = MinMaxScaler(feature_range=(-1, 1))
        likes_scaler = StandardScaler()
        likes_scaler.fit(likes.reshape(-1, 1))
        # sc_likes = likes_scaler.transform(likes.reshape(-1, 1))
        print("using neural net")
        likes = np.array(likes_scaler.inverse_transform(self.neural_model.predict(params.reshape(-1,1))))
        likes = likes.reshape(len(likes), self.dims)
        return likes



