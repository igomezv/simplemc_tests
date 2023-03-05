"""neuralike management object.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: Jun 2022
"""
import numpy as np
from simplemc.analyzers.neuralike.NeuralManager import NeuralManager

class NeuraLike:
    """
        Manager for neural networks to learn likelihood function over a grid
        Parameters
        -----------
        loglikelihood
        rootname
    """
    def __init__(self, loglikelihood_control, rootname='neural',
                 neuralike_settings=None):
        # simplemc: reading neuralike settings
        self.nstart_samples = neuralike_settings['nstart_samples']
        self.nstart_stop_criterion = neuralike_settings['nstart_stop_criterion']
        self.ncalls_excess = neuralike_settings['ncalls_excess']
        self.updInt = neuralike_settings['updInt']

        self.loglikelihood_control = loglikelihood_control
        # Counters
        self.ncalls_neural = 0
        self.n_neuralikes = 0
        self.train_counter = 0
        self.originalike_counter = 0
        self.trained_net = False
        self.net = None
        self.rootname = rootname
        self.neuralike_settings = neuralike_settings

    def run(self, delta_logz, it, nc, samples, likes, logl_tolerance=0.05):
        if self.training_flag(delta_logz, it):
            self.train(samples, likes)
        if self.trained_net:
            self.neural_switch(nc, samples, likes, logl_tolerance=logl_tolerance)

        info = "\nneural calls: {} | neuralikes: {} | "\
               "neural trains: {} | Using: ".format(self.ncalls_neural, self.n_neuralikes,
                                                    self.train_counter)
        if self.trained_net:
            print(info+'Neuralike')
        else:
            if self.train_counter > 0:
                self.originalike_counter += 1
            print(info+'Original_fn {}-aft'.format(self.originalike_counter))
        return None

    def training_flag(self, delta_logz, it):
        start_it = (it >= self.nstart_samples)
        startlogz = (delta_logz <= self.nstart_stop_criterion)
        if start_it or startlogz:
            # setting the conditions to train or retrain
            retrain = (self.originalike_counter >= self.updInt//2) and (self.train_counter > 0)
            first = (self.train_counter == 0)
            # if first or retrain:
            if retrain or first:
                self.last_train_it = it
                return True
            else:
                return False
        else:
            return False

    def train(self, samples, likes):
        self.net = NeuralManager(loglikelihood=self.loglikelihood_control,
                                 samples=samples,
                                 likes=likes,
                                 rootname=self.rootname,
                                 n_train = self.train_counter,
                                 neuralike_settings=self.neuralike_settings)
        self.net.training(samples, likes)
        self.train_counter += 1
        self.trained_net = self.net.valid
        self.originalike_counter = 0
        return None

    def neural_switch(self, nc, samples, likes, logl_tolerance=0.05):
        self.n_neuralikes += 1
        self.ncalls_neural += nc
        if nc > 1000:
            self.trained_net = False
            print("\nExcesive number of calls, neuralike disabled")
        neuralike = likes[-1:]
        # like in BAMBI paper, with sigma (tolerance) = 0.
        if (np.min(likes) - logl_tolerance < neuralike) and (neuralike < np.max(likes) + logl_tolerance):
            self.trained_net = True
        else:
            print("\nBad neuralikes predictions!")
            self.trained_net = False
    def likelihood(self, params):
        if self.trained_net:
            return self.net.neuralike(params)
        else:
            return self.loglikelihood_control(params)