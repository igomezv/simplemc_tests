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
        pars_bounds
        rootname
        ndivsgrid
        hidden_layers_neurons
        epochs
        plot
    """
    def __init__(self, loglikelihood_control, rootname='neural',
                 neuralike_settings=None):
        # simplemc: reading neuralike settings
        self.nstart_samples = neuralike_settings['nstart_samples']
        self.nstart_stop_criterion = neuralike_settings['nstart_stop_criterion']
        self.ncalls_excess = neuralike_settings['ncalls_excess']
        self.updInt = neuralike_settings['updInt']
        self.ncalls_neural = 0
        self.n_neuralikes = 0
        self.train_counter = 0
        self.last_train_it = 0
        self.aft_train_it = 0
        self.loglikelihood_control = loglikelihood_control
        self.rootname = rootname
        self.neuralike_settings = neuralike_settings
        self.trained_net = False
        self.originalike_counter = 0
        self.like_name = 'Original'


    def run(self, delta_logz, it, nc, samples, likes, nsize=10,
            absdiff_criterion=None, map_fn=map, perc_tolerance=5):
        start = self.starting_flag(delta_logz, it)
        if start:
            if self.training_flag(it):
                self.train(samples, likes)
            if self.trained_net:
                self.neural_switch(nc, samples, likes, nsize=nsize,
                                   absdiff_criterion=absdiff_criterion, map_fn=map_fn,
                                   perc_tolerance=perc_tolerance)

        info = "\nneural calls: {} | neuralikes: {} | "\
               "neural trains: {} | Using: ".format(self.ncalls_neural, self.n_neuralikes,
                                                    self.train_counter)
        if self.trained_net:
            print(info+' Neural')
        else:
            if self.train_counter > 0:
                self.originalike_counter += 1
            print(info+' Original {}-aft'.format(self.originalike_counter))
        return None

    def starting_flag(self, delta_logz, it):
        if (it >= self.nstart_samples) or (delta_logz <= self.nstart_stop_criterion):
            return True
        else:
            return False

    def training_flag(self, it):
        # Number of iterations after last train
        self.aft_train_it = it - self.last_train_it
        # setting the conditions to train or retrain
        check = (self.aft_train_it % self.updInt == 0)
        retrain = (self.originalike_counter >= self.updInt)
        first = (self.train_counter == 0)
        if (check and first) or retrain:
            self.last_train_it = it
            return True
        else:
            return False

    def train(self, samples, likes):
        self.net = NeuralManager(loglikelihood=self.loglikelihood_control,
                                 samples=samples,
                                 likes=likes,
                                 rootname=self.rootname,
                                 neuralike_settings=self.neuralike_settings)
        self.net.training()
        self.train_counter += 1
        self.trained_net = self.net.valid
        self.originalike_counter = 0
        return None

    def neural_switch(self, nc, samples, likes, nsize=10, absdiff_criterion=None,
                      map_fn=map, perc_tolerance=5):
        if self.trained_net:  # validar dentro de nested
            self.n_neuralikes += 1
            self.ncalls_neural += nc
            if nc > 200:
                self.trained_net = False
                print("\nExcesive number of calls, neuralike disabled")
            elif self.n_neuralikes % (self.updInt // 2) == 0:
                print("\nTesting neuralike predictions...")
                samples_test = samples[-self.updInt:, :]
                neuralikes_test = likes[-self.updInt:]

                real_logl = np.array(list(map_fn(self.loglikelihood_control,
                                                 samples_test)))

                flag_test = self.test_predictions(samples_test, neuralikes_test, real_logl,
                                                  nsize=nsize, absdiff_criterion=absdiff_criterion,
                                                  perc_tolerance=perc_tolerance)
                self.trained_net = False
                if flag_test is False:
                    print("\nBad neuralike predictions")
        return None

    def likelihood(self, params):
        if self.trained_net:
            return self.net.neuralike(params)
        else:
            return self.loglikelihood_control(params)

    @staticmethod
    def test_predictions(x, y_pred, y_real, nsize=10, absdiff_criterion=None, perc_tolerance=5):
        if absdiff_criterion is None:
            absdiff_criterion = (1 / perc_tolerance) * np.min(np.abs(x))
        nlen = len(y_pred)
        # if y_pred.shape != y_real.shape:
        y_pred = y_pred.reshape(nlen, 1)
        y_real = y_real.reshape(nlen, 1)

        shuffle = np.random.permutation(nlen)

        y_pred = y_pred[shuffle][-nsize:]
        y_real = y_real[shuffle][-nsize:]

        absdiff = np.mean((np.abs(y_real - y_pred)))
        # diff_mean = np.mean(np.abs(y_real - y_pred))
        print("Absolute difference in the test set: {:.4f}".format(absdiff))
        # print("diff mean in test set: {:.8f}".format(diff_mean))
        print("Absolute difference criterion: {:.4f}".format(absdiff_criterion))
        if absdiff <= absdiff_criterion:
            return True
        else:
            return False