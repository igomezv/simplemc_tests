"""neuralike management object.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: Jun 2022
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

    # def __init__(self, loglikelihood, samples, likes,
    #              rootname='neural', neuralike_settings=None):
    def __init__(self, loglikelihood, min_live_logL, max_live_logL, rootname='neural',
                 pool=None, neuralike_settings=None):
        self.loglikelihood_fn = loglikelihood
        self.valid = False
        self.fig_path = '{}.png'.format(rootname)
        self.min_live_logL = min_live_logL
        self.max_live_logL = max_live_logL
        self.pool = pool
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
            # self.rmse_criterion = neuralike_settings['rmse_criterion']
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
        self.mse_criterion = self.valid_loss

        # self.model_path = '{}.h5'.format(rootname)
        # if not self.modelChecker():
        #     self.training()
        # else:
        #     self.neural_model = self.load()

    def training(self, samples, likes):
        _, self.dims = np.shape(samples)
        self.topology = [self.dims] + self.hidden_layers_neurons + [1]
        # print(likes)
        # print("minl, maxl", self.min_live_logL, self.max_live_logL)
        # idx_val_logL1 = np.argwhere(likes > self.min_live_logL)
        # idx_val_logL2 = np.argwhere(likes < self.max_live_logL)
        # idx_val_logL = np.intersect1d(idx_val_logL1, idx_val_logL2)

        # likes = likes[idx_val_logL]
        # samples = samples[idx_val_logL]
        print(np.shape(likes), np.shape(samples), type(likes), type(samples))

        ml_idx = np.argmax(likes)
        means = samples[ml_idx, :]
        print("\nGenerating training set...")
        # rsampling = RandomSampling(self.loglikelihood_fn, means=means,
        #                            cov=np.cov(samples.T),
        #                            nrand=self.nrand, pool=self.pool)
        rsampling = RandomSampling(self.loglikelihood_fn, mins=np.min(samples, axis=0),
                                   maxs=np.max(samples, axis=0),
                                   nrand=self.nrand, pool=self.pool)
        #
        rsamples, rlikes = rsampling.make_dataset()
        print("Training dataset created!")
        likes = np.append(rlikes, likes)
        samples = np.append(rsamples, samples, axis=0)
        print("\nGenerating test set...")
        # rsampling_test = RandomSampling(self.loglikelihood_fn,
        #                                 means=means,
        #                                 cov=np.cov(samples.T),
        #                                 nrand=int(0.1*len(likes)))
        # rsamples_test, rlikes_test = rsampling_test.make_dataset()
        print("Test dataset created!")

        ## scale params
        # self.samples_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        self.samples_scaler = StandardScaler(with_mean=False)
        self.samples_scaler.fit(samples)
        sc_samples = self.samples_scaler.transform(samples)
        print(sc_samples)
        # # create scaler
        self.likes_scaler = MinMaxScaler(feature_range=(0.5, 1))
        self.likes_scaler.fit(likes.reshape(-1, 1))
        # # # apply transforms
        sc_likes = self.likes_scaler.transform(likes.reshape(-1, 1))

        self.neural_model = NeuralNet(X=sc_samples, Y=sc_likes, topology=self.topology,
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      learrning_rate=self.learning_rate,
                                      patience=self.patience,
                                      minsample=np.min(np.abs(samples)))

        self.neural_model.train()
        # neural_model.save_model('{}'.format(self.model_path))
        if self.plot:
            self.neural_model.plot(save=True, figname='{}'.format(self.fig_path), show=False)
        lastval = self.neural_model.loss_val[-1]
        lasttrain = self.neural_model.loss_train[-1]
        ## preparing for testing
        # y_pred_test = self.neural_model.predict(rsamples_test)
        # y_pred_test = self.likes_scaler.inverse_transform(y_pred_test.reshape(-1, 1))
        # test_set_test = self.test_neural(y_pred=y_pred_test, y_real=rlikes_test, nfrac=1)
        test_mse = self.neural_model.test_mse()
        # test_rmse = np.sqrt(test_mse)
        # criterion = float(self.likes_scaler.transform(np.array([self.rmse_criterion]).reshape(1, 1)))
        print("Test MSE: {:.4f} | Criterion: {:.4f}".format(test_mse, self.mse_criterion))
        if test_mse < self.mse_criterion:
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
        # sc_params = self.samples_scaler.transform(np.array(params).reshape(len(params), self.dims))
        # print(params.shape)
        sc_params = self.samples_scaler.transform(np.array(params).reshape(1, self.dims))
        # pred = self.neural_model.predict(sc_params)
        pred = self.neural_model.predict(sc_params)
        # likes = np.array(pred)
        likes = self.likes_scaler.inverse_transform(pred.reshape(-1, 1))
        likes = np.array(likes)
        return likes

    def test_neural(self, y_pred, y_real, nsize=10, absdiff_criterion=5):
        nlen = len(y_pred)
        # if y_pred.shape != y_real.shape:
        y_pred = y_pred.reshape(nlen, 1)
        y_real = y_real.reshape(nlen, 1)

        shuffle = np.random.permutation(nlen)

        y_pred = y_pred[shuffle][-nsize:]
        y_real = y_real[shuffle][-nsize:]

        absdiff = np.mean((np.abs(y_real - y_pred)))
        # diff_mean = np.mean(np.abs(y_real - y_pred))
        print("Abs diff in test set: {:.8f}".format(absdiff))
        # print("diff mean in test set: {:.8f}".format(diff_mean))
        print("abs diff criterion", absdiff_criterion)
        if absdiff < absdiff_criterion:
            return True
        else:
            return False