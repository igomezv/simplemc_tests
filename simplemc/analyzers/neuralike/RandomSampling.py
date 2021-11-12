import os
import numpy as np
import time


class RandomSampling:
    def __init__(self, like, pars_info, nrand=500, pool=None, files_path='randomsampling'):
        """
        Create a random samples in the parameter space and evaluate the likelihood in them.
        This is used to generate the training set for a neural network.

        Parameters
        ----------
        like: likelihood object
        pars: list of Parameter objects
        nrand: number of random points in the parameter space. Default is 500
        """
        self.like = like

        self.means = np.array([p.value for p in pars_info])
        self.paramsList = [p.name for p in pars_info]
        self.dims = len(self.paramsList)

        # squared_errors = [((p.bounds[1] - p.bounds[0])*2)**2 for p in pars_info]
        squared_errors = [min(np.abs(p.value - p.bounds[1]), np.abs(p.value - p.bounds[0]))**2 for p in pars_info]
        self.cov = np.diag(squared_errors)

        # self.bounds = [p.bounds for p in pars_info]

        self.nrand = nrand
        self.pool = pool
        self.files_path = files_path
        if pool:
            self.M = pool.map
        else:
            self.M = map
        print("Generating a random sample of points in the parameter space...")

    def make_sample(self):
        if not self.filesChecker():
            samples = np.random.multivariate_normal(self.means, self.cov, size=(self.nrand, ))
        else:
            print('Loading existing random_samples and likelihoods: {}'.format(self.files_path))
            samples = np.load('{}_random_samples.npy'.format(self.files_path))
        print("cov, means, samples", np.shape(self.cov), np.shape(self.means), np.shape(samples))
        print("Random samples in the parameter space generated!")
        return samples


    def make_dataset(self):
        """
        Evaluate the Likelihood function on the grid
        Returns
        -------
        Random samples in the parameter space and their respectives likelihoods.
        """
        samples = self.make_sample()
        t1 = time.time()
        if not self.filesChecker():
            print("Evaluating likelihoods...")
            likes = np.array(list(self.M(self.like, samples)))
            idx_nan = np.argwhere(np.isnan(likes))
            likes = np.delete(likes, idx_nan)
            samples = np.delete(samples, idx_nan, axis=0)
            np.save('{}_random_samples.npy'.format(self.files_path), samples)
            np.save('{}_likes.npy'.format(self.files_path), likes)
        else:
            print('Loading existing random samples and likelihoods: {}'.format(self.files_path))
            likes = np.load('{}_likes.npy'.format(self.files_path))
        # likes = np.array([self.like(x) for x in samples_grid])

        tf = time.time() - t1
        print("Time of {} likelihood evaluations {:.4f} min".format(len(likes), tf/60))
        print("Training dataset created!")
        print("cov\n", self.cov)
        print("samples\n", np.shape(samples))
        print("likes\n", np.shape(likes))
        print("shapes like, samples", np.shape(likes), np.shape(samples))
        if self.pool:
            self.pool.close()
        # print("Time of evaluating {} likelihoods with apply_along_axis: {:.4} s".format(len(likes), tf))

        return samples, likes

    def filesChecker(self):
        """
        This method checks if the name of the random_samples.npy and likes.npy exists, if it already exists use it
        """
        if os.path.isfile('{}_random_samples.npy'.format(self.files_path)):
            if os.path.isfile('{}_likes.npy'.format(self.files_path)):
                return True
        else:
            return False

