

from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
import scipy.linalg as la
import scipy as sp
from simplemc import cdir


class CompressedSNLikelihood(BaseLikelihood):
    def __init__(self, name, values_filename, cov_filename):
        """
        This module calculates likelihood for the compressed SN.
        Parameters
        ----------
        name
        values_filename
        cov_filename

        Returns
        -------

        """
        BaseLikelihood.__init__(self, name)
        print("Loading ", values_filename)
        da = sp.loadtxt(values_filename)
        self.zs  = da[:, 0]
        self.mus = da[:, 1]
        print("Loading ", cov_filename)
        cov = sp.loadtxt(cov_filename, skiprows=1)
        assert(len(cov) == len(self.zs))
        vals, vecs = la.eig(cov)
        vals = sorted(sp.real(vals))
        print("Eigenvalues of cov matrix:", vals[0:3], '...', vals[-1])
        print("Adding marginalising constant")
        cov += 3**2
        self.icov = la.inv(cov)



    def loglike(self):
        tvec = sp.array([self.theory_.distance_modulus(z) for z in self.zs])
        # This is the factor that we need to correct
        # note that in principle this shouldn't matter too much, we will marginalise over this
        # Graficar
        tvec += 43
        delta = tvec-self.mus
        # |delta|
        return -sp.dot(delta, sp.dot(self.icov, delta))/2.0


class BetouleSN(CompressedSNLikelihood):
    def __init__(self):
    	
        # CompressedSNLikelihood.__init__(self, "BetouleSN", "/home/cosmocicatais/Documents/github/neuralCosmoReconstruction/notebooks/fake_binned_JLADO_inter.dat",
        #                                 "/home/cosmocicatais/Documents/github/neuralCosmoReconstruction/notebooks/fake_binned_JLA_COVAE_DO.dat")


        # CompressedSNLikelihood.__init__(self, "BetouleSN", "/home/cosmocicatais/Documents/github/neuralCosmoReconstruction/notebooks/fake_binned_JLA_inter.dat",
        #                                 "/home/cosmocicatais/Documents/github/neuralCosmoReconstruction/notebooks/fake_binned_JLA_COVAE.dat")


        # CompressedSNLikelihood.__init__(self, "BetouleSN", "simplemc/data/jla_binned_distances_31nodes_v1.txt",
        #                                 "simplemc/data/cov_jla_binned_distances_31nodes_v1.txt")
        CompressedSNLikelihood.__init__(self, "BetouleSN", cdir+"/data/jla_binned_distances_31nodes_v1.txt",
                                        cdir+"/data/cov_jla_binned_distances_31nodes_v1.txt")



class UnionSN(CompressedSNLikelihood):
    def __init__(self):
        CompressedSNLikelihood.__init__(self, "UnionSNV2", cdir+"/data/binned-sne-union21-v2.txt",
                                        cdir+"/data/binned-covariance-sne-union21-v2.txt")


class BinnedPantheon(CompressedSNLikelihood):
    def __init__(self):
        CompressedSNLikelihood.__init__(self, "BinnedPantheon", cdir+"/data/binned_pantheon_15.txt",
                                        cdir+"/data/binned_cov_pantheon_15.txt")

class NeuralSN(CompressedSNLikelihood):
    def __init__(self):
        CompressedSNLikelihood.__init__(self, "NeuralSN", cdir+"/data/sneural/sn_news.dat",
                                        cdir+"/data/sneural/sn_news_cov.dat")
        # in pantheon but not in jla
        # CompressedSNLikelihood.__init__(self, "NeuralSN", cdir+"/data/sneural/not_jla.dat",
        #                                 cdir+"/data/sneural/not_jla_cov.dat")