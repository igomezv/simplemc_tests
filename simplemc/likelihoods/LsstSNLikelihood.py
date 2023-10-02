from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
import sys
import numpy as np
import scipy.linalg as la
from scipy.interpolate import interp1d
from simplemc import cdir


class LsstSNLikelihood(BaseLikelihood):
    """
            This module calculates likelihood for the compressed SN.

            Parameters
            ----------
            name : str
                Name of the dataset
            values_filename : str
                File text with the observational data.
            cov_filename : str
                File text with the covariance matrix of the observational data.
    """
    def __init__(self, name, values_filename, cov_filename, ninterp=150, dataset_type='large_cov'):
        # dataset_type can be 'short', 'large', 'large_cov'
        self.name_ = name
        BaseLikelihood.__init__(self, name)
        print("Loading", values_filename)
        da = np.loadtxt(values_filename, skiprows=5)
        self.zcmb  = da[:, 0]
        self.mag = da[:, 2]
        self.dmag = da[:, 3]
        self.N = len(self.mag)
        print("Data points: ", self.N)
        if dataset_type == 'short':
            self.syscov = np.loadtxt(cov_filename, skiprows=0).reshape((self.N, self.N))
        elif dataset_type == 'large':
            self.syscov = np.diag(np.loadtxt(cov_filename, skiprows=0))
        elif dataset_type == 'large_cov':
            self.syscov = np.loadtxt(cov_filename, skiprows=0).reshape((self.N, self.N))
        else:
            sys.exit("Dataset type not available.")
        print("COV MATRIX SHAPE:", np.shape(self.syscov))
        # print(self.syscov[1, :])
        self.cov = np.copy(self.syscov)
        self.cov[np.diag_indices_from(self.cov)] += self.dmag ** 2
        self.xdiag = 1 / self.cov.diagonal()  # diagonal before marginalising constant
        # add marginalising over a constant, check it!
        self.cov += 3 ** 2
        self.zmin = self.zcmb.min()
        self.zmax = self.zcmb.max()
        self.zmaxi = 1.1  ## we interpolate to 1.1 beyond that exact calc
        self.zinter = np.linspace(1e-3, self.zmaxi, ninterp)
        self.icov = la.inv(self.cov)

    def loglike(self):
        # we will interpolate distance
        dist = interp1d(self.zinter, [self.theory_.distance_modulus(z) for z in self.zinter],
                        kind='cubic', bounds_error=False)(self.zcmb)
        # tvec = sp.array([self.theory_.distance_modulus(z) for z in self.zs])
        who = np.where(self.zcmb > self.zmaxi)
        dist[who] = np.array([self.theory_.distance_modulus(z) for z in self.zcmb[who]])
        tvec = self.mag - dist

        # tvec = self.mag-np.array([self.theory_.distance_modulus(z) for z in self.zcmb])
        # print (tvec[:10])
        # first subtract a rought constant to stabilize marginaliztion of
        # intrinsic mag.
        tvec -= (tvec * self.xdiag).sum() / (self.xdiag.sum())
        # print(tvec[:10])
        chi2 = np.einsum('i,ij,j', tvec, self.icov, tvec)
        # print("chi2=",chi2)
        return -chi2 / 2


class SN_photo(LsstSNLikelihood):
    """
    Likelihood to binned photo dataset.
    """
    def __init__(self):
        LsstSNLikelihood.__init__(self, "SNlsstphoto", cdir+"/data/Data_SNIa_LSST/hubble_diagram_Pr.txt",
                                        cdir+"/data/Data_SNIa_LSST/covsys_000_P.txt")


class SN_spec(LsstSNLikelihood):
    """
    Likelihood to binned spec dataset.
    """
    def __init__(self):
        LsstSNLikelihood.__init__(self, "SNlsstspec", cdir+"/data/Data_SNIa_LSST/hubble_diagram_Sr.txt",
                                        cdir+"/data/Data_SNIa_LSST/covsys_000_S.txt")

class SN_large(LsstSNLikelihood):
    """
    Likelihood to binned spec dataset.
    """
    def __init__(self):
        LsstSNLikelihood.__init__(self, "SNlsstLarge", cdir+"/data/Data_SNIa_LSST/lsst_large_hubble_diagram.txt",
                                        cdir+"/data/Data_SNIa_LSST/covsys_000.txt")
                                        # cdir+"/data/Data_SNIa_LSST/large_cov.dat")

