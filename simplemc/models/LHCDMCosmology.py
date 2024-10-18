from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.paramDefs import alpha_1_par
import math as N
import numpy as np

#TODO Add more DE EoS for comparison



class LHCDMCosmology(LCDMCosmology):
    """
        This is CDM cosmology with w, wa and Ok.
        CPL parameterization with curvature.
        This class inherits LCDMCosmology class as the rest of the cosmological
        models already included in SimpleMC.

        :param varyw: variable w0 parameter
        :type varyw: Boolean
        :param varywa: variable wa parameter
        :type varywa: Boolean
        :param varyOk: variable Ok parameter
        :type varyOk: Boolean

    """
    def __init__(self):
        self.alpha_1 = alpha_1_par.value
        LCDMCosmology.__init__(self)


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        l.append(alpha_1_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "alpha_1":
                self.alpha_1 = p.value
        return True


    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    # '1.0 +Omrad/a**4 +(1.0 +alpha_1)*Om*(a**(-3/(1.0+alpha_1)) -1.0)'
    def RHSquared_a(self, a):
        return 1.0 +self.Omrad/a**4 + (1.0 +self.alpha_1)*self.Om*(a**(-3/(1.0+self.alpha_1)) -1.0)
