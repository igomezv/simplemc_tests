from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.paramDefs import Ode_par
import math as N
import numpy as np

#TODO Add more DE EoS for comparison



class PEDECosmology(LCDMCosmology):
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
    def __init__(self, varyOde=True):
        self.varyOde  = varyOde
        self.Ode = Ode_par.value
        LCDMCosmology.__init__(self)


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        if (self.varyOde):  l.append(Ode_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "Ode":
                self.Ode = p.value
        return True


    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        term1 = self.Om/a**3
        term2 = self.Ode * (1 - np.tanh(np.log10(1/a)))
        return term1+term2
