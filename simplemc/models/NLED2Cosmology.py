from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.paramDefs import alfa_par
import math as N

#TODO Add more DE EoS for comparison



class NLED2Cosmology(LCDMCosmology):
    """
         Power Law Lagrangian

    """
    def __init__(self, disable_radiation=False):
        self.alfa = alfa_par.value
        LCDMCosmology.__init__(self)


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        l.append(alfa_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "alfa":
                self.alfa = p.value
            # elif p.name == "Ok":
            #     self.Ok = p.value
            #     self.setCurvature(self.Ok)
            #     if (abs(self.Ok) > 1.0):
            #         return False
        return True


    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        return self.Om/a**3 + self.Omrad/a**4+(1/(3*(100*self.h)**2)/(a**4)**(4*self.alfa))