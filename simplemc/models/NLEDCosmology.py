from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.paramDefs import b_par, beta_par
import math as N

#TODO Add more DE EoS for comparison



class NLEDCosmology(LCDMCosmology):
    """
        Racional Lagrangian

    """
    def __init__(self, disable_radiation=False):
        self.b = b_par.value
        self.beta = beta_par.value
        # self.wa = wa_par.value
        LCDMCosmology.__init__(self)


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        l.append(b_par)
        l.append(beta_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "b":
                self.b = p.value
            elif p.name == "beta":
                self.beta = p.value
            # elif p.name == "Ok":
            #     self.Ok = p.value
            #     self.setCurvature(self.Ok)
            #     if (abs(self.Ok) > 1.0):
            #         return False
        return True


    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        return self.Ocb/a**3 + self.Omrad/a**4+(4*self.b*((self.Omrad/a**4)/(((24*self.beta*(self.h*100)**2)/a**4)+1)))
                # Ocb/a**3+self.Ok/a**2+self.Omrad/a**4+NuContrib+(1.0-self.Om-self.Ok)*rhow)
