from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.paramDefs import b_par


class NLEDCosmology(LCDMCosmology):
    def __init__(self, varyb=True):
        """
        This is a CDM cosmology with constant eos w for DE
        Parameters
        ----------
        varyw

        Returns
        -------

        """

        self.varyb = varyb
        self.b = b_par.value
        LCDMCosmology.__init__(self)


    # my free parameters. We add b on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        if (self.varyb): l.append(b_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "b":
                self.b = p.value
        return True


    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        NuContrib = self.NuDensity.rho(a)/self.h**2
        beta = 1.0
        H_0 = 100*self.h
        num_term = self.b*self.Omrad/a**4
        den_term = (24*beta*H_0**2*self.Omrad/a**4)+1
        return self.Ocb/a**3+self.Omrad/a**4+NuContrib + num_term/den_term
