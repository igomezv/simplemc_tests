import numpy as np
from simplemc.cosmo.Parameter import Parameter
class ToyModels:
    def __init__(self, model):
        """
        This class contains some toy models to test nested samplers

        Parameters
        ----------
        model : str
            {'eggbox', 'ring', 'gaussian', 'square', 'himmel'}
        """
        # self.bounds contains x and y bounds.
        if model == 'eggbox':
            self.bounds = [(0., np.round(10*np.pi, 4)), (0., np.round(10*np.pi, 4))]
            self.loglike = self.eggLoglike
            self.dims = 2

        elif model in ['ring', 'gaussian', 'himmel', 'square']:
            self.bounds = [(-5., 5.), (-5., 5.)]
            if model == 'ring':
                self.loglike = self.ringLoglike
                self.dims = 2
            elif model == 'gaussian':
                self.loglike = self.gaussLoglike
                self.dims = 2
            elif model == 'himmel':
                self.loglike = self.himmelLoglike
                self.dims = 3
            elif model == 'square':
                self.loglike = self.squareLoglike
                self.dims = 2

    def eggLoglike(self, cube):
        x, y = cube
        return (2+np.cos(x/2.0)*np.cos(y/2.0))**5.0

    def himmelLoglike(self, cube):
        return -(cube[0] ** 2 + cube[1] - 11) ** 2.0 - (cube[0] + cube[1] ** 2 - 7) ** 2

    def gaussLoglike(self, x):
        return -((x[0]) ** 2 + (x[1]) ** 2 / 2.0 - 1.0 * x[0] * x[1]) / 2.0

    def ringLoglike(self, x):
        r2 = x[0] ** 2 + x[1] ** 2
        return -(r2 - 4.0) ** 2 / (2 * 0.5 ** 2)

    def squareLoglike(self, x):
        sq = 1.
        if abs(x[0]) > 5 or abs(x[1]) > 5:
            sq = 0.
        return sq

    def freeParameters(self):
        x = Parameter('x', 0.5,  0.5,   (0.1, 1.0),    'x')
        y = Parameter('y', 0.5, 0.5, (0.1, 1.0), 'y')
        return [x, y]

    def printFreeParameters(self):
        print("Free parameters:")
        self.printParameters(self.freeParameters())
    def printParameters(self, params):
        l = []
        for p in params:
            print(p.name, '=', p.value, '+/-', p.error)
            l.append("{}: {} = +/- {}".format(p.name, p.value, p.error))
        return l

    def updateParams(self, pars):
        for p in pars:
            if p.name == "x":
                self.x = p.value
            elif p.name == "y":
                self.y = p.value
        return True

    def loglike_wprior(self, cube):
        return self.loglike(cube) + self.theory_.prior_loglike(cube)

    def theory_loglike_prior(self, cube):
        return cube*0
    def name(self):
        return "toy model"
    # def priorTransform(self, theta):
    #     priors = []
    #     for c, bound in enumerate(self.bounds):
    #         priors.append(theta[c] * (bound[1] - bound[0]) + bound[0])
    #     return np.array(priors)










