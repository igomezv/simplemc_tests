from simplemc.DriverMC import DriverMC

outputpath = '/home/isidro/Documents/gitHub/simplemc_tests/simplemc/chains/'
analyzer = DriverMC(analyzername="mcmc", model="LCDM", datasets="HD",
                    getdist=True, corner=True, simpleplot=True, showfig=True,
                    chainsdir=outputpath)
# args are related with the analyzer
analyzer.executer(nsamp=100, skip=0)
analyzer.postprocess()
