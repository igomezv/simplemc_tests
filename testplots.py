from simplemc.DriverMC import DriverMC

# "read all setting from .ini file"
# inifile = "baseConfig.ini"
#
# analyzer = DriverMC(iniFile=inifile)
# analyzer.executer()
# #analyzer.postprocess()


""" useful for short tests,
    or when just a few settings customize it from here"""
analyzer = DriverMC(analyzername="nested", model="LCDM", datasets="HD",
                    chainsdir='/home/isidro/Documents/gitHub/simplemc_tests/simplemc/chains/')
analyzer.executer(nlivepoints=10)
analyzer.postprocess()
