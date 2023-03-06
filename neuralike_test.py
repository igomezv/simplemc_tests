from simplemc.DriverMC import DriverMC

"read all setting from .ini file"
inifileANN = "neuralikeConfig_ANN.ini"
inifileGA = "neuralikeConfig_GA.ini"
inifileANNGA = "neuralikeConfig_ANN_GA.ini"


analyzer = DriverMC(iniFile=inifileANN)
analyzer.executer()
#analyzer.postprocess()

#analyzer = DriverMC(iniFile=inifileGA)
#analyzer.executer()

#analyzer = DriverMC(iniFile=inifileANNGA)
#analyzer.executer()

""" useful for short tests,
    or when just a few settings customize it from here"""
#analyzer = DriverMC(analyzername="nested", model="LCDM", datasets="HD")
#analyzer.executer(nlivepoints=10)
#analyzer.postprocess()
