from simplemc.DriverMC import DriverMC
import numpy as np
import matplotlib.pyplot as plt
import time

"""
This script calls toy distributions from the ToyModel class and make a sampling 
for these models through dynesty with and without a neural network (based on pybambi).
"""
np.random.seed(0)

# ##### SETTINGS ###########
# modelname can be {'eggbox', 'himmel', 'ring', 'square', 'gaussian'}
show_plots = True  # choose False if you are in a server
dims = 2
nlive = 100


# ###### FIRST SAMPLING WITH ONLY DYNESTY
# sampler1 = DriverMC(analyzername='nested', model='LCDM', datasets='HD')
sampler1 = DriverMC(analyzername='nested', model='eggbox')

ti = time.time()
res1 = sampler1.executer(nlivepoints=100)
# resultnested = sampler1.results

resultnested = res1['result']['samples']
tfnested = time.time() - ti

# ###### SECOND SAMPLING WITH DYNESTY + NEURAL NETWORK
# sampler2 = NestedSampler(loglike, priorTransform, ndim=dims,
#                         bound='multi', sample='unif', nlive=nlive,
#                         pool=pool, queue_size=nworkers,
#                         use_pool={'loglikelihood': False}, neuralike=True, use)

# print("\nNext sampling:")
# ti = time.time()
# sampler2.run_nested(dlogz=0.01, outputname=modelname+"_bambi", dumper=dumper, netError=0.1)
# resultbambi = sampler2.results
# tfbambi = time.time() - ti

# ###### PRINT SUMMARY OF BOTH SAMPLING PROCESSES

# print("\n\nDynesty:")
# resultnested.summary()

# print("\n\nDynesty + ANN :")
# resultbambi.summary()
# print("\nTime dynesty: {:.4f} min | Time dynesty+ANN: {:.4f} min".format(tfnested/60, tfbambi/60 ))

# ### Plot results if you aren't in a server
# if show_plots:
#     nestdata = np.loadtxt(modelname+'_dynesty_1.txt', usecols=(2, 3))
#     # bambidata = np.loadtxt(modelname+'_bambi_1.txt', usecols=(2, 3))
#     znest = np.zeros(len(nestdata))
#
#     # for i, row in enumerate(nestdata):
#     #     znest[i] = tm.loglike(row)
#
#     # zbambi = np.zeros(len(bambidata))
#     #
#     # for i, row in enumerate(bambidata):
#     #     zbambi[i] = tm.loglike(row)
#     fig = plt.figure()
#
#     # 3D plots
#     ax = fig.add_subplot(111, projection='3d')
#
#     ax.scatter(nestdata[:, 0], nestdata[:, 1],  znest, c='red', alpha=0.5)
#     # ax.scatter(bambidata[:, 0], bambidata[:, 1], zbambi, c='green', alpha=0.5)
#     # plt.legend(["dynesty", "dynesty + neural net"],  loc="upper right")
#
#     plt.savefig(modelname+"_bambi3D.png")
#     plt.show()
#
#     # 2D plots
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.scatter(nestdata[:, 0], nestdata[:, 1], c='red', alpha=0.5)
#     # plt.scatter(bambidata[:, 0], bambidata[:, 1], c='green', alpha=0.5)
#     plt.legend(["dynesty", "dynesty + neural net"],  loc="upper right")
#     ax.set_aspect('equal', adjustable='box')
#     # plt.savefig(modelname+"_bambi2D.png")
#     plt.show()

