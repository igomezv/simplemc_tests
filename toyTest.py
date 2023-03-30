from simplemc.DriverMC import DriverMC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'tkagg')
import time

"""
This script calls toy distributions from the ToyModel class and make a sampling 
for these models through dynesty with and without a neural network (geneuralike based on pybambi).
"""
np.random.seed(0)

# ##### SETTINGS ###########
show_plots = True  # choose False if you are in a server
dims = 2
# modelname can be {'eggbox', 'himmel', 'ring', 'square', 'gaussian'}
modelname = 'eggbox'
# ###### FIRST SAMPLING WITH ONLY DYNESTY
# sampler1 = DriverMC(analyzername='nested', model='LCDM', datasets='HD')
sampler1 = DriverMC(analyzername='nested', model=modelname)

ti = time.time()
res1 = sampler1.executer(useNeuralike=False, useGenetic=False, nlivepoints=500)

samplesnested = res1['result']['samples']
loglikes = res1['result']['loglikes']
tfnested = time.time() - ti

# ###### SECOND SAMPLING WITH DYNESTY + NEURAL NETWORK
sampler2 = DriverMC(analyzername='nested', model=modelname)
ti = time.time()
res2 = sampler2.executer(useNeuralike=True, useGenetic=False, nlivepoints=500,
                         valid_loss=0.5, nstart_samples=200000,
                         nstart_stop_criterion=100,
                         updInt=1000, ncalls_excess=1000, learning_rate = 0.0001,
                         epochs=100, batch_size=2, patience=100)
# for all except himmel lr = 0.001 and 100 epochs, bs=16m 1000 lp

samplesneuralike = res2['result']['samples']
neuralloglikes = res2['result']['loglikes']
tfneural = time.time() - ti

# ### Plot results if you aren't in a server
if show_plots:
    znest = np.zeros(len(samplesnested))

    # 3D plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samplesnested[:, 0],samplesnested[:, 1],  loglikes, c='red', alpha=0.5)
    plt.title("Nested sampling")
    plt.savefig(modelname + "_nested3D.png")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samplesneuralike[:, 0],samplesneuralike[:, 1],  neuralloglikes, c='green', alpha=0.5)
    plt.title("Nested sampling + neural net")
    plt.savefig(modelname+"_neuralike3D.png")

    # # 2D plots
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(samplesnested[:, 0], samplesnested[:, 1], c='red', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Nested sampling")
    plt.savefig(modelname + "_nested2D.png")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(samplesneuralike[:, 0], samplesneuralike[:, 1], c='green', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Nested sampling + neural net")
    plt.savefig(modelname+"_neuralike2D.png")
