from simplemc.DriverMC import DriverMC
import numpy as np
np.random.seed(1234)
#import tensorflow as tf

# inputs = tf.keras.Input(shape=(3,))
# x = tf.keras.layers.Dense(300, activation=tf.nn.relu)(inputs)
# outputs = tf.keras.layers.Dense(1, activation=tf.nn.relu)(x)      
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

analyzer = DriverMC(analyzername="nested", model="LCDM", datasets="SN", chainsdir="chains")
					# varys8=True)

analyzer.nestedRunner(nlivepoints=100, nproc=2, accuracy=0.01)
# analyzer.nestedRunner(neuralNetwork=True, nlivepoints=500, proxy_tolerance=0.3, dlogz_start=10,	
#                       numNeurons=50, failure_tolerance=0.2, epochs=100)
analyzer.postprocess()

