from .Hyperparameter import Hyperparameter
import numpy as np

deep = Hyperparameter(name='deep', values=np.array([2, 3, 4]), val=3)
num_units = Hyperparameter(name='num_units', values=np.array([50, 100, 200]), val=100)
batch_size = Hyperparameter(name='batch_size', values=np.array([16, 32, 64]), val=16)
learning_rate = Hyperparameter(name='learning_rate', values=np.array([1e-2, 1e-3, 1e-4]), val=1e-3)
epochs = Hyperparameter(name='epochs', values=np.array([50, 100, 200]), val=50)
act_fn = Hyperparameter(name='act_fn', values=['relu', 'sigmoid', 'tanh'], val='relu')
last_act_fn = Hyperparameter(name='last_act_fn', values=['relu', 'sigmoid', 'linear'], val='linear')
loss_fn = Hyperparameter(name='loss_fn', values=['categorical_crossentropy', 'mean_squared_error'], val='mean_squared_error')
