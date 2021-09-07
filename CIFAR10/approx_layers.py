import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras as keras

import sys 
sys.path.append('../')
from layer_approx_utils import *

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Further modularize to command line file
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

test_images = test_images.astype("float32")/255.0
test_labels = test_labels.astype("float32")

train_images = train_images.astype("float32")/255.0
train_labels = train_labels.astype("float32")

val_images = train_images[-10000:]
val_labels = train_labels[-10000:]
train_images = train_images[:-10000]
train_labels = train_labels[:-10000]

# random shuffling is very very important ...
np.random.seed(8)
ind = np.random.choice(train_images.shape[0], train_images.shape[0], replace=False)
train_images = train_images[ind, :]
train_labels = train_labels[ind]

loadedModel = keras.models.load_model('Lenet_cifar10_Run2')
loadedModel.predict(tf.ones((1, 32, 32, 3))) # hack to for the model to properly instantiated

N_app = 256

#layer 1 has idx = 5
epsilon_list = [0.01,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.2,0.3]
#epsilon_list = [0.14,0.16,0.2,0.3]
#epsilon_list = [0.01]
bias_NetTrim, U_dict_NetTrim = getLayerApprox(model=loadedModel, layer_idx=5, N=N_app,
                                      epsilon_list=epsilon_list, inputs=train_images,
                                      method = 'NetTrim')
np.save('approx_matrices/net_trim_layer_1_Run2_N_256.npy', U_dict_NetTrim)

epsilon_list = [0.01,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.2,0.3]
#epsilon_list = [0.14,0.16,0.2,0.3]
bias_DLR, U_dict_DLR = getLayerApprox(model=loadedModel, layer_idx=5, N=N_app,
                                  epsilon_list=epsilon_list, inputs=train_images,
                                  method = 'DLR', verbose=True)
np.save('approx_matrices/DLR_layer_1_Run2_N_256.npy', U_dict_DLR)


#layer 2 has idx = 6
epsilon_list = [0.01,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.2,0.3]
#epsilon_list = [0.14,0.16,0.2,0.3]
bias_NetTrim, U_dict_NetTrim = getLayerApprox(model=loadedModel, layer_idx=6, N=N_app,
                                      epsilon_list=epsilon_list, inputs=train_images,
                                      method = 'NetTrim')
np.save('approx_matrices/net_trim_layer_2_Run2_N_256.npy', U_dict_NetTrim)

epsilon_list = [0.01,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.2,0.3]
#epsilon_list = [0.14,0.16,0.2,0.3]
bias_DLR, U_dict_DLR = getLayerApprox(model=loadedModel, layer_idx=6, N=N_app,
                                  epsilon_list=epsilon_list, inputs=train_images,
                                  method = 'DLR', verbose=True)
np.save('approx_matrices/DLR_layer_2_Run2_N_256.npy', U_dict_DLR)