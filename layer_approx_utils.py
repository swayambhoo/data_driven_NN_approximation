### this file contains various ways to approximate the layer

import cvxpy as cvx
import numpy as np 
from numpy import linalg as LA
from cvxpy import * 
import gurobipy 
import numpy.matlib
import tensorflow.keras as keras
import tensorflow as tf

def NetTrimApprox(X, Y, b, epsilon_list, verbose=False): 
    N,m = X.shape
    n = Y.shape[1]
    b_temp = np.expand_dims(b, axis=1)
    bb = np.matlib.repmat(b_temp.T, N, 1)
    const = np.linalg.norm(X,'fro')

    # Mask Matrices 
    Mpos = np.zeros((N,n))
    Mneg = np.zeros((N,n))

    Mpos[Y>0] = 1 
    Mneg[Y<=0] = 1

    U_dict = {}
    for epsilon in epsilon_list:
        U = Variable((m,n))
        obj = cvx.Minimize(cvx.mixed_norm(U, 1,1))
        constraints = [cvx.norm(cvx.multiply(X @ U + bb - Y, Mpos), 'fro') <= epsilon * const, cvx.multiply(X @ U + bb - Y, Mneg) <= 0]
        prob = cvx.Problem(obj, constraints)
        print("Optimal value", prob.solve(solver=cvx.GUROBI, verbose=verbose, max_iters = 10000))
        U_dict[epsilon] = U.value
    
    return U_dict

def DLRApprox(X, Y, b, epsilon_list, verbose=False):
    N,m = X.shape
    n = Y.shape[1]
    b_temp = np.expand_dims(b, axis=1)
    bb = np.matlib.repmat(b_temp.T, N, 1)
    const = np.linalg.norm(X,'fro')

    # Mask Matrices 
    Mpos = np.zeros((N,n))
    Mneg = np.zeros((N,n))

    Mpos[Y>0] = 1 
    Mneg[Y<=0] = 1

    U_dict = {}
    for epsilon in epsilon_list:
        U = Variable((m,n))
        obj = cvx.Minimize(cvx.norm(U,'nuc'))
        constraints = [cvx.norm(cvx.multiply(X @ U + bb - Y, Mpos), 'fro') <= epsilon * const, cvx.multiply(X @ U + bb - Y, Mneg) <= 0]
        prob = cvx.Problem(obj, constraints)
        print("Optimal value", prob.solve(solver=cvx.SCS, verbose=verbose, max_iters = 10000))
        U_dict[epsilon] = U.value
    
    return U_dict


def GetApproxTrainData(model, N, idx, inputs):
    desired_layer = model.get_layer(index=idx)
    layer_ip_op = keras.Model(inputs=model.input, outputs=[desired_layer.input,desired_layer.output]) 
    X, Y = layer_ip_op(inputs[:N])
    X, Y = X.numpy(), Y.numpy()
    return X, Y

def getLayerBias(model, idx):
    desired_layer = model.get_layer(index=idx)
    return desired_layer.bias.numpy()



def getLayerApprox(model, layer_idx, N, epsilon_list, inputs, method, verbose=False):
    X, Y = GetApproxTrainData(model, N, layer_idx, inputs)
    bias = getLayerBias(model, layer_idx)

    U_dict = {}
    if method=='NetTrim':
        U_dict = NetTrimApprox(X, Y, bias, epsilon_list=epsilon_list, verbose=verbose)
    
    if method=='DLR':
        U_dict = DLRApprox(X, Y, bias, epsilon_list=epsilon_list, verbose=verbose)


    return bias, U_dict


class lowrankLayer(keras.layers.Layer):
    def __init__(self, n_input, n_output, rank):
        super(lowrankLayer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.U = tf.Variable(initial_value=w_init(shape=(n_input, rank)), 
                        dtype="float32",
                        trainable=True)
        self.V = tf.Variable(initial_value=w_init(shape=(rank, n_output)),
                        dtype="float32",
                        trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(n_output,)), 
                             dtype="float32", 
                             trainable=True)
    
    def call(self, x):
        x = tf.matmul(x, self.U) 
        return tf.nn.relu(tf.matmul(x, self.V) + self.b)
    
    
class sparseLayer(keras.layers.Layer):
    def __init__(self, n_input=None, n_output=None, Wip=None, Mip=None, bias_ip=None):
        
        super(sparseLayer, self).__init__()
        
        if Wip is None:
            w_init = tf.random_normal_initializer()
            self.W = tf.Variable(initial_value=w_init(shape=(n_input,n_output)),
                                dtype="float32",
                                trainable=True)
        else:
            self.W = tf.Variable(initial_value=Wip, 
                                dtype="float32",
                                trainable=True)
                    
        if Mip is None:
            self.M = tf.Variable(intial_value = tf.ones(shape=(n_input,n_output), dtype="float32"),
                                     dtype="float32",
                                     trainable=False)
        else:
            self.M = tf.Variable(initial_value = Mip,
                                 dtype="float32",
                                 trainable=False)
        
        if bias_ip is None:
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(initial_value=b_init(shape=(n_output,)), 
                             dtype="float32", 
                             trainable=True)
        else:
            self.b = tf.Variable(initial_value=bias_ip, 
                             dtype="float32", 
                             trainable=True)
        
    def call(self, x):
        x = tf.matmul(x, tf.multiply(self.M, self.W)) + self.b
        return tf.nn.relu(x)