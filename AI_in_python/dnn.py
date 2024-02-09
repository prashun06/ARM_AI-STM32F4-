import numpy as np


# only for two layers
def init_parameters(num_X, num_H, num_Y):
    np.random.seed(1)  # generate random number after every single number

    w1 = np.random.randn(num_H, num_X) * 0.01
    w2 = np.random.randn(num_Y, num_H) * 0.01

    b1 = np.zeros((num_H,1))
    b2 = np.zeros((num_Y,1))

    parameters = {"W1": w1, "W2": w2, "B1": b1, "B2": b2}  # make a dictanary of output
    return parameters

# use this for multiple layers
def initialize_parameters_multi_layer(layer_dims):
    np.random.seed(1)
    parameters = {}  # empty
    L = len(layer_dims)  # number of layers (12288, 20, 8, 6, 1)= 5 layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        ##/ np.sqrt(layer_dims[l - 1]): After generating therandom values, they are divided by the square root of the number
        # of neurons in the previous layer (layer_dims[l - 1]). This is a common practice called "Xavier" or "Glorot" initialization.
        # It helps in initializing the weights in a way that prevents the vanishing or exploding gradient problem during training.

        parameters['B' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


# Forward activation function.................................
def Linear_forword(A, W, B):
    Z = W.dot(A) + B  # algo of linear forward, A is the input here
    cache = (A, W, B)  # store them/ linear cache
    return Z, cache


def ReLu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def Sigmoid(Z):
    np.seterr(over='ignore')  # ignor the overflow, should not use
    A = 1.0 / (1 + np.exp(-Z))
    cache = Z  # sigmoid cache
    return A, cache


# forword propagetion oparetion................................
def linear_activation_forword(A_prev, W, B, activation_func):
    if activation_func == "Sigmoid":
        Z, line_cache = Linear_forword(A_prev, W, B)
        A, activation_cache = Sigmoid(Z)

    elif activation_func == "ReLu":
        Z, line_cache = Linear_forword(A_prev, W, B)
        A, activation_cache = ReLu(Z)

    cache = (line_cache, activation_cache)
    return A, cache

#  cost function ........................
def compute_cost(yHat, Y):
    m = Y.shape[1]
    cost = (1 / m) * (-np.dot(Y, np.log(yHat).T) - np.dot(1 - Y, np.log(1 - yHat).T))
    cost = np.squeeze(cost)
    return cost


# Backword activation function......................
def linear_backword(dZ, cache):
    A_prev, W, B = cache
    m = A_prev.shape[1]  # number of training example

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)  # dimention will not change
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def relu_backword(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # convert dZ to correct object
    dZ[Z <= 0] = 0  # if dZ is less then or equal to 0 then dz =0
    return dZ


def sigmoid_backword(dA, cache):
    Z = cache
    s = 1.0 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def linear_activation_backward(dA, cache, activation_func):
    line_cache, activation_cache = cache
    if activation_func == "ReLu":
        dZ = relu_backword(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, line_cache)

    elif activation_func == "Sigmoid":
        dZ = sigmoid_backword(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, line_cache)

    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2  # round to nearest number/number of layers, calculates the floor division of the length,
    # floor division operator. It divides the left operand by the right operand and rounds down the result to the nearest whole number
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate*grads['dW' + str(l + 1)]
        parameters['B' + str(l + 1)] = parameters['B' + str(l + 1)] - learning_rate*grads['db' + str(l + 1)]

    return parameters


# complete Forward Propagetion.................................................

def model_forward_prop(X, parameters):
    caches = []
    A = X  # examples
    L = len(parameters) // 2  # length of layers
    # hidden layer
    for l in range(1, L):  # until L-1
        A_prev = A
        A, cache = linear_activation_forword(A_prev, parameters['W' + str(l)], parameters['B' + str(l)],
                                             activation_func="ReLu")
        caches.append(cache)
    # last layer
    AL, cache = linear_activation_forword(A, parameters['W' + str(L)], parameters['B' + str(L)],
                                          activation_func="Sigmoid")
    caches.append(cache)
    return AL, caches


# complete backward propagetion.................................................
def model_back_prop(AL,Y,caches):
    grads ={}
    L  = len(caches)
    Y = Y.reshape(AL.shape)  # hare AL is Yhat

    dAL = -(np.divide(Y,AL) - np.divide(1-Y, 1-AL) )  # main equation
    current_cache = caches[L-1]

    # bring output value weight and bias and put them in gradient,     this here for only output layer which is sigmoid
    grads["dA" + str(L)],grads["dW" + str(L)],grads["db" + str(L)] =linear_activation_backward(dAL,current_cache, activation_func ="Sigmoid")

    # all other previous layers with relu
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,activation_func ="ReLu")
        grads["dA"+str(l+1)] = dA_prev_temp  # same process
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp

    return grads


def predict(X, Y, parameters):
    m = X.shape[1]  # number of examples
    n = len(parameters)//2  # number of layers
    p = np.zeros((1, m))

    probabilities, caches = model_forward_prop(X, parameters)

    for i in range(0, probabilities.shape[1]):
        if probabilities[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy" + str(np.sum((p == Y) / m)))
    return p
