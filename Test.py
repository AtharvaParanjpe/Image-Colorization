import numpy as np
import pandas as pd
import math
import random


def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y


def derivative_sigmoid(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def tanh_func(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.power(x,2)

# X = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
# y = [[0],[1],[1],[0],[1],[0],[0],[1]]

X = np.array([[5,-1], [2,1], [6,-1], [3,0], [7,-3], [21,-5], [0,2], [12,0], [15,-3], [11,2]],dtype=float)
Y = [14, 7, 17, 9, 18, 58, 2, 36, 42, 35]
y=Y


## normalization 
i = 0
for x in X:
    x[0],x[1],y[i] = x[0]/(x[0]**2+x[1]**2),x[1]/(x[0]**2+x[1]**2),y[i]/(x[0]**2+x[1]**2)
    # print(x[0],x[1])
    i+=1

lr = 0.01
b1 = np.zeros((2, 1))
b2 = np.zeros((1, 1))

weights1 = np.random.rand(2, 2)
weights2 = np.random.rand(2, 1)

for j in range(200):
    loss = 0
    for i in range(len(X)):
        # print(X[i].shape)
        hidden1 = tanh_func(np.dot(weights1.T, X[i].reshape(2,1)) + b1)
        # print(hidden1.shape)
        hidden2 = sigmoid(np.dot(weights2.T,hidden1 ) + b2)
        # print(hidden2.shape)

        loss = math.pow(hidden2 - y[i], 2)

        dZ2 = hidden2-y[i]
        dW2 = np.dot(dZ2,hidden1.T)
        db2 = np.sum(dZ2,axis=1,keepdims=True)

        dZ1 = np.dot(weights2,dZ2)*tanh_deriv(hidden1)
        dW1 = np.dot(dZ1,X[i].reshape(2,1).T)
        db1 = np.sum(dZ1,axis=1,keepdims=True)

        weights2-= lr*dW2.T
        b2-= lr*db2
        weights1-= lr*dW1
        b1-= lr*db1
    print(loss)
    # input()



hidden1 = tanh_func(np.dot(weights1.T, X[0].reshape(2,1)) + b1)
# print(hidden1.shape)
hidden2 = sigmoid(np.dot(weights2.T,hidden1 ) + b2)
# print(hidden2.shape)

hidden2 = hidden2*26
print(hidden2)