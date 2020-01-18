import numpy as np
import pandas as pd
import math
import random

class ImageColorizer:
    def __init__(self,numHiddenLayers,epochs,neuronsEachLayer,inputVal,outputVal,learningRate):
        self._numHiddenLayers = numHiddenLayers
        self._neuronsEachLayer = neuronsEachLayer
        self._epochs = epochs
        self._weights, self._bias = self.initialize_weights()
        self._x,self._y = inputVal,outputVal
        self._learningRate = learningRate
    
    def tanh(self,x):
        return math.tanh(x)

    def deriv_tanh(self,x):
        return 1-np.power(x,2s)

    def initialize_weights(self):
        for n in self._num
        weights = np.zeros((self._numHiddenLayers,self._neuronsEachLayer))
        for i in range(self._numHiddenLayers):
            for j in range(self._neuronsEachLayer):
                weights[i][j] = random.random()
        bias = np.zeros((1,self._numHiddenLayers))
        return weights,bias


    def sigmoid(self,x):
        x = 1/(1-math.exp(x))
        return x
    
    def derivative_sigmoid(self,x):
        return (self.sigmoid(x)*(1-self.sigmoid(X)))

    def fit(self,X):
        for i in range(self._epochs):
            for k in range(len(X)):
                x_temp = X[k]
                for j in range(self._numHiddenLayers):
                    z = np.dot(self._weights[j].T,x_temp)+self._bias[j]
                    out = self.activationFunction(z) if i<self._numHiddenLayers else self.sigmoid(z)
                    x = out
                loss = out - self._y[k]
                for j in range(self._numHiddenLayers,-1,-1):
                    tempGradient = out - y[self._num] if j == self._numHiddenLayers else np.dot(self._weights[j+1].T,dz)(1-np.power(y[j],2))
                    dz = tempGradient
                    db = np.sum(tempGradient,axis=1,keepdims=True)
                    self._weights[j] -= self._learningRate*dw
                    self._bias[j] -= self._learningRate*db    
        
            


X = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
y = [[0],[1],[1],[0],[1],[0],[0],[1]]

img = ImageColorizer(1,10,3,X,y,0.01)
img.fit(X)