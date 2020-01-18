import numpy as np
import pandas as pd
import math
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y


def derivative_sigmoid(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def tanh_func(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.power(x,2)

def rgb_grayscale(img):
    
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    gray = 0.21*r + 0.72*g + 0.07*b

    return gray

X = []
Y = []
numImages = 1

for i in range(numImages):
    y = mpimg.imread('pic.jpg')[1980:2180, 1070:1270]
    x = rgb_grayscale(y)
    yr = y[:,:,0]
    yg = y[:,:,1]
    yb = y[:,:,2]


X_flat = (x.flatten().reshape(40000, 1))/255.0
Yr_flat = (yr.flatten().reshape(40000, 1))/255.0
Yg_flat = (yg.flatten().reshape(40000, 1))/255.0
Yb_flat = (yb.flatten().reshape(40000, 1))/255.0

og = np.zeros((200, 200, 3))
og[:,:,0] = (Yr_flat*255).reshape(200, 200).astype(int)
og[:,:,1] = (Yg_flat*255).reshape(200, 200).astype(int)
og[:,:,2] = (Yb_flat*255).reshape(200, 200).astype(int)


import scipy
scipy.misc.imsave('yash.jpg', og)

plt.imshow(og)
plt.savefig("atharva.jpg")
plt.show()

print(og)

lr = 0.01
b1r = np.zeros((2, 1))
b2r = np.zeros((1, 1))
b1g = np.zeros((2, 1))
b2g = np.zeros((1, 1))
b1b = np.zeros((2, 1))
b2b = np.zeros((1, 1))

weights1r = np.random.rand(2, 1)
weights2r = np.random.rand(1, 2)
weights1g = np.random.rand(2, 1)
weights2g = np.random.rand(1, 2)
weights1b = np.random.rand(2, 1)
weights2b = np.random.rand(1, 2)

for j in range(20):
    loss = 0
    for i in range(len(X_flat)):
        # print(X[i].shape)
        hidden1r = tanh_func(np.dot(X_flat[i].reshape(1,1), weights1r.T) + b1r.T)  #(1,2)
        # print("hidden1 shape:", hidden1.shape)
        # print(hidden1.shape)
        hidden2r = sigmoid(np.dot(weights2r, hidden1r.T) + b2r)  #(1,1)
        # print("hidden2 shape:", hidden2.shape)
        # print(hidden2.shape)
        hidden1g = tanh_func(np.dot(X_flat[i].reshape(1,1), weights1g.T) + b1g.T)
        hidden2g = sigmoid(np.dot(weights2g, hidden1g.T) + b2g)
        hidden1b = tanh_func(np.dot(X_flat[i].reshape(1,1), weights1b.T) + b1b.T)
        hidden2b = sigmoid(np.dot(weights2b, hidden1b.T) + b2b)

        loss1 = np.power(hidden2r - Yr_flat[i], 2)
        loss2 = np.power(hidden2g - Yg_flat[i], 2)
        loss3 = np.power(hidden2b - Yb_flat[i], 2)

        dZ2r = hidden2r-Yr_flat[i]   #(1,1)
        # print("dz2 shape:", dZ2.shape)
        dW2r = np.dot(dZ2r.T, hidden1r)
        # print("dw2 shape:", dW2.shape)
        db2r = np.sum(dZ2r, axis=1, keepdims=True)
        dZ2g = hidden2g-Yg_flat[i]
        dW2g = np.dot(dZ2g.T, hidden1g)
        db2g = np.sum(dZ2g, axis=1, keepdims=True)
        dZ2b = hidden2b-Yb_flat[i]
        dW2b = np.dot(dZ2b.T, hidden1b)
        db2b = np.sum(dZ2b, axis=1, keepdims=True)

        dZ1r = np.dot(weights2r.T, dZ2r) * tanh_deriv(hidden1r.T)   #(2, 1)
        # print("dz1 shape:", dZ1.shape)
        dW1r = np.dot(dZ1r, X_flat[i].reshape(1,1))
        # print("dw1 shape:", dW1.shape)
        db1r = np.sum(dZ1r, axis=1, keepdims=True)
        dZ1g = np.dot(weights2g.T, dZ2g) * tanh_deriv(hidden1g.T)
        dW1g = np.dot(dZ1g, X_flat[i].reshape(1,1))
        db1g = np.sum(dZ1g, axis=1, keepdims=True)
        dZ1b = np.dot(weights2b.T, dZ2b) * tanh_deriv(hidden1b.T)
        dW1b = np.dot(dZ1b, X_flat[i].reshape(1,1))
        db1b = np.sum(dZ1b, axis=1, keepdims=True)

        weights2r-= lr*dW2r
        b2r-= lr*db2r
        weights1r-= lr*dW1r
        b1r-= lr*db1r
        weights2g-= lr*dW2g
        b2g-= lr*db2g
        weights1g-= lr*dW1g
        b1g-= lr*db1g
        weights2b-= lr*dW2b
        b2b-= lr*db2b
        weights1b-= lr*dW1b
        b1b-= lr*db1b
    print(loss1, loss2, loss3)
    # input()


resultr = np.zeros((40000, 1))
resultg = np.zeros((40000, 1))
resultb = np.zeros((40000, 1))

for i in range(len(X_flat)):
    hidden1r = tanh_func(np.dot(X_flat[i].reshape(1,1), weights1r.T) + b1r.T)
    hidden2r = sigmoid(np.dot(weights2r, hidden1r.T) + b2r)
    resultr[i, 0] = hidden2r
    hidden1g = tanh_func(np.dot(X_flat[i].reshape(1,1), weights1g.T) + b1g.T)
    hidden2g = sigmoid(np.dot(weights2g, hidden1g.T) + b2g)
    resultg[i, 0] = hidden2g
    hidden1b = tanh_func(np.dot(X_flat[i].reshape(1,1), weights1b.T) + b1b.T)
    hidden2b = sigmoid(np.dot(weights2b, hidden1b.T) + b2b)
    resultb[i, 0] = hidden2b


resultr = resultr.reshape(200, 200)*255.0
Yr_flat = Yr_flat.reshape(200, 200)*255.0
resultg = resultg.reshape(200, 200)*255.0
Yg_flat = Yg_flat.reshape(200, 200)*255.0
resultb = resultb.reshape(200, 200)*255.0
Yb_flat = Yb_flat.reshape(200, 200)*255.0


final_image = np.zeros((200, 200, 3))
final_image[:,:,0] = resultr.astype(int)
final_image[:,:,1] = resultg.astype(int)
final_image[:,:,2] = resultb.astype(int)


# og = np.zeros((200, 200, 3))
# og[:,:,0] = Yr_flat
# og[:,:,1] = Yg_flat
# og[:,:,2] = Yb_flat

print(final_image)
print(final_image.shape)


plt.imshow(final_image)
plt.savefig("final.jpg")
plt.show()

# plt.imshow(og)
# plt.savefig("og.jpg")
# plt.show()

# plt.imshow(Y_flat, cmap="gray")
# plt.savefig("Y_flat.jpg")
# plt.show()


# print(result)
# print(Y_flat)

# hidden1 = tanh_func(np.dot(weights1.T, X[0].reshape(2,1)) + b1)
# # print(hidden1.shape)
# hidden2 = sigmoid(np.dot(weights2.T,hidden1 ) + b2)
# # print(hidden2.shape)
