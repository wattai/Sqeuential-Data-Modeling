# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:42:55 2018

@author: wattai
"""

import numpy as np
from copy import copy
import links as L


class Network:

    def __init__(self, layers, loss, lr=0.1):
        self.lr = lr
        self.layers = copy(layers)
        self.loss_value = None
        self.y = None
        self.loss = loss

    def forward(self, x):
        if x.ndim == 1:
            self.y = x[None, :].copy()
        elif x.ndim == 2:
            self.y = x.copy()

        for j, layer in enumerate(self.layers):
            self.y = layer.forward(self.y)
        return self.y

    def backward(self,):
        dx = self.loss.backward(dy=1.)
        for layer in self.layers[::-1]:
            dx = layer.backward(dx)

    def update(self,):
        for layer in self.layers[::-1]:
            layer.update(lr=lr)

    def calcloss(self, y_label, y_pred):
        return self.loss.forward(y_label, y_pred)

    def predict(self, X):
        return self.forward(X)


if __name__ is "__main__":

    # param setting.
    b3, b2, b1 = 3, 1, 7

    lr = 0.1
    W1 = np.array([[b2+1, -(b3+1), b1+1],
                   [b1+1, b2+1, -(b3+1)]]) / 10
    W2 = np.array([[-(b1+1)],
                   [-(b2+1)],
                   [-(b3+1)]]) / 10

    X = np.array([[1, 1],
                  [0, 1],
                  [0, 0],
                  [1, 0]])
    y_label = np.array([1, -1, 1, -1])

    # layer (network) setting.
    lin1 = L.Linear(W=W1, nobias=True)
    act1 = L.Tanh()
    lin2 = L.Linear(W=W2, nobias=True)
    act2 = L.Tanh()
    loss = L.SquaredError()

    layers = [lin1, act1, lin2, act2, ]
    net = Network(layers, loss, lr=lr)

    # iterate loops.
    for n_iter in range(1):
        for i, x in enumerate(X):
            print("# No: %d -------------------------------------" % (i+1))
            y_pred = net.forward(x)
            loss_value = net.calcloss(y_label[i], y_pred)
            net.backward()
            net.update()

            print("y1:\n", net.layers[1].y)
            print("y2:\n", net.layers[3].y)

            print("z1:\n", net.layers[0].y)
            print("z2:\n", net.layers[2].y)

            print("dW1:\n", net.layers[0].dW)
            print("dW2:\n", net.layers[2].dW)

            print("W1:\n", net.layers[0].W)
            print("W2:\n", net.layers[2].W)

    W1 = net.layers[0].W
    y1 = net.layers[1].y
    W2 = net.layers[2].W
    y2 = net.layers[3].y

    print("# predicted result: --------------------------")
    print(net.predict(X))
