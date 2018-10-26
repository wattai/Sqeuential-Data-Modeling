# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 17:58:39 2018

@author: wattai
"""

import numpy as np
from copy import copy


class Linear:

    def __init__(self, W, nobias=False):
        self.W = W.copy()
        self.dW = np.zeros_like(W)
        self.b = np.zeros([1, W.shape[1]])
        self.db = np.zeros_like(self.b)
        self.x = None
        self.y = None
        self.nobias = nobias

    def forward(self, x):
        self.x = x.copy()
        if self.nobias:
            self.y = x @ self.W
        else:
            self.y = x @ self.W + self.b
        return self.y

    def backward(self, dy):
        self.dW = self.x.T @ dy
        if self.W.shape != self.dW.shape:
            raise ValueError("the weight size mismatch !")
        self.db = dy.copy()
        dx = dy @ self.W.T
        return dx

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Tanh:

    def __init__(self,):
        self.y = None

    def forward(self, x):
        self.y = self.tanh(x)
        return self.y

    def backward(self, dy):
        dx = self.dtanh(self.y) * dy
        return dx

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 4 / (np.exp(-x) + np.exp(x))**2

    def update(self, lr):
        None


class SquaredError:

    def __init__(self,):
        self.y_label = None
        self.y_pred = None

    def forward(self, y_label, y_pred):
        self.y_label = copy(y_label)
        self.y_pred = copy(y_pred)
        self.loss_value = self.lossfunc(self.y_label, self.y_pred)
        return self.loss_value

    def backward(self, dy):
        dx = self.dlossfunc(self.y_label, self.y_pred) * dy
        return dx

    def lossfunc(self, y_label, y_pred):
        return np.sum((y_label - y_pred)**2 / 2, axis=0)

    def dlossfunc(self, y_label, y_pred):
        return -1. * (y_label - y_pred)
