# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:49:24 2018

@author: imd
"""

import numpy as np

if __name__ is "__main__":

    b3, b2, b1, b0 = 1, 3, 1, 7

    # problem 1.
    # [1]
    X = np.array([[b2, b3],
                  [b1+b3, b0+b2],
                  [b0+b3, b1],
                  [b0+b2, b1+b2],
                  [b0+b2, b0+b1+b2]])
    mu = np.mean(X, axis=0)
    X_dash = X - mu
    print(mu*5)
    print(X_dash*5)
    
    # [2]
    
