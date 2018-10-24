# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:49:24 2018

@author: wattai
"""

import numpy as np

if __name__ is "__main__":

    b3, b2, b1, b0 = 1, 3, 1, 7

    # problem 1.
    X = np.array([[b2, b3],
                  [b1+b3, b0+b2],
                  [b0+b3, b1],
                  [b0+b2, b1+b2],
                  [b0+b2, b0+b1+b2]]).T
    W = np.array([[1],
                  [0]])
    mu = np.array([[0],
                   [0]])
    sigma = np.array([[1, 0],
                      [0, 1]])

    # [1]
    mu_hat = np.mean(X, axis=1, keepdims=True)
    X_dash = X - mu_hat
    print("mu_hat:\n", mu_hat)
    print("X_dash:\n", X_dash)

    # [2]
    sigma_inv = np.linalg.inv(sigma)
    Wt_S_W = W.T @ sigma_inv @ W
    E = np.eye(Wt_S_W.shape[0], Wt_S_W.shape[1])
    sigma_zx = np.linalg.inv(Wt_S_W + E)
    mu_zx = sigma_zx @ W.T @ sigma_inv @ X_dash

    z_n = mu_zx.copy()
    z_zt_n = np.diag(sigma_zx + (mu_zx.T @ mu_zx))[None, :]
    print("z_n:\n", z_n)
    print("z_zt_n:\n", z_zt_n)

    # [3]
    N = X.shape[1]
    x_xt = np.diag(np.diag(X_dash @ X_dash.T))
    z_zt = np.sum(z_zt_n, axis=1, keepdims=True)
    x_zt = X_dash @ z_n.T
    print("N:\n", N)
    print("x_xt:\n", x_xt)
    print("z_zt:\n", z_zt)
    print("x_zt:\n", x_zt)

    # [4]
    W_hat = x_zt @ np.linalg.inv(z_zt)
    sigma_hat = (1/N) * np.diag(np.diag(x_xt - x_zt @ W_hat.T))
    print("W_hat:\n", W_hat)
    print("sigma_hat:\n", sigma_hat)
