# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:45:20 2018

@author: wattai
"""

import numpy as np
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt
from copy import copy


class KalmanFilter:

    def __init__(self, A, W, mu0, P0, sigma, gamma):
        self.A = copy(A)
        self.W = copy(W)
        self.mu = copy(mu0)
        self.P = copy(P0)
        self.sigma = copy(sigma)
        self.gamma = copy(gamma)
        self.mu_ = None
        self.P_ = None
        self.K = None

    def predict(self,):
        self.mu_ = self.A @ self.mu
        self.P_ = self.A @ self.P @ self.A.T + self.gamma

    def update(self, x_new):
        self.K = self.P_ @ self.W.T @ np.linalg.inv(
                self.W @ self.P_ @ self.W.T + self.sigma)
        self.mu = self.mu_ + self.K @ (x_new - self.W @ self.mu_)
        E = np.eye(self.W.shape[0])
        self.P = (E - self.K @ self.W) @ self.P_

    def prediciton_distribution_of_latent_variable(self, z):
        mean = copy(self.mu_)
        cov = copy(self.P_)
        p_z_now_x_prev = mn.pdf(z, mean, cov)
        return p_z_now_x_prev

    def prediction_distribution_of_observation_data(self, x):
        mean = self.W @ self.mu_
        cov = (self.W @ self.P_ @ self.W.T) + self.sigma
        p_x_now_x_prev = mn.pdf(x, mean, cov)
        return p_x_now_x_prev

    def update_distribution_of_latent_variable(self, z):
        mean = copy(self.mu)
        cov = copy(self.P)
        p_z_now_x_now = mn.pdf(z, mean, cov)
        return p_z_now_x_now


if __name__ is "__main__":

    # define params ----------------------------
    b2, b1, b0 = 3, 1, 7
    B = np.array([b0, b1, b2])
    x_sample = 10 * B + 20

    A = np.array([[1]])
    W = np.array([[1]])
    sigma = np.array([[10]])
    gamma = np.array([[20]])
    mu0 = np.array([[5*b0 + 30]])
    P0 = ([[5*b1 + 50]])
    # ------------------------------------------

    # set axis for pdf -------------------------
    x = np.linspace(0, 100, 1000)
    z = np.linspace(0, 100, 1000)
    # ------------------------------------------

    # begin filtering --------------------------
    kf = KalmanFilter(A, W, mu0, P0, sigma, gamma)

    # t = 0
    kf.predict()
    p_z1x0 = kf.prediciton_distribution_of_latent_variable(z)
    p_x1x0 = kf.prediction_distribution_of_observation_data(x)
    p_x1x0_ln = np.log(p_x1x0)

    # t = 1
    kf.update(x_new=x_sample[1])
    p_z1x1 = kf.update_distribution_of_latent_variable(z)
    kf.predict()
    p_z2x1 = kf.prediciton_distribution_of_latent_variable(z)
    p_x2x1 = kf.prediction_distribution_of_observation_data(x)
    p_x2x1_ln = np.log(p_x2x1)

    # t = 2
    kf.update(x_new=x_sample[2])
    p_z2x2 = kf.update_distribution_of_latent_variable(z)
    # -------------------------------------------

    # visualization for each distribution -------
    fig = plt.figure(figsize=(6, 4))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(z, p_z1x0, label="p_z1x0")
    ax1.plot(z, p_z2x1, label="p_z2x1")
    ax1.set_xlabel("z")
    ax1.set_ylabel("probability")
    ax1.grid()
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(x, p_x1x0, label="p_x1x0")
    ax2.plot(x, p_x2x1, label="p_x2x1")
    ax2.set_xlabel("x")
    ax2.set_ylabel("probability")
    ax2.grid()
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(z, p_z1x1, label="p_z1x1")
    ax3.plot(z, p_z2x2, label="p_z2x2")
    ax3.set_xlabel("z")
    ax3.set_ylabel("probability")
    ax3.grid()
    ax3.legend(loc="upper right")

    fig.tight_layout()
    fig.show()
    # -------------------------------------------
