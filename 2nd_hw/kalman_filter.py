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
        print("mu_:\n", self.mu_)
        print("P_:\n", self.P_)

    def update(self, x_new):
        self.K = self.P_ @ self.W.T @ np.linalg.inv(
                self.W @ self.P_ @ self.W.T + self.sigma)
        self.mu = self.mu_ + self.K @ (x_new - self.W @ self.mu_)
        E = np.eye(self.W.shape[0])
        self.P = (E - self.K @ self.W) @ self.P_
        print("K:\n", self.K)
        print("mu:\n", self.mu)
        print("P:\n", self.P)

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
    print("t=0")
    p_z0_x0 = kf.update_distribution_of_latent_variable(z)

    # t = 1
    print("t=1")
    kf.predict()
    p_z1_x0 = kf.prediciton_distribution_of_latent_variable(z)
    p_x1_x0 = kf.prediction_distribution_of_observation_data(x)
    p_x1_x0_ln = np.log(p_x1_x0)
    kf.update(x_new=x_sample[1])
    p_z1_x1 = kf.update_distribution_of_latent_variable(z)

    # t = 2
    print("t=2")
    kf.predict()
    p_z2_x1 = kf.prediciton_distribution_of_latent_variable(z)
    p_x2_x1 = kf.prediction_distribution_of_observation_data(x)
    p_x2_x1_ln = np.log(p_x2_x1)
    kf.update(x_new=x_sample[2])
    p_z2_x2x1 = kf.update_distribution_of_latent_variable(z)

    # t = 3
    print("t=3")
    kf.predict()
    p_z3_x2x1 = kf.prediciton_distribution_of_latent_variable(z)
    p_x3_x2x1 = kf.prediction_distribution_of_observation_data(x)
    p_x3_x2x1_ln = np.log(p_x3_x2x1)
    # -------------------------------------------

    # visualization for each distribution -------
    fig = plt.figure(figsize=(6, 5))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(z, p_z0_x0, label="p_z0_x0", color='skyblue')
    ax1.plot(z, p_z1_x1, label="p_z1_x1")
    ax1.plot(z, p_z2_x2x1, label="p_z2_x2x1")
    ax1.set_xlabel("z")
    ax1.set_ylabel("probability")
    ax1.grid()
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(z, p_z1_x0, label="p_z1_x0")
    ax2.plot(z, p_z2_x1, label="p_z2_x1")
    ax2.plot(z, p_z3_x2x1, label="p_z3_x2x1")
    ax2.set_xlabel("z")
    ax2.set_ylabel("probability")
    ax2.grid()
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(x, p_x1_x0, label="p_x1_x0")
    ax3.plot(x, p_x2_x1, label="p_x2_x1")
    ax3.plot(x, p_x3_x2x1, label="p_x3_x2x1")
    ax3.set_xlabel("x")
    ax3.set_ylabel("probability")
    ax3.grid()
    ax3.legend(loc="upper right")

    fig.tight_layout()
    fig.show()
    # -------------------------------------------

    # confirmation of your answer ---------------
    print("# confirmation of your answer -------")

    mu_10, P_10 = np.array([[65]]), np.array([[75]])
    mean_x1_x0 = W @ mu_10
    cov_x1_x0 = W @ P_10 @ W.T + sigma
    pp_x1_x0_ln = np.log(mn.pdf(x, mean=mean_x1_x0, cov=cov_x1_x0))
    rms_x1_x0_ln = np.linalg.norm(p_x1_x0_ln - pp_x1_x0_ln)
    print("rms_x1_x0_ln: %f" % rms_x1_x0_ln)

    mu1, P1 = np.array([[580/17]]), np.array([[150/17]])
    mean_z1_x1 = copy(mu1)
    cov_z1_x1 = copy(P1)
    pp_z1_x1 = mn.pdf(z, mean=mean_z1_x1, cov=cov_z1_x1)
    rms_z1_x1 = np.linalg.norm(p_z1_x1 - pp_z1_x1)
    print("rms_z1_x1: %f" % rms_z1_x1)

    mu_21, P_21 = np.array([[580/17]]), np.array([[490/17]])
    mean_z2_x1 = copy(mu_21)
    cov_z2_x1 = copy(P_21)
    pp_z2_x1 = mn.pdf(z, mean=mean_z2_x1, cov=cov_z2_x1)
    rms_z2_x1 = np.linalg.norm(p_z2_x1 - pp_z2_x1)
    print("rms_z2_x1: %f" % rms_z2_x1)

    mean_x2_x1 = W @ mu_21
    cov_x2_x1 = W @ P_21 @ W.T + sigma
    pp_x2_x1_ln = np.log(mn.pdf(x, mean=mean_x2_x1, cov=cov_x2_x1))
    rms_x2_x1_ln = np.linalg.norm(p_x2_x1_ln - pp_x2_x1_ln)
    print("rms_x2_x1_ln: %f" % rms_x2_x1_ln)

    mu2, P2 = np.array([[505/11]]), np.array([[245/33]])
    mean_z2_x2x1 = copy(mu2)
    cov_z2_x2x1 = copy(P2)
    pp_z2_x2x1 = mn.pdf(z, mean=mean_z2_x2x1, cov=cov_z2_x2x1)
    rms_z2_x2x1 = np.linalg.norm(p_z2_x2x1 - pp_z2_x2x1)
    print("rms_z1_x1: %f" % rms_z2_x2x1)
    # -------------------------------------------
