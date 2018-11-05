# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:23:32 2018

@author: wattai
"""

import numpy as np
import itertools
from copy import copy


class CRFs:
    def __init__(self, S, W_dict, W_arr, y_true):
        self.y_true = copy(y_true)
        self.S = copy(S)
        self.W = copy(W_dict)
        self.W_arr = copy(W_arr)
        self.alpha = None
        self.beta = None
        self.Z = None
        self.p_yx = None
        self.p_edge = None
        self.dW = None
        self.feat_dot_W_vec = None

    def featvec_awhole(self, y1, y2):
        featvec = []
        for i in range(len(S)-1):
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                featvec += [seq in [('s', y1), (y1, y2), (y2, '/s')]]
        return np.array(featvec)

    def calc_feat_dot_W_vec(self,):
        feat_dot_W_vec = []
        for i, seq in enumerate(list(itertools.product(*self.S))):
            feat_dot_W = self.W_arr @ self.featvec_awhole(y1=seq[1], y2=seq[2])
            feat_dot_W_vec += [feat_dot_W]
        self.feat_dot_W_vec = np.array(feat_dot_W_vec)
        return self.feat_dot_W_vec

    def softmax(self, x):
        self.Z = np.sum(np.exp(x))
        return np.exp(x) / self.Z

    def calc_p_yx(self,):
        self.p_yx = self.softmax(self.feat_dot_W_vec)
        return self.p_yx

    def forward(self, alpha0):
        self.alpha = alpha0
        for i in range(len(self.S)-1):  # initialize to 0
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                self.alpha[i+1][seq[1]] = 0

        for i in range(len(self.S)-1):  # forward
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                self.alpha[i+1][seq[1]] += np.exp(
                        self.W_arr @ self.featvec_awhole(y1=seq[0], y2=seq[1])
                        ) * self.alpha[i][seq[0]]
                self.Z = self.alpha[i+1][seq[1]]

    def backward(self, beta0):
        self.beta = beta0
        for i in range(len(self.S)-2, -1, -1):  # initialize to 0
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                self.beta[i][seq[0]] = 0

        for i in range(len(self.S)-2, -1, -1):  # backward
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                self.beta[i][seq[0]] += np.exp(
                        self.W_arr @ self.featvec_awhole(y1=seq[0], y2=seq[1])
                        ) * self.beta[i+1][seq[1]]

    def calc_p_edge(self,):
        prob = {}
        for i in range(len(self.S)-1):
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                prob[seq] = self.alpha[i][seq[0]] * \
                            np.exp(
                            self.W_arr @ self.featvec_awhole(y1=seq[0],
                                                             y2=seq[1])
                            ) * self.beta[i+1][seq[1]] / self.Z
        self.p_edge = prob
        return prob

    def update(self, learning_rate=1.0):
        efc_tmp = []
        for i, seq in enumerate(list(itertools.product(*self.S))):
            efc_tmp += [np.array(list(p_edge.values())) *
                        crf.featvec_awhole(y1=seq[1], y2=seq[2])]
        self.expected_feat_cnt = np.sum(np.array(efc_tmp), axis=0)

        dW_arr = np.zeros(self.expected_feat_cnt.shape)
        for i, seq in enumerate(list(itertools.product(*self.S))):
            dW_arr += crf.featvec_awhole(y1=seq[1],
                                         y2=seq[2]
                                         ) - self.expected_feat_cnt
        self.dW_arr = np.array(dW_arr)
        self.W_arr += learning_rate * self.dW_arr


def generate_weight_dict_from_arr(W_tmp, S):
    W = {}
    cnt = 0
    for i in range(len(S)-1):
        for j, seq in enumerate(list(itertools.product(*S[i:i+2]))):
            W[seq] = W_tmp[cnt]
            cnt += 1
    return W


if __name__ == "__main__":

    # your number
    x3, x2, x1 = 3, 1, 7

    W1_arr = np.array([x3+8, x2+9, x1+10,
                      x2+x3, x1+x2, x1+x3, x1+4, x2+6, x3+4,
                      x1+2, x2+3]) / 20

    S = np.array([['s'],
                  ['A', 'V', 'N'],
                  ['N', 'V'],
                  ['/s']])

    pairs = []
    for i in range(len(S)-1):
        for j, seq in enumerate(list(itertools.product(*S[i:i+2]))):
            pairs += [seq]

    paths = []
    for i, seq in enumerate(list(itertools.product(*S))):
        paths += [seq]

    alpha0 = [{S[0][0]: 1.0}, {}, {}, {}]
    beta0 = [{}, {}, {}, {S[-1][0]: 1.0}]

    W1_dict = generate_weight_dict_from_arr(W1_arr, S)

    y_true = [('s', 'A'), ('A', 'N'), ('N', '/s')]
    crf = CRFs(S, W1_dict, W1_arr, y_true)

    # [1]
    print('[1] ---------------------------------------')
    feat_dot_W_vec = crf.calc_feat_dot_W_vec()
    for i, seq in enumerate(list(itertools.product(*S))):
        print('feat_dot_W_%s: %f' % (seq[1:3], feat_dot_W_vec[i]))
    print('')

    # [2]
    print('[2] ---------------------------------------')
    p_yx = crf.calc_p_yx()
    for i, path in enumerate(paths):
        print('prob%s: %f' % (path[1:3], p_yx[i]))
    print('')

    # [3]
    print('[3] ---------------------------------------')
    crf.forward(alpha0)
    for i, a in enumerate(crf.alpha):
        for key in a.keys():
            print('alpha_%d(%s): %f' % (i, key, a[key]))
    print('')

    # [3]
    crf.backward(beta0)
    for i, b in enumerate(crf.beta):
        for key in b.keys():
            print('beta_%d(%s): %f' % (i, key, b[key]))
    print('')

    # [3]
    p_edge = crf.calc_p_edge()
    for key in p_edge.keys():
        print('prob%s: %f' % (key, p_edge[key]))
    print('')

    # [4]
    print('[4] ---------------------------------------')
    W2_arr = np.array([x3+8, x2+9, x1+10,
                      x2+x3, x1+x2, x1+x3, x1+2, x2+3, x3+4,
                      x1+1, x2+2]) / 20
    W2_dict = generate_weight_dict_from_arr(W2_arr, S)

    y_true = [('s', 'A'), ('A', 'N'), ('N', '/s')]
    crf = CRFs(S, W2_dict, W2_arr, y_true)
    crf.calc_feat_dot_W_vec()
    crf.calc_p_yx()
    crf.forward(alpha0)
    crf.backward(beta0)
    crf.calc_p_edge()
    crf.update(learning_rate=1.0)

    for i, pair in enumerate(pairs):
        print('dW_%s: %f' % (pair, crf.dW_arr[i]))
    print('')
    for i, pair in enumerate(pairs):
        print('updated_W_%s: %f' % (pair, crf.W_arr[i]))
    print('')
