# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:13:56 2018

@author: wattai
"""

import numpy as np
import itertools
from copy import copy


class CRFs:
    def __init__(self, S, W, y_true):
        self.y_true = copy(y_true)
        self.S = copy(S)
        self.W = copy(W)
        self.alpha = None
        self.beta = None
        self.Z = None
        self.p_yx = None
        self.p_edge = None
        self.dW = None

    def feat(self, y_now, y_prev):  # feature function.
        return (y_prev, y_now) in self.y_true

    def weight(self, y_now, y_prev):  # weight.
        return self.W[(y_prev, y_now)]

    def calc_p_yx(self,):
        Z = 0
        prob = {}
        for i, seq in enumerate(list(itertools.product(*self.S))):
            z = 0
            for j, t in enumerate(range(1, len(seq))):
                z += self.weight(seq[t], seq[t-1]) \
                   * self.feat(seq[t], seq[t-1])
            prob[seq[1:-1]] = z
            Z += np.sum(np.exp(z))

        for key in prob.keys():
            prob[key] = np.exp(prob[key]) / Z

        self.Z = Z
        self.p_yx = prob
        return prob

    def forward(self, alpha0):
        self.alpha = alpha0
        for i in range(len(self.S)-1):  # initialize to 0
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                self.alpha[i+1][seq[1]] = 0

        for i in range(len(self.S)-1):  # forward
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                self.alpha[i+1][seq[1]] += np.exp(
                        self.weight(seq[1], seq[0]) * self.feat(seq[1], seq[0])
                        ) * self.alpha[i][seq[0]]

    def backward(self, beta0):
        self.beta = beta0
        for i in range(len(self.S)-2, -1, -1):  # initialize to 0
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                self.beta[i][seq[0]] = 0

        for i in range(len(self.S)-2, -1, -1):  # backward
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                self.beta[i][seq[0]] += np.exp(
                        self.weight(seq[1], seq[0]) * self.feat(seq[1], seq[0])
                        ) * self.beta[i+1][seq[1]]

    def calc_p_edge(self,):
        prob = {}
        for i in range(len(self.S)-1):
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                prob[seq] = self.alpha[i][seq[0]] * \
                            np.exp(self.weight(seq[1], seq[0]) *
                                   self.feat(seq[1], seq[0])
                                   ) * self.beta[i+1][seq[1]] / self.Z
        self.p_edge = prob
        return prob

    def update(self, learning_rate=1.0):
        dW = {}
        for i in range(len(self.S)-1):
            expected_feat_cnt = 0
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                expected_feat_cnt += \
                    self.p_edge[seq] * self.feat(seq[1], seq[0])

            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                dW[seq] = self.feat(seq[1], seq[0]) - expected_feat_cnt
                self.W[seq] += learning_rate * dW[seq]
        self.dW = dW


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

    W_tmp1 = np.array([x3+8, x2+9, x1+10,
                      x2+x3, x1+x2, x1+x3, x1+4, x2+6, x3+4,
                      x1+2, x2+3]) / 20

    S = np.array([['s'],
                  ['A', 'V', 'N'],
                  ['N', 'V'],
                  ['/s']])

    alpha0 = [{S[0][0]: 1.0}, {}, {}, {}]
    beta0 = [{}, {}, {}, {S[-1][0]: 1.0}]

    W1 = generate_weight_dict_from_arr(W_tmp1, S)

    y_true = [('s', 'A'), ('A', 'N'), ('N', '/s')]
    crf = CRFs(S, W1, y_true)

    # [1]
    print('[1] ---------------------------------------')
    for i, seq in enumerate(list(itertools.product(*S))):
        feat_W = crf.feat(y_now=seq[2], y_prev=seq[1]) * \
                  crf.weight(y_now=seq[2], y_prev=seq[1])
        print('feat_W_%s: %f' % (seq[1:3], feat_W))
    print('')

    # [2]
    print('[2] ---------------------------------------')
    p_yx = crf.calc_p_yx()
    for key in p_yx.keys():
        print('prob%s: %f' % (key, p_yx[key]))
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
    W_tmp2 = np.array([x3+8, x2+9, x1+10,
                      x2+x3, x1+x2, x1+x3, x1+2, x2+3, x3+4,
                      x1+1, x2+2]) / 20
    W2 = generate_weight_dict_from_arr(W_tmp2, S)

    y_true = [('s', 'A'), ('A', 'N'), ('N', '/s')]
    crf = CRFs(S, W2, y_true)
    crf.calc_p_yx()
    crf.forward(alpha0)
    crf.backward(beta0)
    crf.calc_p_edge()
    crf.update(learning_rate=1.0)

    for key in crf.dW.keys():
        print('dW_%s: %f' % (key, crf.dW[key]))
    print('')
    for key in crf.W.keys():
        print('updated_W_%s: %f' % (key, crf.W[key]))
    print('')
