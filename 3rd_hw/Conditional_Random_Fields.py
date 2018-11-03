# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:13:56 2018

@author: wattai
"""

import numpy as np
import itertools


class CRFs:
    def __init__(self, S, W, y_true):
        self.y_true = y_true
        self.S = S
        self.W = W
        self.alpha = None
        self.beta = None
        self.Z = None

    def feat(self, y_now, y_prev):  # feature function.
        return (y_now, y_prev) in self.y_true

    def weight(self, y_now, y_prev):  # weight.
        return self.W[(y_now, y_prev)]

    def p_yx(self,):
        Z = 0
        prob_tmp = {}
        for i, seq in enumerate(list(itertools.product(*self.S))):
            z = 0
            # print(seq[1:-1])
            for j, t in enumerate(range(1, len(seq))):
                z += self.weight(seq[t], seq[t-1]) \
                   * self.feat(seq[t], seq[t-1])
            prob_tmp[seq[1:-1]] = z
            Z += np.sum(np.exp(z))

        for key in prob_tmp.keys():
            prob_tmp[key] = np.exp(prob_tmp[key]) / Z

        self.Z = Z
        return prob_tmp
    
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

    def p_edge(self,):
        prob = {}
        for i in range(len(S)-1):
            for j, seq in enumerate(list(itertools.product(*self.S[i:i+2]))):
                prob[seq] = self.alpha[i][seq[0]] * \
                            np.exp(self.weight(seq[1], seq[0]) * 
                                   self.feat(seq[1], seq[0])
                                   ) * self.beta[i+1][seq[1]] / self.Z
        return prob


if __name__ == "__main__":

    # your number
    x3, x2, x1 = 3, 1, 7

    W_tmp = np.array([x3+8, x2+9, x1+10,
                      x2+x3, x1+x2, x1+x3, x1+4, x2+6, x3+4,
                      x1+2, x2+3]) / 20

    S = np.array([['s'],
                  ['A', 'V', 'N'],
                  ['N', 'V'],
                  ['/s']])

    alpha0 = [{S[0][0]: 1.0}, {}, {}, {}]
    beta0 = [{}, {}, {}, {S[-1][0]: 1.0}]

    W = {}
    cnt = 0
    for i in range(len(S)-1):
        for j, seq in enumerate(list(itertools.product(*S[i:i+2]))):
            # print(i, j, seq)
            W[seq[::-1]] = W_tmp[cnt]
            cnt += 1

    # [1]
    y_true = [('A', 's'), ('N', 'A'), ('/s', 'N')]
    crf = CRFs(S, W, y_true)

    # [2]
    p_yx = crf.p_yx()
    for key in p_yx.keys():
        print('prob%s: %f' % (key, p_yx[key]))
    print('')

    # [3]
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
    p_edge = crf.p_edge()
    for key in p_edge.keys():
        print('prob%s: %f' % (key, p_edge[key]))
    print('')
