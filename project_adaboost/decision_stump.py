import math
import numpy as np
import operator


class StumpTree:
    def __init__(self, err, threshold, s):
        self.err = err
        self.threshold = threshold
        self.s = s
    def __cmp__(self, other):
        return cmp(self.err, other.err)

class DecisionStump(object):
    def __init__(self):
        pass
    def stump_train(self, optional):
        X = self.X
        Y = self.Y
        w = self.weights
        feature_index, stump, best_threshold  = train_decision_stump(X, Y, w)
        if optional:
            print('Best Feature: ', feature_index)
            print('Threshold: ', best_threshold )
        self.feature_index = feature_index
        self.stump = stump


    def stump_predict(self, X):
        N, d = X.shape
        feature_i = self.feature_index
        threshold = self.stump.threshold
        s = self.stump.s
        # update the Y for the given threshold
        Y = np.ones(N)
        Y[np.where(X[:,feature_i] < threshold)[0]] = -1 * s
        Y[np.where(X[:,feature_i] >= threshold)[0]] = 1 * s
        return Y


    def set_training_sample(self, X, Y, w=None):
        self.X = X
        self.Y = Y
        if w: self.weights = w
        else: self.set_uni_weights()


    def set_uni_weights(self):
        N = len(self.Y)
        self.weights = (1.0/N)*np.ones(N)

def train_decision_stump(X, Y, w):
    stumps = [build_stump_1D(x, Y, w) for x in X.T]
    feature_i, best_s = min(enumerate(stumps),key=operator.itemgetter(1))
    best_t = best_s.threshold
    return feature_i, best_s, best_t
    
def build_stump_1D(x, y, w):
    xyw = np.array(sorted(zip(x, y, w), key=operator.itemgetter(0)))
    xsorted = xyw[:, 0]
    wy = xyw[:,1 ] * xyw[:, 2]
    # calculate the scores to define boundaary
    score_left = np.cumsum(wy)
    score_right = np.cumsum(wy[::-1])
    score = - score_left[0:-1:1] + score_right[-1:0:-1]
    I = np.where(xsorted[:-1] < xsorted[1:])[0]
    
    if len(I) > 0:
        ind, max_score = max(zip(I, abs(score[I])), key=operator.itemgetter(1))
        err = 0.5-0.5*max_score 
        threshold = (xsorted[ind] + xsorted[ind+1])/2
        s = np.sign(score[ind])
    else: err, threshold, s = 0.5, 0, 1.0
    return StumpTree(err, threshold, s)
