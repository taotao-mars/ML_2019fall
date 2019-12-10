import sys
import math
import numpy as np


class AdaBoost(object):
    def __init__(self, weak_classifier):
        self.WeakClassifier = weak_classifier
    def ada_train(self, T, X, Y, optional=False):
       # Defines variables 
        self.weak_classifier_ens = []
        self.alpha = []
        self.X = X
        self.Y = Y
        self.T = T
        self.e = []
        N = len(self.Y)
        Z = (1.0/N)*np.ones(N)
        for t in range(T):
            weak_learner = self.WeakClassifier()
            weak_learner.set_training_sample(X,Y)
            weak_learner.weights = Z
            if t < 10 and optional:
                print("For t = ", t+1)
                print('Y= ', int(Y[t]))
                opt = True
            else: opt = False
        weak_learner.stump_train(opt)
        self.weak_classifier_ens.append(weak_learner)
        Y_p= weak_learner.stump_predict(X)
        epsilon = sum(0.5*Z*abs((Y-Y_p)))/sum(Z)
        self.e = epsilon
        inside = abs( (1-epsilon)/(epsilon*1.0)+0.00001 )
            #print inside
        alpha = 0.5*math.log(inside)
        self.alpha.append(alpha)
        Z *= np.exp(-alpha*Y*Y_p)
        Z /= sum(Z)
    def ada_predict(self, X=[]):
        if X == None: return
        X = np.array(X)
        N, d = X.shape
        Y = np.zeros(N)
        score = []
        for t in range(self.T):
            weak_learner = self.weak_classifier_ens[t]
            Y += self.alpha[t]*weak_learner.stump_predict(X)
            score.append(np.sign(Y))
        return score




    def run_adaboost(self, X_train, Y_train, T, X_test=None, optional=False):
        self.ada_train(T, X_train, Y_train, optional)
        return self.ada_predict(X_train), self.ada_predict(X_test)


