import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils import *
from sklearn import *
import scipy
import itertools
import random

class RAKEL_original:
    
    def __init__(self, threshold = 0.5, n_classifier = 1):
        self.n_classifier = n_classifier
        self.threshold = threshold
    def fit(self, X, y, k, method = 'SVM'):
        _, self.n_label = y.shape
        L = [i for i in range(self.n_label)]
        L_k = scipy.special.comb(len(L), k)
        
        R = set(itertools.combinations(L, k))
        classifiers = []
        for i in range(min(self.n_classifier, L_k)):
            Y_i = random.choice(list(R))
            y_tmp = y[:,Y_i]
            m, im = LP_map(k)
            y_tmp = transform(m,im,y_tmp)
            model = svm.LinearSVC()
            
            model.fit(X, y_tmp)
            print("fitted model %d, training error %.3f" % (i, np.abs(np.mean(model.predict(X) - y_tmp))))
            #print(m)
            classifiers.append((model, Y_i, m, im))
            R = set(R) - {Y_i}
            
        self.classifiers = classifiers
        #determine threshold t
        #self.t = t

    def predict(self, X):
        n,d = X.shape
        #predicted = np.zeros((n, self.n_label))
        
        summation = np.zeros((n, self.n_label))
        votes = np.zeros((n, self.n_label))
        for m in range(len(self.classifiers)):
            model, Yi, m, im = self.classifiers[m]
            #predict the current example
            predict_tmp = model.predict(X)
            predict_tmp = reverse_transform(m, im, predict_tmp, len(Yi))
            summation[:,Yi] = summation[:,Yi] + predict_tmp
            votes[:,Yi] = votes[:,Yi] + np.ones((n, len(Yi)))
        avg = summation/votes
        avg[avg>self.threshold] = 1
        avg[avg<=self.threshold] = 0
        return avg

            
        
    
    