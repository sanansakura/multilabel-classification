"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k ,method = 'L2'):
        self.k = k
        self.method = method
    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 
        self.labels = y.shape[1]

    def predict(self, Xtest):
        #Compute the Euclidean distance
        N, D = self.X.shape
        T, D = Xtest.shape

        y_pred = np.zeros((T, self.y.shape[1]))
        if self.method == "L2":
            distance = utils.euclidean_dist_squared(self.X, Xtest)
        elif self.method == "cosine":
            distance =  utils.cosine_similarity(self.X, Xtest)
            #print(distance.shape)
        elif self.method == "pearson":
            distance = utils.pearson_corr(self.X, Xtest)
            #print(distance.shape)
        for t in range(T):
            sorted_distance_k =  np.argsort(distance[:, t])[:self.k]
            #print(sorted_distance_k)
            for l in range(self.labels):
                #calculate the conditional probability that P(y_j = 1|x)
                p = (1/self.k)*np.sum(self.y[:,l][sorted_distance_k])
                #print(p)
                if p>0.5:
                    y_pred[t,l] = 1
                else:
                    y_pred[t, l] = 0
        	    #y_pred[t] = utils.mode(self.y[sorted_distance_k] )
        	
        return y_pred