import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils import *
from sklearn import *
import scipy
import itertools
import random
from sklearn.metrics import *
from sklearn import tree

class RAKEL:
    
    def __init__(self, n_classifier = 15, method = "SVM"):
        self.n_classifier = n_classifier
        self.method = method
        
    def fit(self, X, y, k, method = 'SVM'):
        self.train_model( X, y, k, method)
       
    def train_model(self, X, y, k, method = 'SVM'):
        _, self.n_label = y.shape
        L = [i for i in range(self.n_label)]
        L_k = scipy.special.comb(len(L), k)
        
        
        X_lst, y_lst = cross_valid(X, y, 10)    
        avg_loss_so_far = np.inf
        for t in range(0, 6):
            t= 0.2*t
            sum_loss = 0 
            #print("current t is ", t)
            for fold in range(10):
                #print("current fold is ", fold)
                Xtrain, ytrain, Xvalid, yvalid = get_fold(X_lst, y_lst, fold)
                
                labels = (i for i in range(self.n_label))
                R = set(itertools.combinations(L, k))
                classifiers = []
                #print(L_k, self.n_classifier)
                for i in range(int(min(self.n_classifier, L_k))):
                    Y_i = random.choice(list(R))
                    y_tmp = ytrain[:,Y_i]
                    m, im = LP_map(k)
                    y_tmp = transform(m,im,y_tmp)
                    if self.method == "SVM":
                        model = svm.LinearSVC()
                        #print("use model " + "SVM")
                    elif self.method == "DT":
                        model = tree.DecisionTreeClassifier()
                        #print("use model " + "DT")
                    elif self.method == "KNN":
                        model = KNeighborsClassifier()
                        #print("use model " + "KNN")
                    elif self.method == "mix":
                        model_lst = [svm.LinearSVC(),tree.DecisionTreeClassifier(),  KNeighborsClassifier()]
                        model_name_lst = ["SVM", "DT", "KNN"]
                        model_index = random.choice([0,1,2])
                        model = model_lst[model_index]
                        #print("use model" + model_name_lst[model_index])
            
                    model.fit(Xtrain, y_tmp)
                    #print("fitted model %d, training error %.3f" % (i, np.abs(np.mean(model.predict(Xtrain) - y_tmp))))
                    #print(m)
                    classifiers.append((model, Y_i, m, im))
                    R = set(R) - {Y_i}
                    labels = tuple(set(labels) - set(Y_i))
                    #print(labels)
            
                    # if there are labels remaining:
                if labels:
                    #print(labels)
                    y_tmp = ytrain[:,labels]
                    #print(labels)
                    m, im = LP_map(len(labels))
                    y_tmp = transform(m,im,y_tmp)
                    additional_classifier = svm.LinearSVC()
                    additional_classifier.fit(Xtrain, y_tmp)
                    classifiers.append((additional_classifier, labels, m, im))
                self.classifiers_tmp = classifiers
                predicted = self.predict_helper(Xvalid, t)
                sum_loss += hamming_loss(yvalid, predicted)
                avg_loss = sum_loss/10
            
                if avg_loss < avg_loss_so_far:
                    avg_loss_so_far = avg_loss
                    self.t= t
                    self.classifiers = self.classifiers_tmp
                    #print("update t: ", self.t)
                #print("Current t is %.1f, Current fold is: %d, Avg loss is: %.3f, Min loss so far is: %.3f" %(t, fold, avg_loss, avg_loss_so_far))
        #print("Assign model weights: ...")
        self.model_weight_helper(X_lst, y_lst, self.classifiers)
        
    def predict_helper(self, X, t):
        n,d = X.shape
        #predicted = np.zeros((n, self.n_label))
        
        summation = np.zeros((n, self.n_label))
        votes = np.zeros((n, self.n_label))
        for m in range(len(self.classifiers_tmp)):
            model, Yi, m, im = self.classifiers_tmp[m]
            #predict the current example
            predict_tmp = model.predict(X)
            predict_tmp = reverse_transform(m, im, predict_tmp, len(Yi))
            summation[:,Yi] = summation[:,Yi] + predict_tmp
            votes[:,Yi] = votes[:,Yi] + np.ones((n, len(Yi)))
        avg = summation/votes
        avg[avg>t] = 1
        avg[avg<=t] = 0
        return avg
            #print(predict_tmp)
    def model_weight_helper(self, X_lst, y_lst, models):
        n_models = len(models)
        loss = np.zeros(n_models)
        for fold in range(10):
            
            Xtrain, ytrain, Xvalid, yvalid = get_fold(X_lst, y_lst, fold)
            for i in range(len(models)):
                model_i, Yi, m, im = self.classifiers[i]
                predict_tmp = model_i.predict(Xvalid)
                predict_tmp = reverse_transform(m, im, predict_tmp, len(Yi))
               
                loss[i] += hamming_loss(yvalid[:,Yi], predict_tmp)
            #print(fold, loss)
        reverse_loss = 1 - loss
        self.model_weights = reverse_loss/np.sum(loss)
        print(self.model_weights)
    def predict(self, X):
        n,d = X.shape
        #predicted = np.zeros((n, self.n_label))
        weights = self.model_weights
        summation = np.zeros((n, self.n_label))
        votes = np.zeros((n, self.n_label))
        for i in range(len(self.classifiers)):
            model, Yi, m, im = self.classifiers[i]
            #predict the current example
            predict_tmp = model.predict(X)
            predict_tmp = reverse_transform(m, im, predict_tmp, len(Yi))
            summation[:,Yi] += summation[:,Yi] + weights[i]*predict_tmp
            votes[:,Yi] += weights[i]*np.ones((n, len(Yi)))
        avg = summation/votes
        avg[avg>self.t] = 1
        avg[avg<=self.t] = 0
        return avg
            
        
    
    