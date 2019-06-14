from knn import *
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.preprocessing import *
from utils import *
from ensemble import *

def preprocessing(file_name):
    '''
    Preprocessing the data.
    
    Return a numpy array.
    '''
    data_file = open(file_name, "r")
    if file_name == "yeast.dat":
        data_original = data_file.readlines()[121:]
    elif file_name == "scene.dat":
        data_original = data_file.readlines()[304:]
    data = []
    for line in data_original:
        line = list(map(float, line.strip("/n").split(',')))
        data.append(line)
    return np.array(data)
def load_data():
    '''
    Preprocess the datasets, split the data arrays into X's and y's.
    '''
    file_name1 = "yeast.dat"
    data1 = preprocessing(file_name1)
    Xdat1 = data1[:, :103]
    ydat1 = data1[:, 103:]
    n_dat1,_ = Xdat1.shape
    
    file_name2 = "scene.dat"
    data2 = preprocessing(file_name2)
    Xdat2 = data2[:, :294]
    ydat2 = data2[:, 294:]
    n_dat2,_ = Xdat2.shape
    
    #split the yeast data into training and test set
    proportion = 0.8
    n1 = math.floor(proportion*n_dat1)
    Xtrain1, ytrain1 = Xdat1[:n1,:], ydat1[:n1, :]
    Xtest1, ytest1 = Xdat1[n1:,:], ydat1[n1:, :]
    
    #split the scene data into training and test set
    n2 = math.floor(proportion*n_dat2)
    Xtrain2, ytrain2 = Xdat2[:n2,:], ydat2[:n2, :]
    Xtest2, ytest2 = Xdat2[n2:,:], ydat2[n2:, :]
    
    return (normalize(Xtrain1), ytrain1, normalize(Xtest1), ytest1, normalize(Xtrain2), ytrain2, normalize(Xtest2), ytest2)



def main_knn(n_neighbors, Xtrain, ytrain, Xtest, ytest, method = "L2"):
    '''
    model_knn = KNN(5)
    model_knn.fit(Xtrain, ytrain[:, 0])
    y = model_knn.predict(Xtest)
    error = 1 - np.mean(ytest[:, 0] - y)
    '''
    
    #our implementation
    model_knn = KNN(n_neighbors, method)
    model_knn.fit(Xtrain, ytrain)
    y = model_knn.predict(Xtest)
    #sklearn's implementation
    #model_sklearn_knn = KNeighborsClassifier(n_neighbors =n_neighbors, metric='cosine')
    #model_sklearn_knn.fit(Xtrain,ytrain)
    #y_sklearn = model_sklearn_knn.predict(Xtest)
    #compute hamming_loss 
    print("K = %d, Hamming loss is: %.3f" % (n_neighbors, hamming_loss(ytest, y)))
    
def main_rakel(k_lst, n_model_lst, Xtrain, ytrain, Xtest, ytest, model_method):
    for k in k_lst:
        for n_model in n_model_lst:
            rakel_model = RAKEL(n_classifier = n_model, method = model_method)
            rakel_model.fit(Xtrain, ytrain, k)
            predicted= rakel_model.predict(Xtest)
            print("k = %d, n_model = %d, Hamming loss is: %.3f" % (k, n_model, hamming_loss(ytest, predicted)))
if __name__ == "__main__":
    #load data
    Xtrain1, ytrain1, Xtest1, ytest1, Xtrain2, ytrain2, Xtest2, ytest2 = load_data()
    
    n1, dX1 = Xtrain1.shape
    t1, dy1 = ytest1.shape
    
    n2, dX2 = Xtrain2.shape
    t2, dy2 = ytest2.shape
    #run modules for different methods
    k_set = [2,3,4,5,6,7,8]
    #for k in k_set:
        #main_knn(k, Xtrain2, ytrain2, Xtest2, ytest2, "pearson")
    #m, im = LP_map(dy2)
    #a = transform(m, im, ytrain2)
    #b = reverse_transform(m, im, a, dy2)
    k_lst = [2,3]
    n_model_lst = [5, 10, 15, 20]
    model_method = "mix"
    main_rakel(k_lst, n_model_lst, Xtrain2, ytrain2, Xtest2, ytest2, model_method)
    #k = 2
    #ra = RAKEL(n_classifier = 20, method = "SVM")
    #ra.fit(Xtrain2, ytrain2, 3)
    #r= ra.predict(Xtrain2)
     
    

    
    
    
    


    
        