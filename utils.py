import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
import itertools
import math

def mode(y):
    """Computes the element with the maximum count
    Parameters
    ----------
    y : an input numpy array
    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y)==0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]
        
def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """

    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)

    # without broadcasting:
    # n,d = X.shape
    # t,d = Xtest.shape
    # D = X**2@np.ones((d,t)) + np.ones((n,d))@(Xtest.T)**2 - 2*X@Xtest.T
def cosine_similarity(X, Xtest):
    '''
    Calculate the cosine similarity, suppose X and Xtest have already been normalized.
    '''
    return -np.dot(X,Xtest.T)
def pearson_corr(X, Xtest):
    n,d = X.shape
    t, _ = Xtest.shape
    corr = np.zeros((n, t))
    for i in range(n):
        for j in range(t):
            corr[i,j] = np.abs(np.corrcoef(X[i,:], Xtest[j,:])[1,0])
    return - corr

def cross_valid(X, y, n_fold):
    '''
    Split the data into n_fold, helper function to do cross validation.
    Assume there are equal number of examples in both X and y.
    '''
    n= X.shape[0]
    X_folds = []
    y_folds = []
    n_per_fold  = math.floor(n/n_fold)
    for n in range(n_fold):
        if n == 0:
            foldx = X[:n_per_fold, ]
            foldy = y[:n_per_fold, ]
            X_folds.append(foldx)
            y_folds.append(foldy)
        elif n == n_fold - 1:
            index = n_per_fold*(n_fold - 1)
            foldx = X[index:,]
            foldy = y[index:,]
            X_folds.append(foldx)
            y_folds.append(foldy)
        else:
            begin = n*n_per_fold
            end = (n+1)*n_per_fold
            foldx = X[begin:end,]
            foldy = y[begin:end,]
            X_folds.append(foldx)
            y_folds.append(foldy)
    
    return (X_folds, y_folds)

def get_fold(X_lst, y_lst, current_fold):
    Xtrain = np.vstack(X_lst[:current_fold] + X_lst[current_fold+1:])
    Xvalid =  X_lst[current_fold]
    ytrain =  np.vstack(y_lst[:current_fold] + y_lst[current_fold+1:])
    yvalid =  y_lst[current_fold]
    
    return (Xtrain, ytrain, Xvalid, yvalid)
def LP_map(n_label):
    '''
    Helper function to convert the multilabel question into multiclass question.
    i.e. Label power set.
    '''
    lst = list(map(list, itertools.product([0, 1], repeat=n_label)))
    mapping = {}
    inverse_map = {}
    for i in range(len(lst)):
        #mapping is a dict { int, tuple(original label)}
        mapping[i] = tuple(lst[i])
        #inverse mapping is a dict {tuple(original label):int}
        inverse_map[tuple(lst[i])] = i 
    return (mapping, inverse_map)
    
        
def transform(mapping, inverse_map, y):
    '''
    Transform y into multiclass.
    
    y is of shape (n, n_labels)
    '''
    n, n_labels = y.shape
    y_new_int = np.zeros(n)
    for i in range(n):
        y_new_int[i] = inverse_map[tuple(y[i,:])]
    return y_new_int

def reverse_transform(mapping, inverse_map, y, n_label):
    '''
    Reverse the transformation.
    y is of shape (n, )
    Example: 
    m, im = LP_map(dy2)
    a = transform(m, im, ytrain2)
    b = reverse_transform(m, im, a, dy2)
    '''
    n = y.shape[0]
    y_new = np.zeros((n, n_label))
    for i in range(n):
        y_new[i,:] = mapping[y[i]]
        
    return y_new


