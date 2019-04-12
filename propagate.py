import numpy as np 
import sigmoid as sg

def propagate(w, b, X, y):
    
    m = X.shape[0]
    A = sg.sigmoid(np.dot(X, w) + b)
    cost = -(1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A)) #computing the cost function or the error function
    dw = (1 / m) * np.dot(X.T, (A - y))   #this is derivative of the cost function with respect to w
    db = (1 / m) * np.sum(A - y)          #this is the derivative of the cost function with respect to b
    grads = {'dw': dw, 'db': db}          #these values are stored in a dictionary so as to access them later
    return grads, cost 