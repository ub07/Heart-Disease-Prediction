import initialize_with_zeros as iz
import optimize as op
import predict as pr
import numpy as np
import data 

def model(X_train, X_test, y_train, y_test, num_iterations, learning_rate, print_cost = True):
    w, b = iz.initialize_with_zeros(data.X.shape[1])
    parameters, grads, costs = op.optimize(w, b, data.X, data.y, num_iterations, learning_rate, print_cost = True)
    w = parameters["w"]
    b = parameters["b"]
    y_prediction_test = pr.predict(w, b, X_test)
    y_prediction_train = pr.predict(w, b, X_train)
    
    print('train accuracy: {}'.format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print('test accuracy: {}'.format("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)))
    
    d = {"costs": costs,
         "y_prediction_test": y_prediction_test, 
         "y_prediction_train" : y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(data.X_train, data.X_test, data.y_train, data.y_test, num_iterations = 100000, learning_rate = 0.00015, print_cost = True)