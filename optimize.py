import numpy as np
import propagate as pp
import matplotlib.pyplot as plt      #importing the library used for plots (not high end)

#We are trying to get the parameters w and b after modifying them using the knowledge of the cost function
def optimize(w, b, X, y, num_iterations, learning_rate, print_cost = False):
    costs = []                    #This is an empty list created so that it stores all the values later
    for i in range(num_iterations):
        grads, cost = pp.propagate(w, b, X, y)       #we are calling the previously defined function 
        dw = grads['dw']                          #we are accessing the derivatives of cost with respect to w
        db = grads['db']                          #we are accessing the derivatives of cost with respect to b
        w = w - learning_rate * dw                #we are modifying the parameter w so that the cost would reduce in the long run
        b = b - learning_rate * db                #we are modifying the parameter b so that the cost would reduce in the long run
        np.squeeze(cost)
        if i % 100 == 0:
            costs.append(cost)                    #we are giving all the cost values to the empty list that was created initially
        if print_cost and i % 1000 == 0:
            print("cost after iteration {}: {}".format(i, cost))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    params = {'w': w, 'b': db}                    #we are storing this value in the dictionary so that it could be accessed later
    grads = {'dw': dw, 'db': db}                  #we are storing these valeus in the dictionary so that they could be accessed later
    return params, grads, costs