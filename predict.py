import numpy as np
import sigmoid as sg

def predict(w, b, X):
    m = X.shape[0]
    y_prediction = np.zeros((m, 1))
    A = sg.sigmoid(np.dot(X, w) + b)
    for i in range(A.shape[0]):
        if (A[i, 0] <= 0.5):
            y_prediction[i, 0] = 0
        else:
            y_prediction[i, 0] = 1
            
    return y_prediction

