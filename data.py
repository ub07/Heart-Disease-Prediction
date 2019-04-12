import pandas as pd                  #importing the library for data manipulation and storage 
import numpy as np                   #importing the library for scientific computation
from sklearn.model_selection import train_test_split   #importing train_test_split which we would use later  

df = pd.read_csv('data/heart.csv')   #reading the csv from the directory and storing the values in df


X = df.drop(['target'], axis = 1)      #here, we would take all the columns except 'target' as input vector
y = df['target']                       #here, we are taking the output as the 'target' column in our dataset
ynewtest = y
xnewtest = X
y = y[:, np.newaxis]                   #converting the output to an array 

#dividing the giving input and output values into 2 sets namely the training set and the test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)

