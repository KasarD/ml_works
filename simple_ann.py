#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:30:45 2018

@author: agaev
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('5ppm_data.csv', sep=';')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Encoding categorical data (gas name)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y.reshape(-1, 1)).toarray()

# Split dataset into two parts: traning and validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Force move data to the same scale. It helps us to remove unnecessary
# impact in math calculations, because all data has same priority
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build classifier model and store it in function object to be able to
# pass it in best parameters matcher
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer='adam'):
    # Create ANN
    classifier = Sequential()
    # Add 2 hidden layers and one output layer with 3 end classes (because we have
    # three different gases in dataset)
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_classifier)

# Create list with parameters which will be testing
parameters = {'batch_size': [5, 15],
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

# For benchmark fix start time
t_start = datetime.datetime.now()
grid_search = grid_search.fit(X_train, y_train)
t_stop = datetime.datetime.now() - t_start
print('Time for parameters tuning: {}'.format(t_stop))

# And now we have best accuracy with target params (abount 86%)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Best params list: {}".format(best_params))
print("Best accuracy score: {}".format(best_accuracy))
