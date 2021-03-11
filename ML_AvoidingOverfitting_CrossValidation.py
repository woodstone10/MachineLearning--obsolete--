###########################################################################################
#
# ML_AvoidingOverfitting_CrossValidation.py
#
#
# Created by Jonggil Nam
# LinkedIn: https://www.linkedin.com/in/jonggil-nam-6099a162/
# Github: https://github.com/woodstone10
# e-mail: woodstone10@gmail.com
# phone: +82-10-8709-6299
###########################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.preprocessing
import sklearn.model_selection

# Data
df = pd.read_csv('ML_AvoidingOverfitting_CrossValidation.csv', header=None)
dataset = df.values
x_data = dataset[:,0:60].astype(float)
y_data = dataset[:,60]
print(y_data) # R,M

# target label (y) encoding and transform
e = sklearn.preprocessing.LabelEncoder()
e.fit(y_data)
y_data = e.transform(y_data)
print(y_data) # 0,1

# Model
def NeuralNetworkWithKeras():
    model = \
        tf.keras.Sequential([
            tf.keras.layers.Flatten(input_dim=60),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# Training with Cross-validation (K-fold), different accuracy on different train data set
k_fold = 10
skf = sklearn.model_selection.StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=48)

accuracy = []
for train, test in skf.split(x_data, y_data): # split data for cross-validation
    model = NeuralNetworkWithKeras()
    model.fit(x_data[train], y_data[train], epochs=100, batch_size=5)
    k_accuracy = "%.3f" % (model.evaluate(x_data[test], y_data[test])[1])
    accuracy.append(k_accuracy)

print("\n %.f fold accuracy:" % k_fold, accuracy)
