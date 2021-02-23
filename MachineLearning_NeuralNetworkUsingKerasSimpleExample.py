###########################################################################################
#
# MachineLearning_NeuralNetworkUsingKerasSimpleExample.py
#
# This is sample code for Machine Learning with Python
# simple example for single-layer Neural network with Keras
# for Linear Regression in Supervised Training Models
#
# Created by Jonggil Nam
# LinkedIn: https://www.linkedin.com/in/jonggil-nam-6099a162/
# Github: https://github.com/woodstone10
# e-mail: woodstone10@gmail.com
# phone: +82-10-8709-6299
###########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # required Installing TensorFlow 2

# Train data (simple example)
x = [1, 2, 3, 4, 5] # independent var
y = [2.1, 3.7, 6.9, 7.9, 10.4] # dependent var (y = ~2 * x)

# Model with Keras
X = tf.keras.layers.Input(shape=[1]) # [number of independent var]
Y = tf.keras.layers.Dense(1)(X) # (number of dependent var) (independent var)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse') # Compile

# Let's train (FIT)
model.fit(x,y,epochs=3000) # You can see reducing of loss

# Prediction
print("Predictions:", model.predict([1, 2, 3, 4, 5, 6]))
