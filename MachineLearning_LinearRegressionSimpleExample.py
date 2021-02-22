###########################################################################################
#
# MachineLearning_LinearRegressionSimpleExample.py
#
# This is sample code for Machine Learning with Python
# simple example for Linear Regression using TensorFlow
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
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Data (simple example)
x = [1, 2, 3, 4, 5]
y = [2.1, 3.7, 6.9, 7.9, 10.4]

# Model
W = tf.Variable(tf.random_uniform([1],-100,100)) # weight
b = tf.Variable(tf.random_uniform([1],-100,100)) # bias
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W * X + b # Linear Regression
cost = tf.reduce_mean(tf.square(H - Y)) # cost/loss
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# FIT
for i in range(5001):
    sess.run(train, feed_dict={X:x, Y:y})
    if i%100 == 0:
        print(i, sess.run(cost, feed_dict={X:x, Y:y}), sess.run(W), sess.run(b))

# Prediction
print("6:",sess.run(H, feed_dict={X:6}))
print("100:",sess.run(H, feed_dict={X:100}))
