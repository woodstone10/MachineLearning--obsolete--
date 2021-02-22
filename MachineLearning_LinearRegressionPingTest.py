###########################################################################################
#
# MachineLearning_LinearRegressionPingTest.py
#
# This is sample code for Machine Learning with Python
# simple Linear Regression
# with Real Data (Spirent LTE Ping delay test)
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

# Data from CSV
df = pd.read_csv('L760DL LTE Data Ping Moving Re-Test.csv') # for example, Spirent Moving Ping Test
print(df.shape) # (row, col)
print(df.columns) # col names
print(df.head()) # display top 5
x = df[['MeanRoundTripTime']] #.dropna() # dependant var
y = df[['MaximumRoundTripTime']] #.dropna() # independant var

print(x.shape, y.shape)
print(x)
print(y)

# Find Training Model
# plt.subplot(3,1,1)
# plt.plot(x)
# plt.subplot(3,1,2)
# plt.plot(y)
# plt.subplot(3,1,3)
# plt.plot(x, y)
# plt.show()

# Model
W = tf.Variable(tf.random_uniform([1],-100,100)) # weight
b = tf.Variable(tf.random_uniform([1],-100,100)) # bias
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W * X + b # Linear Regression
cost = tf.reduce_mean(tf.square(H - Y)) # cost/loss
a = tf.Variable(0.0001)
optimizer = tf.train.GradientDescentOptimizer(a) # Gradient Descent
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session() #Session
sess.run(init)

# FIT
count = []
error = []
for i in range(1000001):
    sess.run(train, feed_dict={X:x, Y:y})
    if i%100 == 0:
        count.append(i)
        error.append(sess.run(cost, feed_dict={X:x, Y:y}))
        print(i, sess.run(cost, feed_dict={X:x, Y:y}), sess.run(W), sess.run(b))

plt.subplot(2,1,1)
plt.plot(x, y)
plt.subplot(2,1,2)
plt.plot(count, error)
plt.show()

# Prediction
print("40:",sess.run(H, feed_dict={X:40}))
print("50:",sess.run(H, feed_dict={X:50}))
print("60:",sess.run(H, feed_dict={X:60}))
