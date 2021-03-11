###########################################################################################
#
# ML_DeepLearning_ImageClassification_MNIST.py
#
# Comparison FC vs CNN, Deep Learning Neural Network algorithms
# using MNIST dataset (prediction/classification 0 ~ 9 number)
# - FC (Fully Connected) shows overfitting
# - CNN (Convolutional neural networks) also shows overfitting but its value lower than FC
#
# Created by Jonggil Nam
# LinkedIn: https://www.linkedin.com/in/jonggil-nam-6099a162/
# Github: https://github.com/woodstone10
# e-mail: woodstone10@gmail.com
# phone: +82-10-8709-6299
###########################################################################################

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Data from MNIST (black-and-white 2D image)
# In this sample, we will be using the MNIST dataset, which is a set of 70,000 small
# images of digits handwritten by high school students and employees of the US Cen‐
# sus Bureau. Each image is labeled with the digit it represents. This set has been stud‐
# ied so much that it is often called the “Hello World” of Machine Learning: whenever
# people come up with a new classification algorithm
MNIST = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = MNIST.load_data()
# display it using Matplotlib’s imshow() function:
class_names = ["0","1","2","3","4","5","6","7","8","9"]
plt.figure(figsize=(6,6))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.suptitle('MNIST train data', fontsize=15)
plt.tight_layout()
plt.savefig('ML_DeepLearning_ImageClassification_MNIST_data.png')
plt.show()
# There are 70,000 images, and each image has 784 features. This is because each image
# is 28×28 (=784) pixels, and each feature simply represents one pixel’s intensity, from 0
# (white) to 255 (black).
# scaling (0 ~ 255) >> (0 ~ 1)

# -------------------------------------------------------
# FC (Fully Connected) Neural Network Model - overfitting observed
# -------------------------------------------------------
# Let’s take a peek at one digit from the dataset. All you need to
# do is grab an instance’s feature vector, reshape it to a 28×28 array
# FC uses 1D array data
x_train = x_train.reshape(x_train.shape[0], 784).astype(float)/255
x_test = x_test.reshape(x_test.shape[0], 784).astype(float)/255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = \
    tf.keras.Sequential([
        tf.keras.layers.Dense(512, input_dim=784, activation='relu'), # input node 784 (25x25)
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax') # classify 10 classes (0~9)
    ])
model.summary()

model.compile(loss='categorical_crossentropy', # loss function is cross entropy
              optimizer='adam',
              metrics=['accuracy'])

# Training (fitting)
FC = model.fit(x_train, y_train, # image, label
                     validation_split=0.3, # split data into train (70%) and validation (30%)
                     epochs=30, batch_size=200,
                     verbose=2
                     # verbose 0: silence
                     # verbose 1: progressbar
                     # verbose 2: oneline acc, loss
                     )

# model.evaluate(x_test, y_test, verbose=2)

# -------------------------------------------------------
# Convolutional neural networks (CNNs)
# -------------------------------------------------------
(x_train, y_train), (x_test, y_test) = MNIST.load_data()
# CNN uses orignal 2D image data - 4 tensors (batch, width, height, channel)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = \
    tf.keras.Sequential([
        # CNN
        # tf.keras.layers.Conv2D(
        #     filters, kernel_size, strides=(1, 1), padding='valid',
        #     data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
        #     use_bias=True, kernel_initializer='glorot_uniform',
        #     bias_initializer='zeros', kernel_regularizer=None,
        #     bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        #     bias_constraint=None, **kwargs
        # )
        # 26x26, 32
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), input_shape=(28, 28, 1), activation='relu'),
        # 24x24, 64
        tf.keras.layers.Conv2D(64, (3,3), (1,1), activation='relu'),
        # 12x12, 64
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.25),
        # 2D -> 1D
        tf.keras.layers.Flatten(),
        # FC
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # classify to 10 classes
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

CNN = model.fit(x_train, y_train,
                validation_data=(x_test, y_test), epochs=30, batch_size=200, verbose=2)

# Learning curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange((len(FC.history['val_accuracy']))), FC.history['val_accuracy'], label='FC validation accuracy')
plt.plot(np.arange((len(FC.history['accuracy']))), FC.history['accuracy'], label='FC train accuracy')
plt.plot(np.arange((len(CNN.history['val_accuracy']))), CNN.history['val_accuracy'], label='CNN validation accuracy')
plt.plot(np.arange((len(CNN.history['accuracy']))), CNN.history['accuracy'], label='CNN train accuracy')
plt.legend(loc='best')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.subplot(1, 2, 2)
plt.plot(np.arange((len(FC.history['val_loss']))), FC.history['val_loss'], label='FC validation loss')
plt.plot(np.arange((len(FC.history['loss']))), FC.history['loss'], label='FC train loss')
plt.plot(np.arange((len(CNN.history['val_loss']))), CNN.history['val_loss'], label='CNN validation loss')
plt.plot(np.arange((len(CNN.history['loss']))), CNN.history['loss'], label='CNN train loss')
plt.legend(loc='best')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.suptitle('Learning curve - FC vs. CNN \n for MNIST data', fontsize=15)
plt.tight_layout()
plt.savefig('ML_DeepLearning_ImageClassification_MNIST_learning_curve.png')
plt.show()
