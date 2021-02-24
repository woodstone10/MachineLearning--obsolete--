###########################################################################################
#
# MachineLearning_NeuralNetworkUsingKerasMultilayerExample.py
#
# This is sample code for Machine Learning with Python
# simple example for multi-layer Neural network with Keras
# for Building an Image Classifier Using the Sequential API in Unsupervised learning techniques
# - this example is presented on book in chapter 10 (Introduction to Artificial Neural Networks
#   with Keras,
#   "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)
# - code from https://www.tensorflow.org/tutorials/keras/classification
#
# Neural network architecture with Keras
# ------------------------------------------------------------
#                        Dense
#             Input      Hidden     Output
#             Layer      Layer      Layer
# Input 1
#                  Weights    Weights
# Input 2                                   --->Target
#   ...            Weights    Weights
# Input n
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
print(tf.__version__)
from pltimg import plot_image, plot_value_array

# Using Keras to Load the Dataset
# In this sample, we will be using the MNIST dataset, which is a set of 70,000 small
# images of digits handwritten by high school students and employees of the US Census
# Bureau. Each image is labeled with the digit it represents. This set has been studied
# so much that it is often called the “Hello World” of Machine Learning: whenever
# people come up with a new classification algorithm, they are curious to see how it
# will perform on MNIST. Whenever someone learns Machine Learning, sooner or
# later they tackle MNIST.
# Keras provides some utility functions to fetch and load common datasets, including
# MNIST, Fashion MNIST, the original California housing dataset, and more. Let’s load
# Fashion MNIST:
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # (set of train), (set of test)
# When loading MNIST or Fashion MNIST using Keras rather than Scikit-Learn, one
# important difference is that every image is represented as a 28×28 array rather than a
# 1D array of size 784.
print("train data:", train_images.shape, train_labels.shape) #.shape used for tf.keras.layers.Flatten(input_shape=(28, 28)),
print("test data:", test_images.shape, test_labels.shape)

# For simplicity, we just
# scale the pixel intensities down to the 0-1 range by dividing them by 255.0 (this also
# converts them to floats):
train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# show train data
plt.figure(figsize=(6,6))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.suptitle('train data', fontsize=15)
plt.tight_layout()
plt.savefig('MachineLearning_NeuralNetworkUsingKerasMultilayerExample_train.png')
plt.show()

# Model: creating the Model Using the Sequential API
# Keras is a high-level Deep Learning API that allows you to easily build, train, evaluate
# and execute all sorts of neural networks.
model = \
    tf.keras.Sequential([
        # The first line creates a Sequential model. This is the simplest kind of Keras
        # model, for neural networks that are just composed of a single stack of layers, connected
        # sequentially. This is called the sequential API.
    tf.keras.layers.Flatten(input_shape=(28, 28)),
        # Next, we build the first layer and add it to the model. It is a Flatten layer whose
        # role is simply to convert each input image into a 1D array: if it receives input data
        # X, it computes X.reshape(-1, 1). This layer does not have any parameters, it is
        # just there to do some simple preprocessing. Since it is the first layer in the model,
        # you should specify the input_shape: this does not include the batch size, only the
        # shape of the instances. Alternatively, you could add a keras.layers.InputLayer
        # as the first layer, setting shape=[28,28].
    tf.keras.layers.Dense(300, activation='relu'),
        # Next we add a Dense hidden layer with 300 neurons. It will use the ReLU activation
        # function. Each Dense layer manages its own weight matrix, containing all the
        # connection weights between the neurons and their inputs. It also manages a vec‐
        # tor of bias terms (one per neuron). When it receives some input data, it computes
        # Equation 10-2.
        #
        # Other activation functions are
        # available in the keras.activations package, we will use many of
        # them in this book. See https://keras.io/activations/ for the full list.
        # - relu function : ReLU (Rectified Linear Unit) Function
        # - sigmoid function
        # - softmax function
        # - softplus function
        # - softsign function
        # - tanh function
        # - selu function
        # - elu function
        # - exponential function
    tf.keras.layers.Dense(100, activation='relu'),
        # Next we add a second Dense hidden layer with 100 neurons, also using the ReLU
        # activation function.
    tf.keras.layers.Dense(10, activation='softmax')
        # Finally, we add a Dense output layer with 10 neurons (one per class), using the
        # softmax activation function (because the classes are exclusive).
])

# Compiling the Model
# Keras Optimizer
# - Adam: Adaptive moment estimation, RMSprop + Momentum,
#   works well in practice and outperforms other Adaptive techniques.
#   Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum)
#   Usually works well even with little tuning of hyperparameters.
# - SGD: Stochastic gradient descent, SGD + Nesterov enabled
#   it works well for shallow networks.
#   Nesterov accelerated gradient (NAG)
#   Intuition how it works to accelerate gradient descent.
#   We’d like to have a smarter ball, a ball that has a notion of where
#   it is going so that it knows to slow down before the hill slopes up again.
# - Adagrad
# - AdaDelta
model.compile(optimizer='adam',
              #optimizer="sgd",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Let's train with train data, fitting model to train datas
# The neural network is trained. At each epoch during training, Keras displays
# the number of instances processed so far (along with a progress bar), the mean
# training time per sample, the loss and accuracy (or any other extra metrics you asked
# for), both on the training set and the validation set. You can see that the training loss
# went down, which is a good sign, and the validation accuracy reached 80~90% after k
# epochs, not too far from the training accuracy, so there does not seem to be much
# overfitting going on.
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(train_images, train_labels))

# you get the learning curves
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# Validation with Test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

# show test data
plt.figure(figsize=(6,6))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
plt.suptitle('test data', fontsize=15)
plt.tight_layout()
plt.savefig('MachineLearning_NeuralNetworkUsingKerasMultilayerExample_test.png')
plt.show()

# Using the Model to Make Predictions
# Next, we can use the model’s predict() method to make predictions on new instances.
# As you can see, for each instance the model estimates one probability per class, from
# class 0 to class x. For example, for the first image it estimates that the probability of
# class 9 (ankle boot) is 79%, the probability of class 7 (sneaker) is 12%, the probability
# of class 5 (sandal) is 9%, and the other classes are negligible. In other words, it
# “believes” it’s footwear, probably ankle boots, but it’s not entirely sure, it might be
# sneakers or sandals instead. If you only care about the class with the highest estimated
# probability (even if that probability is quite low) then you can use the pre
# dict_classes() method instead:
predictions = model.predict(test_images)
print(predictions.round(2))

# show prediction result with test data
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(10,9)) #figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images, class_names)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
    if i >= (num_rows-1)*3: # display x-axis at last row only
        _ = plt.xticks(range(10), class_names, rotation=90)
plt.suptitle('prediction of test data', fontsize=15)
plt.tight_layout()
plt.savefig('MachineLearning_NeuralNetworkUsingKerasMultilayerExample_predict.png')
plt.show()
