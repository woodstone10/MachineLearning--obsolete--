###########################################################################################
#
# MachineLearning_ClusteringUsingDBSCAN.py
#
# This is sample code for Machine Learning with Python
# for Clustering of unlabeled data (Unsupervised learning techniques)
#
# Comparison of clustering algorithms between K-Means and DBSCAN
#   1. K-Means
#       : The K-Means algorithm is a simple algorithm capable of clustering this
#       kind of dataset very quickly and efficiently, often in just a few iterations.
#       Suppose you were
#       given the centroids: you could easily label all the instances in the dataset by assigning
#       each of them to the cluster whose centroid is closest. Conversely, if you were given all
#       the instance labels, you could easily locate all the centroids by computing the mean of
#       the instances for each cluster. But you are given neither the labels nor the centroids,
#       so how can you proceed? Well, just start by placing the centroids randomly (e.g., by
#       picking k instances at random and using their locations as centroids). Then label the
#       instances, update the centroids, label the instances, update the centroids, and so on
#       until the centroids stop moving. The algorithm is guaranteed to converge in a finite
#       number of steps (usually quite small), it will not oscillate forever2.
#   2. DBSCAN
#       : In short, DBSCAN is a very simple yet powerful algorithm, capable of identifying any
#       number of clusters, of any shape, it is robust to outliers, and it has just two hyper‐
#       parameters (eps and min_samples). However, if the density varies significantly across
#       the clusters, it can be impossible for it to capture all the clusters properly. Moreover,
#       its computational complexity is roughly O(m log m), making it pretty close to linear
#       with regards to the number of instances. However, Scikit-Learn’s implementation can
#       require up to O(m2) memory if eps is large.
#
# Two examples
#   1. QPSK modulation datas
#       - K-Means gets the clustering, which looks perfect.
#       Whereas, DBSCAN algorithm works well if all the clusters are dense enough, and they are well sepa‐
#       rated by low-density regions, that is low noise and high epsilon
#   2. Scikit-learn datasets make_moons (DBSCAN clustering using two different neighborhood radiuses)
#       - DBSCAN gets the clustering, which looks perfect.
#       Whereas, K-Means does not get proper clustering in this case
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
from sklearn.cluster import KMeans # please notice that, in the packages search 'Scikit-learn', instead 'sklearn'
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

def genQPSK(num_samples, noise_ratio):
    # QPSK (Quadrature Phase-Shift Keying) modulation generation
    # - code from https://pysdr.org/content/digital_modulation.html
    num_symbols = num_samples
    noise_power = noise_ratio/100
    x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
    x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
    x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
    n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2) # AWGN with unity power
    r = x_symbols + n * np.sqrt(noise_power)
    # print(r)
    # plt.plot(np.real(r), np.imag(r), '.')
    # plt.grid(True)
    # plt.show() # you will find 4 blobs in form of QPSK
    return np.real(r), np.imag(r)

def KMeans_(k, X, c):
    # K-Means clustering algorithm
    #   input: k is the number of clusters to predict k, X is train data, c is cluster name
    #   Let’s train a K-Means clusterer on this dataset.
    #   It will try to find each blob’s center and
    #   assign each instance to the closest blob:
    model = KMeans(n_clusters=k)
    model.fit(X)  # fitting the model to X
    centroids = model.cluster_centers_  # We can also take a look at the k centroids that the algorithm found:
    print(centroids)
    model.predict(X)  # predicting labels (y)
    df[c] = model.labels_
    print(df.head(10))

def DBSCAN_(e, m, X, c):
    # DBSCAN clustering algorithm
    #   input: e is epsilon, m is min_samples, X is train data, c is cluster name
    #   For each instance, the algorithm counts how many instances are located within a
    #   small distance ε (epsilon) from it. This region is called the instance’s ε-
    #   neighborhood.
    #   If an instance has at least min_samples instances in its ε-neighborhood (includ‐
    #   ing itself), then it is considered a core instance. In other words, core instances are
    #   those that are located in dense regions.
    model = DBSCAN(eps=e, min_samples=m)
    model.fit(X)
    model.fit_predict(X)
    df[c] = model.labels_
    print(df.head(10))

#
# Example 1) QPSK modulation data
#
n_sample = 100
noise_ratio = 2 # noise ratio % >> /100
x, y = genQPSK(n_sample, noise_ratio)
print(x.shape, y.shape)
df = pd.DataFrame(columns=['x','y'])
for i in range(len(x)):
    df.loc[i] = [x[i], y[i]]
X = df.values

KMeans_(4, X, 'cluster1')
DBSCAN_(0.3, 10, X, 'cluster2')

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.scatter(x,y)
plt.legend()
plt.title("Original data of QPSK modulation")
plt.subplot(1,3,2)
for name, group in df.groupby("cluster1"):
    plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name)
plt.legend()
plt.title("Clustering of QPSK modulation datas \n using K-Means algorithm")
plt.subplot(1,3,3)
for name, group in df.groupby("cluster2"):
    plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name)
plt.legend()
plt.title("Clustering of QPSK modulation datas \n using DBSCAN algorithm")
plt.savefig("MachineLearning_ClusteringUsingDBSCAN_QPSK.png")
plt.show()

#
# Example 2) Scikit-learn datasets make_moons
#   DBSCAN clustering using two different neighborhood radiuses
#
x, y = make_moons(n_samples=1000, noise=0.05)
df = pd.DataFrame(columns=['x','y'])
for i in range(len(x)):
    df.loc[i] = [x[i,0], x[i,1]]
X = df.values

KMeans_(2, X, 'cluster3')
DBSCAN_(0.2, 5, X, 'cluster4')

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.scatter(x[:,0],x[:,1])
plt.legend()
plt.title("Original data of make_moons")
plt.subplot(1,3,2)
for name, group in df.groupby("cluster3"):
    plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name)
plt.legend()
plt.title("Clustering of make_moons \n using K-Means algorithm")
plt.subplot(1,3,3)
for name, group in df.groupby("cluster4"):
    plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name)
plt.legend()
plt.title("Clustering of make_moons \n using DBSCAN algorithm")
plt.savefig("MachineLearning_ClusteringUsingDBSCAN_make_moons.png")
plt.show()
