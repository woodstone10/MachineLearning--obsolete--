###########################################################################################
#
# MachineLearning_ClusteringUsingK-MeansQPSKModulation.py
#
# This is sample code for Machine Learning with Python
# simple example for K-Means algorithm (referred to as Lloyd-Forgy) with Scikit-learn
# for Clustering of unlabeled data in Unsupervised learning techniques
# - QPSK modulation data generated then clustering of Machine learning
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

# QPSK (Quadrature Phase-Shift Keying) modulation generation
# - code from https://pysdr.org/content/digital_modulation.html
def genQPSK(num):
    num_symbols = num # number of symbols
    noise_power = 0.5 # noise ratio
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

# Unlabeled datasheet
n = 100
x, y = genQPSK(n)
print(x.shape, y.shape)
df = pd.DataFrame(columns=['x','y'])
for i in range(len(x)):
    df.loc[i] = [x[i], y[i]]

# Model
# Let’s train a K-Means clusterer on this dataset.
# It will try to find each blob’s center and
# assign each instance to the closest blob:
k = 4
X = df.values
model = KMeans(n_clusters=k) # number of clusters to predict k
model.fit(X) # fitting the model to X
centroids = model.cluster_centers_ # We can also take a look at the 4 centroids that the algorithm found:
print(centroids)
y_pred = model.predict(X) # predicting labels (y) and saving to y_pred
df['cluster'] = model.labels_
print(df.head(n))

# Visualization
# method 1.
# plt.scatter(df['x'], df['y'], c=y_pred)
# plt.colorbar()
# plt.show()
# method 2.
for name, group in df.groupby("cluster"):
    plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name)
plt.legend()
plt.title("Clustering of QPSK modulation data \n using K-Means algorithm with Scikit-learn")
plt.show()
