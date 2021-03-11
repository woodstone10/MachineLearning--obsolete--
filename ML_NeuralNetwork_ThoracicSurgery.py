###########################################################################################
#
# ML_NeuralNetwork_ThoracicSurgery.py
#
# Thoracic Surgery prediction example:
# The data is dedicated to classification problem related to the post-operative life
# expectancy in the lung cancer patients after thoracic surgery
# in which there are two classes class 1 - the death of patients
# within one year after surgery and class 2 – the patients who survive.
# The data was collected retrospectively at Wroclaw Thoracic Surgery Centre
# for patients who underwent major lung resections for primary lung cancer in the years 2007 to 2011.
# The Centre is associated with the Department of Thoracic Surgery of the Medical University
# of Wroclaw and Lower-Silesian Centre for Pulmonary Diseases, Poland,
# while the research database constitutes a part of the National Lung Cancer Registry,
# administered by the Institute of Tuberculosis and Pulmonary Diseases in Warsaw, Poland.
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
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Data from the UC Irvine Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data
# Attribute Information:
# id,DGN,PRE4,PRE5,PRE6,PRE7,PRE8,PRE9,PRE10,PRE11,PRE14,PRE17,PRE19,PRE25,PRE30,PRE32,AGE,Risk1Yr
# 1. DGN: Diagnosis - specific combination of ICD-10 codes for primary and secondary
#       as well multiple tumours if any (DGN3,DGN2,DGN4,DGN6,DGN5,DGN8,DGN1)
# 2. PRE4: Forced vital capacity - FVC (numeric)
# 3. PRE5: Volume that has been exhaled at the end of the first second of forced expiration
#       - FEV1 (numeric)
# 4. PRE6: Performance status - Zubrod scale (PRZ2,PRZ1,PRZ0)
# 5. PRE7: Pain before surgery (T,F)
# 6. PRE8: Haemoptysis before surgery (T,F)
# 7. PRE9: Dyspnoea before surgery (T,F)
# 8. PRE10: Cough before surgery (T,F)
# 9. PRE11: Weakness before surgery (T,F)
# 10. PRE14: T in clinical TNM - size of the original tumour,
#       from OC11 (smallest) to OC14 (largest) (OC11,OC14,OC12,OC13)
# 11. PRE17: Type 2 DM - diabetes mellitus (T,F)
# 12. PRE19: MI up to 6 months (T,F)
# 13. PRE25: PAD - peripheral arterial diseases (T,F)
# 14. PRE30: Smoking (T,F)
# 15. PRE32: Asthma (T,F)
# 16. AGE: Age at surgery (numeric)
# 17. Risk1Y: 1 year survival period - (T)rue value if died (T,F)

# data frame with format of csv, converted from arff
df = pd.read_csv('ThoraricSurgery.csv')
print(df.shape, "\n", df.head())
df1 = df
df1.replace({
    'T':1,
    'F':0,
    'PRZ0':0,
    'PRZ1':1,
    'PRZ2':2,
    'OC10':0,
    'OC11':1,
    'OC12':2,
    'OC13':3,
    'OC14':4,
    'OC15':5,
},inplace=True)
print(df1.head(), df1.dtypes)

# Data analysis
fig = plt.figure(figsize = (9,9))
ax = fig.gca()
df1.hist(ax = ax)
plt.tight_layout()
plt.savefig("ML_NeuralNetwork_ThoracicSurgery_hist")

plt.figure(figsize=(14,9))
sns.heatmap(df1.corr(), fmt='.1%', annot=True)
plt.tight_layout()
plt.savefig("ML_NeuralNetwork_ThoracicSurgery_corr")
plt.show()

# Train data
x = df1.loc[:,['PRE4','PRE5','PRE6','PRE7','PRE8','PRE9','PRE10',
               'PRE11','PRE14','PRE17','PRE19','PRE25','PRE30','PRE32','AGE']]
y = df1['Risk1Yr']
# x = df1.values[:,2:17]
# y = df1.values[:,17]

# Model using Keras high-level API
model = \
    tf.keras.Sequential([
    tf.keras.layers.Flatten(input_dim=15),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(x, y, epochs=50, batch_size=10)

# Model evaluation
loss, acc = model.evaluate(x, y, verbose=2)
print('Loss:',loss,'Accuracy:',acc)

# This problem also can be solved by Logistic Regression, SVM, Boost and so on
# example, refer to https://www.youtube.com/watch?v=Tnfmvxtz9rc
