###########################################################################################
#
# MachineLearning_BostonHousingPrices.py
#
# This is sample code for Machine Learning with Python
# Prediction example for the Boston Housing Prices dataset
# refer to
#   https://medium.com/analytics-vidhya/boston-house-price-prediction-using-machine-learning-ad3750a866cd
#   https://www.kaggle.com/shreayan98c/boston-house-price-prediction
# Comparison between algorithms
#   1. Neural Network (TensorFlow Keras)
#   2. Linear Regression (Scikit-learn)
#       : To find the value of θ that minimizes the cost function, there is a closed-form solution
#       —in other words, a mathematical equation that gives the result directly. This is called
#       the Normal Equation
#   3. SVM (Support Vector Machines) Regression (Scikit-learn)
#       : SVM algorithm is quite versatile: not only does it sup‐
#       port linear and nonlinear classification, but it also supports linear and nonlinear
#       regression.
#   4. Random Forest Regression (Scikit-learn)
#       : Random Forest is an ensemble of Decision Trees, generally
#       trained via the bagging method
#   5. XGBoost Regression (XGBoost)
#       : The general idea of most
#       boosting methods is to train predictors sequentially, each trying to correct its prede‐
#       cessor.
#       In fact, XGBoost is often an important component of the winning
#       entries in ML competitions. XGBoost’s API is quite similar to Scikit-Learn’s
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
import seaborn as sns
from sklearn.datasets import load_boston # train data
from sklearn import metrics
import tensorflow as tf # model for Keras Neural Network
from sklearn.linear_model import LinearRegression # model for Linear Regression
from sklearn.ensemble import RandomForestRegressor # model for Random Forest Regressor
from xgboost import XGBRegressor # modem for XGBoost Regressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm # model for SVM REgressor

# Train data, load from sklearn dataset load_boston()
# As well we can also able to get the dataset from the sklearn datasets.
# Yup! It’s available into the sklearn Dataset.
boston = load_boston()
x = boston['data']
y = boston['target']
features = boston['feature_names']
# variables
# 1. CRIM per capital crime rate by town
# 2. ZN proportion of residential land zoned for lots over 25,000 sq.ft.
# 3. INDUS proportion of non-retail business acres per town
# 4. CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 5. NOX nitric oxides concentration (parts per 10 million)
# 6. RM average number of rooms per dwelling
# 7. AGE proportion of owner-occupied units built prior to 1940
# 8. DIS weighted distances to five Boston employment centers
# 9. RAD index of accessibility to radial highways
# 10.TAX full-value property-tax rate per 10,000 USD
# 11. PTRATIO pupil-teacher ratio by town
# 12. B 1000(Bk — 0.63)² where Bk is the proportion of blacks by town
# 13. LSTAT % lower status of the population
print(features)
print(x.shape, y.shape)

# Data frame with Pandas
df = pd.DataFrame(x, columns=features, index=None)
df['Price'] = y
print(df.head())
print(df.describe())
corr = df.corr() # Correlation between parameters
print("Correlation:\n",corr)
plt.figure(figsize=(14,9))
sns.heatmap(corr, fmt='.2%', annot=True)
plt.title("Correlation between parameters")
plt.tight_layout()
plt.savefig('MachineLearning_BostonHousingPrices_Correlation.png')

# Model 1. Keras Neural Network
X = tf.keras.layers.Input(shape=[13]) # [number of independent var]
Y = tf.keras.layers.Dense(1)(X) # (number of dependent var) (independent var)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse') # Compile
model.fit(x,y,epochs=3000) # Training
y_pred_tf_keras = model.predict(x)
accuracy_tf_keras = metrics.r2_score(y, y_pred_tf_keras)

# Model 2. Scikit-learn Linear Regression
model = LinearRegression()
model.fit(x,y)
y_pred_sklearn_linearRegression = model.predict(x)
# print("Training Accuracy:", model.score(x,y)*100)
accuracy_sklearn_linearRegression = metrics.r2_score(y, y_pred_sklearn_linearRegression)
# weights = model.get_weights()
# print("weight:\n",weights)

# Model 3. SVM (Support Vector Machines) Regressor
sc = StandardScaler()
x_ = sc.fit_transform(x)
model = svm.SVR()
model.fit(x_,y)
y_pred_sklearn_svmRegressor = model.predict(x_)
accuracy_sklearn_svmRegressor = metrics.r2_score(y, y_pred_sklearn_svmRegressor)

# Model 4. Random Forest Regressor
model = RandomForestRegressor()
model.fit(x,y)
y_pred_sklearn_RandomForestRegressor = model.predict(x)
accuracy_sklearn_RandomForestRegressor = metrics.r2_score(y, y_pred_sklearn_RandomForestRegressor)

# Model 5. XGBoost Regressor
model = XGBRegressor()
model.fit(x,y)
y_pred_xgboost_xgbRegressor = model.predict(x)
accuracy_xgboost_xgbRegressor = metrics.r2_score(y, y_pred_xgboost_xgbRegressor)

# Comparison between Models
models = pd.DataFrame({
    'Model': ['Keras Neural Network',
              'Linear Regression',
              'SVM Regression',
              'Random Forest Regression',
              'XGBoost Regression'],
    'R-squared Score': [accuracy_tf_keras*100,
                        accuracy_sklearn_linearRegression*100,
                        accuracy_sklearn_svmRegressor*100,
                        accuracy_sklearn_RandomForestRegressor*100,
                        accuracy_xgboost_xgbRegressor*100
                        ]})
print(models.sort_values(by='R-squared Score', ascending=False))
plt.figure(figsize=(5,5))
plt.bar(models['Model'], models['R-squared Score'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('MachineLearning_BostonHousingPrices_Accuracy.png')

# Data visualizaiton
plt.figure(figsize=(9,9))
plt.subplot(5,2,1)
plt.scatter(y, y_pred_tf_keras, color="green", label="epochs=3000")
plt.title("Prices vs. Predicted prices \n with TensorFlow Keras")
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.legend()
plt.subplot(5,2,2)
sns.distplot(y-y_pred_tf_keras)
plt.title("Histogram of Residuals \n with TensorFlow Keras")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.subplot(5,2,3)
plt.scatter(y, y_pred_sklearn_linearRegression, color="blue")
plt.title("prices vs. predicted prices \n with Linear Regression (Scikit-learn)")
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.subplot(5,2,4)
sns.distplot(y-y_pred_sklearn_linearRegression)
plt.title("Histogram of Residuals \n with Linear Regression")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.subplot(5,2,5)
plt.scatter(y, y_pred_sklearn_svmRegressor, color="cyan")
plt.title("prices vs. predicted prices \n with SVM Regressor (Scikit-learn)")
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.subplot(5,2,6)
sns.distplot(y-y_pred_sklearn_svmRegressor)
plt.title("Histogram of Residuals \n with SVM Regressor")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.subplot(5,2,7)
plt.scatter(y, y_pred_sklearn_RandomForestRegressor, color="red")
plt.title("prices vs. predicted prices \n with Random Forest Regressor (Scikit-learn)")
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.subplot(5,2,8)
sns.distplot(y-y_pred_sklearn_RandomForestRegressor)
plt.title("Histogram of Residuals \n with Random Forest Regressor")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.subplot(5,2,9)
plt.scatter(y, y_pred_xgboost_xgbRegressor, color="purple")
plt.title("prices vs. predicted prices \n with XGBoost Regressor (XGBoost)")
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.subplot(5,2,10)
sns.distplot(y-y_pred_xgboost_xgbRegressor)
plt.title("Histogram of Residuals \n with XGBoost Regressor")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig('MachineLearning_BostonHousingPrices_Residuals.png')

plt.show() #show all figures
