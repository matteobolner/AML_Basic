# -*- coding: utf-8 -*-
"""AML

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/matteobolner/AML_Basic/blob/master/AML.ipynb

**LIBRARIES**

---
"""

import scipy as sp
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import datetime as dt
import joblib
from urllib.request import urlopen
from numpy import set_printoptions
from scipy.ndimage import interpolation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC

"""**LOADING THE DATASET**

---
"""

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

plt.imshow(X[7].reshape(28,28))

"""**DATA PREPROCESSING**

---

According to most literature available and the official MNIST website (http://yann.lecun.com/exdb/mnist/), most handwriting classification models based on Support Vector Machines (SVM) benefit from deskewing, that is the straightening of the numbers in the images.    
Therefore, all the images in the dataset were processed with deskewing, using the method available at https://fsix.github.io/mnist/Deskewing.html
"""

def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1])
    covariance_matrix = np.array([[m00,m01],[m01,m11]])
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)

"""The results of deskewing can be appreciated in the images below"""

plt.subplot(1, 2, 1)
plt.imshow(X[7].reshape(28,28))

newim = deskew(X[7].reshape(28,28))
plt.subplot(1, 2, 2)
plt.imshow(newim)

def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28,28)).flatten())
    return np.array(currents)

X_deskewed = deskewAll(X)

"""Each feature is then transformed by rescaling to a value between 0 and 1"""

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X_deskewed)

"""The dataset is split into training and testing sets, with a test size of 30%"""

x_train, x_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.3, random_state=42)
data = {"train": {"X": x_train, "Y": y_train}, "test": {"X": x_test, "Y": y_test},}

samplesize = len(data["train"]["X"])

"""**PARAMETER TUNING**


---

The main parameters considered in the tuning of the model are:


*   **C** -> Regularization parameter
*   **kernel** -> kernel type used in the algorithm
*   **gamma** -> kernel coefficient
"""

gamma_range = [0.0001, 0.01, 1.0, 5]
C_range = [0.001, 0.1, 10, 50, 100]
kernels = ['rbf','poly']
degrees = [4,5]
parameters_rbf = {'kernel': ['rbf'], 'C':C_range, 'gamma': gamma_range, 'degree': degrees}
parameters_poly = {'kernel': ['rbf'], 'C':C_range, 'gamma': gamma_range, 'degree': degrees}

svm_clf = svm.SVC()
grid_clf_rbf = GridSearchCV(estimator=svm_clf,param_grid=parameters_rbf, cv = 3, n_jobs=7, verbose=10)
grid_clf_poly= GridSearchCV(estimator=svm_clf,param_grid=parameters_poly, cv = 3, n_jobs=7, verbose=10)

"""Grid search was performed on another machine due to time reasons; the grid classifiers were pickled and imported"""

#grid_clf.fit(data["train"]["X"][:samplesize], data["train"]["Y"][:samplesize])

grid_clf_rbf = joblib.load(urlopen('https://github.com/matteobolner/AML_Basic/blob/master/grid_clf.pkl?raw=true'))
grid_clf_poly = joblib.load(urlopen('https://github.com/matteobolner/AML_Basic/blob/master/grid_clf_poly.pkl?raw=true'))

grid_clf_rbf
#grid_clf_poly

classifier_rbf = grid_clf_rbf.best_estimator_
params_rbf = grid_clf_rbf.best_params_
scores_rbf = grid_clf_rbf.cv_results_['mean_test_score']
classifier_poly = grid_clf_poly.best_estimator_
params_poly = grid_clf_poly.best_params_
scores_poly = grid_clf_poly.cv_results_['mean_test_score']

print("Best performing rbf classifier:\n" + str(classifier_rbf))
print("\nBest rbf parameters:\n" + str(params_rbf))
print("\nAll rbf test scores:\n" + str(scores_rbf))
print()
print("Best performing poly classifier:\n" + str(classifier_poly))
print("\nBest poly parameters:\n" + str(params_poly))
print("\nAll poly test scores:\n" + str(scores_poly))

rbf_params_df = pd.concat([pd.DataFrame(grid_clf_rbf.cv_results_["params"]),pd.DataFrame(grid_clf_rbf.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)

poly_params_df = pd.concat([pd.DataFrame(grid_clf_poly.cv_results_["params"]),pd.DataFrame(grid_clf_poly.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)

print("Best parameters set found on training sets:")
print()
print(grid_clf_rbf.best_params_)
print(grid_clf_poly.best_params_)

"""Once the best performing parameters have been identified, they are used in the training of two models, one per kernel"""

mnist_classifier_rbf = SVC(probability=False, kernel="rbf", C=50, gamma=0.01)
mnist_classifier_rbf.fit(data["train"]["X"][:samplesize], data["train"]["Y"][:samplesize])
mnist_classifier_poly = SVC(probability=False, kernel="poly", C=10, gamma=0.01, degree = 4)
mnist_classifier_poly.fit(data["train"]["X"][:samplesize], data["train"]["Y"][:samplesize])

"""The models obtained are used to classify the testing set"""

predicted_rbf = mnist_classifier_rbf.predict(data["test"]["X"])
conf_matrix_df_rbf = pd.DataFrame(metrics.confusion_matrix(data["test"]["Y"], predicted_rbf))
plt.figure(figsize = (10,7))
sns.heatmap(conf_matrix_df_rbf, annot=True, cmap="YlOrRd", fmt="d")

predicted_poly = mnist_classifier_poly.predict(data["test"]["X"])
conf_matrix_df_poly = pd.DataFrame(metrics.confusion_matrix(data["test"]["Y"], predicted_poly))
plt.figure(figsize = (10,7))
sns.heatmap(conf_matrix_df_poly, annot=True, cmap="YlOrRd", fmt="d")

error_rate_rbf = 100*(1 - metrics.accuracy_score(data["test"]["Y"], predicted_rbf))
error_rate_poly = 100*(1 - metrics.accuracy_score(data["test"]["Y"], predicted_poly))
accuracy_rbf = metrics.accuracy_score(data["test"]["Y"], predicted_rbf)
accuracy_poly = metrics.accuracy_score(data["test"]["Y"], predicted_poly)
mcc_rbf = metrics.matthews_corrcoef(data["test"]["Y"], predicted_rbf)
mcc_poly = metrics.matthews_corrcoef(data["test"]["Y"], predicted_poly)

print("Error rate with gaussian kernel: %0.4f"%error_rate_rbf)
print("Accuracy with gaussian kernel: " + str(metrics.accuracy_score(data["test"]["Y"], predicted_rbf)))
print("MCC with gaussian kernel: " + str(metrics.matthews_corrcoef(data["test"]["Y"], predicted_rbf)))
print()
print()
print("Error rate with polynomial kernel: %0.4f"%error_rate_poly)
print("Accuracy with gaussian kernel: " + str(metrics.accuracy_score(data["test"]["Y"], predicted_poly)))
print("MCC with polynomial kernel: " + str(metrics.matthews_corrcoef(data["test"]["Y"], predicted_poly)))

error_rates = [error_rate_rbf, error_rate_poly]
accuracies = [accuracy_rbf, accuracy_poly]
mccs = [mcc_rbf, mcc_poly]
pd.DataFrame(
    {'Error rate': error_rates,
     'Accuracy': accuracies,
     'MCC': mccs
    }, index = ['rbf', 'polynomial'])

"""In the end, the rbf kernel appears to perform slightly better"""