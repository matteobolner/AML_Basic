#!/usr/bin/env python
# coding: utf-8

# **LIBRARIES**
# 
# ---
# 
# 
# 
# 
# 

# In[18]:


import scipy as sp
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import datetime as dt
import joblib
from numpy import set_printoptions
from scipy.ndimage import interpolation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC


#   
# 
# **LOADING THE DATASET**
# 
# ---
# 
# 
# 

# In[19]:


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


# In[20]:

# **DATA PREPROCESSING**
# 
# ---
# 
# According to most literature available, handwriting classification models based on Support Vector Machines (SVM) greatly benefit from deskewing, that is the straightening of the numbers in the images.    
# Therefore, all the images in the dataset were processed with deskewing, using the method available at https://fsix.github.io/mnist/Deskewing.html
# 
# 
# 
# 
# 
# 
# 

# In[21]:


def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)


# The results of deskewing can be appreciated in the images below

# In[22]:


# In[23]:


def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28,28)).flatten())
    return np.array(currents)

X_deskewed = deskewAll(X)


# Each feature is then transformed by rescaling to a value between 0 and 1

# In[24]:


scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X_deskewed)


# The dataset is split into training and testing sets, with a test size 15% of the total

# In[53]:


x_train, x_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.3, random_state=42)
data = {"train": {"X": x_train, "Y": y_train}, "test": {"X": x_test, "Y": y_test},}


# In[26]:


samplesize = len(data["train"]["X"])


# In[27]:


#poly-9 degrees are used with virtual svms (virtualized data)
#mnist_classifier = SVC(probability=False, kernel="poly", degree=9, C =2, gamma=0.01)


# **PARAMETER TUNING**
# 
# 
# ---
# 
# The main parameters considered in the tuning of the model are:
# 
# 
# *   **C** -> Regularization parameter
# *   **kernel** -> kernel type used in the algorithm
# *   **gamma** -> kernel coefficient
# 
# The only kernel considered for now is the RBF (or Gaussian) kernel; the C and gamma parameters will be selected with the grid search method.

# In[28]:


gamma_range = [0.0001, 0.01, 1.0, 10]
C_range = [0.1, 1, 10, 50, 100]
parameters = {'kernel':['poly'], 'C':C_range, 'gamma': gamma_range, 'degree' : [4,5]}


# In[29]:


#gamma_range


# In[30]:


#C_range


# In[61]:


svm_clf = svm.SVC()
grid_clf = GridSearchCV(estimator=svm_clf,param_grid=parameters, cv = 3, n_jobs=7, verbose=10)


# In[62]:


start_time = dt.datetime.now()
print('Start param searching at {}'.format(str(start_time)))


# In[63]:


grid_clf.fit(data["train"]["X"][:samplesize], data["train"]["Y"][:samplesize])

# In[34]:


elapsed_time= dt.datetime.now() - start_time
print('Elapsed time, param searching {}'.format(str(elapsed_time)))
joblib.dump(grid_clf, 'grid_clf_poly.pkl')

# Best parameters:
# 10000 samples: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}

# In[35]:


#print(grid_clf.cv_results_.keys())

classifier = grid_clf.best_estimator_
params = grid_clf.best_params_
scores = grid_clf.cv_results_['mean_test_score']


# In[36]:


# In[37]:


print("\nBest performing classifier:\n" + str(classifier))
print("\nBest parameters:\n" + str(params))
print("\nAll test scores:\n" + str(scores))


# In[38]:


#grid_clf.cv_results_


# In[39]:


#predicted = grid_clf.predict(data["test"]["X"])
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(data["test"]["Y"], predicted))
#print("Accuracy: %0.4f" % metrics.accuracy_score(data["test"]["Y"], predicted))


# In[40]:


predicted = grid_clf.predict(data["test"]["X"])
df_cm = pd.DataFrame(metrics.confusion_matrix(data["test"]["Y"], predicted))
print("Accuracy: %0.4f" % metrics.accuracy_score(data["test"]["Y"], predicted))


# In[41]:


#print(sklearn.metrics.matthews_corrcoef(y_pred, data["test"]["Y"]))


# In[42]:


params_df = pd.concat([pd.DataFrame(grid_clf.cv_results_["params"]),pd.DataFrame(grid_clf.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)


# In[43]:


params_df.to_csv('params_df_poly.csv')

# In[44]:


print("\n\nBest parameters set found on development set:")
print()
print(grid_clf.best_params_)
print()


# In[45]:


means = grid_clf.cv_results_['mean_test_score']
stds = grid_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_clf.cv_results_['params']):
  print("%0.3f (+/-%0.03f) with %r" % (mean, std * 2, params))
