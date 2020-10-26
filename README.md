# AML_Basic
Repository for the Applied Machine Learning project (Basic module only)


The goal of this project was to produce a handwriting classifier based on Support Vector Machines, using the MNIST dataset for training and testing (http://yann.lecun.com/exdb/mnist/).  
The model obtained uses a gaussian kernel function, and is able to correctly classify images in the testing set with an error rate of **1.4857** , which is on par with the value reported on the above website.  
The polynomial kernel seems to underperform slightly, with an error rate of **1.5714**, which is higher than the reported rate.

## REPO STRUCTURE
- AML.ipynb : google colab notebook implementing and describing the project
- AML.py : python script obtained from the notebook
- grid_clf.pkl; grid_clf_poly.pkl : pickled grid search results


## MNIST
" The MNIST database of handwritten digits, [...] has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. " [[1]](#1).  
The database was imported with the fetch_openml function of scikit-learn.  

## Support Vector Machines
Support  Vector  Machines  (SVM)  are  supervised  learning  models  with associated learning algorithms that analyze data used for classification and regression analysis  [[2]](#1).  
Compared  to  other  machine  learning techniques such as neural networks, SVMs have the advantage of being less prone to overfitting and being able to handle large feature spaces.  
The implementation of the kernel function allows the implicit mapping of the data to a high-dimensional space, which facilitates the separation of the data in classes; in this project, two kinds of kernel functions were tested: the gaussian (RBF) and the polynomial kernel function.

## Data preprocessing
According to most literature available and the official MNIST website, handwriting classification models based on Support Vector Machines (SVM) benefit from deskewing, that is the straightening of the numbers in the images.  
Therefore, all the images in the dataset were processed with deskewing, using the method available at https://fsix.github.io/mnist/Deskewing.html.

## Model training and parameter evaluation
Since this project uses a google colab notebook, time-consuming tasks such as the grid search for the model parameters were performed on a local machine.  
The grid search was performed along with 3-fold cross validation.


## References
<a id="1">[1]</a> 
http://yann.lecun.com/exdb/mnist/  
<a id="2">[2]</a> 
https://en.wikipedia.org/wiki/Support-vector_machine
