from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

import numpy as np


#loading the matlab file
import scipy.io
mat = scipy.io.loadmat('MATLAB/dbworld_bodies_stemmed.mat')
#print(mat)

#Separating labels and Data
mlables = mat['labels']  # variable in mat file
mlabels=np.asarray(mlables) #these are labels
print("Shape of labels' set: ",mlabels.shape)

mdata = mat['inputs']  # variable in mat file
mdata=np.asarray(mdata) #this is data
print("Shape of data set: ",mdata.shape)

#Splitting into test and train with test size 25% and fixing random state for consistency
X_train, X_test, y_train, y_test = train_test_split(mdata, mlabels, test_size=0.25, random_state=55)

print ('X_train dimensions: ', X_train.shape)
print ('y_train dimensions: ', y_train.shape)
print ('X_test dimensions: ', X_test.shape)
print ('y_test dimensions: ', y_test.shape)

#Mentioned below are the three models, use one and comment the other

neigh = KNeighborsClassifier(n_neighbors=3)
#model = neigh.fit(X_train,y_train.ravel())
#model = GaussianNB().fit(X_train, y_train.ravel())
model = NearestCentroid().fit(X_train,y_train.ravel())

y_train_pred = model.predict(X_train) #Training the model

#printing the training Ground truth and training predicted results
print("Training Data prediction: \n",y_train_pred)
print("Training Data ground truth: \n",y_train.ravel())

#creating confusion_matrix for training dataset
matrix = metrics.confusion_matrix(y_train, y_train_pred)
#print(matrix)
accuracy = (accuracy_score(y_train,y_train_pred))*100
print("Accuracy for training dataset: ", accuracy,"%")

#plotting confussion matrix
plt.matshow(matrix)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#Using the model of test data
y_test_pred = model.predict(X_test)
print("Testing Data Predicton: \n", y_test_pred)
print("Testing Data Ground Truth: \n", y_test.ravel())

matrix_test = confusion_matrix(y_test, y_test_pred)
#print(matrix_test)
accuracy_test = (accuracy_score(y_test, y_test_pred))*100

print("Accuracy for Testing Dataset: ", accuracy_test,"%")

plt.matshow(matrix_test)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

