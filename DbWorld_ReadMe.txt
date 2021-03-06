Implementation of dataset#3 (DBWorld_Emails)using Weka Tool and Python:
Weka Tool is Weka is a collection of machine learning algorithms for data mining tasks. The algorithms can either be applied directly to a dataset or called from your own Java code. Weka contains tools for data pre-processing, classification, regression, clustering, association rules, and visualization. It is also well-suited for developing new machine learning schemes.
	About the dataset:
DBWorld Email Dataset is a very small dataset created by Michele Filannino. It consists of 64 email instances. Each email is further divided into two datasets; one is consisting of the subjects of emails and other consisting of bodies of the emails. Both datasets are labelled with respect to the two classes: announcement of conference and everything else. These labels are in the binary form i.e. 1 for announcement of conference and 0 otherwise. The properties of the dataset are given below in table 1
 
Table 1 Data Characterstics

Features or attributes in the case of this dataset are the words present in the emails’ subject and body part. Thus the datasets contain term frequency in for each email.  Twenty top or most frequent words from both datasets are mentioned in table 2.
 
Table 2 Top 20 most frequent words in both subjects and bodies datasets.
The dataset consists of four files, two of them contains the original data and the remaining two contains the stemmed version of the original data.  First, training is done on the original dataset for that bodies dataset and subject dataset using Naïve Bayes and KNN. For Rocchio algorithm python has been used. 
Original Datasets:
•	First dbworld_bodies.arff (original dataset) file is loaded in weka.
•	Naïve Bayes and KNN classifiers have been applied and accuracy, precision and recall have been measured and confusion matrix has been drawn. 
1.	Result of Naïve bayes on dbworld_bodies.arff.
 
Figure 1 Naive bayes on original bodeis dataset
2.	Results of K-nearest neighbours using Euclidean distance to for classification of instances in dbworld_bodies.arff.
Choosing K=1 and K=3 the accuracy achieved is 98.4375% and 65.625% respectively.
  
3.	Results of Rocchio Algorithm (nearest centroid)
Code:
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

import numpy as np


# Config the matlotlib backend as plotting inline in IPython
#%matplotlib inline

import scipy.io
mat = scipy.io.loadmat('MATLAB/dbworld_bodies_stemmed.mat')
#print(mat)

mlables = mat['labels']  # variable in mat file
mlabels=np.asarray(mlables)
print(mlabels.shape)

mdata = mat['inputs']  # variable in mat file
mdata=np.asarray(mdata)
print(mdata.shape)

X_train, X_test, y_train, y_test = train_test_split(mdata, mlabels, test_size=0.25, random_state=55)

print ('X_train dimensions: ', X_train.shape)
print ('y_train dimensions: ', y_train.shape)
print ('X_test dimensions: ', X_test.shape)
print ('y_test dimensions: ', y_test.shape)


neigh = KNeighborsClassifier(n_neighbors=3)
#model = neigh.fit(X_train,y_train.ravel())
#model = GaussianNB().fit(X_train, y_train.ravel())
model = NearestCentroid().fit(X_train,y_train.ravel())

y_train_pred = model.predict(X_train)
print(y_train_pred)
print(y_train.ravel())

matrix = metrics.confusion_matrix(y_train, y_train_pred)
print(matrix)
accuracy = accuracy_score(y_train,y_train_pred)

print(accuracy)

plt.matshow(matrix)
plt.title('Confusion Matrix for Validation Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


y_test_pred = model.predict(X_test)
print(y_test_pred)
print(y_test.ravel())

matrix_test = confusion_matrix(y_test, y_test_pred)
print(matrix_test)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(accuracy_test)

plt.matshow(matrix_test)
plt.title('Confusion Matrix for Validation Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
Output:
		

Now using original subjects’ data following results have been achieved:
1.	Result of Naïve bayes on dbworld_subjects.arff.

 
2.	Results of K-nearest neighbours using Euclidean distance to for classification of instances in dbworld_subjects.arff.
With KNN for K=1 and K=3, an accuracy of 100% and 85.93% is obtained respesctively.
 
3.	Results of Rocchio Algorithm (nearest centroid) on dbworld_subjects.arff

Stemmed DataSets:
Now after doing stemming on bodies and subject dataset the accuracy of classifiers are as follows.

Classifiers	Stemmed Bodies	Stemmed subjects
Naïve Bayes	89.0625%	98.4375%
KNN with K=1	98.4375%	100%
KNN with K=3	70.31%	89.06%
Rocchio	
	
Table 3 Accuracy on stemmed dataset

Classifiers	Stemmed Bodies		Stemmed subjects
	     Precision	Recall		Precision	Recall
Naïve Bayes	0.909	0.891		0.985		0.984
KNN with K=1	0.985	9.984		1		1
Rocchio				
Table 4 Comparison of weighted Precision and recall
As can be seen in the above table the results have been improved. 
