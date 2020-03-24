import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="rbf") # kernel='linear', poly, 
clf.fit(features_train, labels_train)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
pred = clf.predict(features_test)



#### store your predictions in a list named pred
pred = clf.predict(features_test)





from sklearn.metrics import accuracy_score
"""
Accuracy classification score.

In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
"""
acc = accuracy_score(pred, labels_test)

print(acc)

