import sys
from prep_terrain_data import makeTerrainData
import sklearn.model_selection as ms
import sklearn.metrics as metrics
from class_vis import prettyPicture, output_image


import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn import tree
features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
print(clf.predict([features_test[0]]))
print(labels_test[0])

y_pred = clf.predict(features_test)
print("Accuracy:",metrics.accuracy_score(labels_test, y_pred))

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
