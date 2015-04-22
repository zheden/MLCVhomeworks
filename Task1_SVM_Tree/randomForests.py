# -*- coding: utf-8 -*-
"""
@author: Dobryden Ievgeniia
"""

from sklearn import ensemble
import matplotlib.pyplot as plt
from loadMNIST import *
from sklearn import cross_validation

#==============================================================================
# load from db
# images, labels = load_mnist("training", "..\db\.")
# imagesTest, labelsTest = load_mnist("testing", "..\db\.")
#==============================================================================

# training data
numToTrain = 60000
images1 = images[0:numToTrain, :]
labels1 = labels[0:numToTrain]

# testing data
numToTest = imagesTest.shape[0]
imagesTest1 = imagesTest[0:numToTest, :]
labelsTest1 = labelsTest[0:numToTest]

# params
forest = ensemble.RandomForestClassifier(n_estimators=10)
forest.set_params(criterion = 'entropy') # entropy #gini
forest.set_params(max_features = 30, max_depth = 10)

# training
forest.fit(images1, labels1)

# prediction
resultsTest1 = forest.predict(imagesTest1)

# results
score = forest.score(imagesTest1, labelsTest1 )
print "Total score = " , score

# pixels importances
plt.figure(1)
plt.gray()
plt.imshow(images[2, :].reshape(28, 28), interpolation='none')
plt.figure(2)
plt.imshow(images[3, :].reshape(28, 28), interpolation='none')

importantPixels = forest.feature_importances_ * (forest.feature_importances_ > 0.007);
plt.figure(3)
plt.gray()
plt.imshow(importantPixels.reshape(28, 28), interpolation='none')

#cross-validation
scores = cross_validation.cross_val_score(forest, images1, labels1, cv=5)
print "Cross-validatio score = " , np.mean(scores)
