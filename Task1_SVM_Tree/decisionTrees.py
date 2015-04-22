# -*- coding: utf-8 -*-
"""
@author: Dobryden Ievgeniia
"""

from sklearn import tree
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
from loadMNIST import *
from sklearn import cross_validation

# load from db
images, labels = load_mnist("training", "..\db\.")
imagesTest, labelsTest = load_mnist("testing", "..\db\.")

# training images
numToTrain = 1000
images1 = images[0:numToTrain, :]
labels1 = labels[0:numToTrain]

# testing images
numToTest = imagesTest.shape[0]
imagesTest1 = imagesTest[0:numToTest, :]
labelsTest1 = labelsTest[0:numToTest]

# params
decTree = tree.DecisionTreeClassifier()
decTree.set_params(criterion = 'entropy') # entropy #gini
decTree.set_params(max_features = 100, max_depth = 12)

# training
decTree.fit(images1, labels1)

# prediction
resultsTest1 = decTree.predict(imagesTest1)
# results
score = decTree.score(imagesTest1, labelsTest1 )
print "Total score = " , score

# output built tree
dot_data = StringIO() 
with open("tree.dot", 'w') as dot_data:
    tree.export_graphviz(decTree, out_file=dot_data)

# classification reate for each number
resHeader = " "
resString = ""
for i in range(0, 10):
    resultBool = resultsTest1 == labelsTest1
    whereIsNumBool = (labelsTest1 == i)
    resNum = float(sum( whereIsNumBool * resultBool)) / sum(whereIsNumBool)
    resString += "{:.2f}  ".format(resNum)
    resHeader += str(i) + "     " 

print resHeader
print resString

# pixels importances
plt.figure(1)
plt.gray()
plt.imshow(images[2, :].reshape(28, 28), interpolation='none')
plt.figure(2)
plt.imshow(images[3, :].reshape(28, 28), interpolation='none')

importantPixels = decTree.feature_importances_ * (decTree.feature_importances_ > 0.001);
plt.figure(3)
plt.gray()
plt.imshow(importantPixels.reshape(28, 28), interpolation='none')

#cross-validation
scores = cross_validation.cross_val_score(decTree, images1, labels1, cv=5)
print "Cross-validatio score = " , np.mean(scores)






