# -*- coding: utf-8 -*-
"""
@author: Dobryden Ievgeniia
"""

from sklearn import svm
from loadMNIST import *
import matplotlib.pyplot as plt

#load from db
images, labels = load_mnist("training", "..\db\.")
imagesTest, labelsTest = load_mnist("testing", "..\db\.")

# training data
numToTrain = 1000
images1 = images[0:numToTrain, :]
labels1 = labels[0:numToTrain]

# testing data
numToTest = 1000#imagesTest.shape[0] #100
imagesTest1 = imagesTest[0:numToTest, :]
labelsTest1 = labelsTest[0:numToTest]

# normalize images
images1 = (images1 - 255/2 * ones(images1.shape)) / 255*2
imagesTest1 = (imagesTest1 - 255/2 * ones(imagesTest1.shape)) / 255*2

# params
weights = {0:1, 1:0.9, 2:5, 3:15, 4:3, 5:4, 6:4, 7:5, 8:14, 9:3}
clf = svm.SVC(class_weight= 'auto', C=100, gamma = 0.001)
clf.set_params(kernel='linear')

# training
clf.fit(images1, labels1)

# prediction
resultsTest1 = clf.predict(imagesTest1)

# results
score = clf.score(imagesTest1, labelsTest1 )
print "Total score = " , score

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








