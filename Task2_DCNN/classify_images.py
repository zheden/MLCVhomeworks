import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import scipy.stats as stats
import pickle

def printClassificationResults( probabilityDistr, classesNames, strInput):
	classNum = probabilityDistr.argmax()
	highestProbability = probabilityDistr[classNum]
	print strInput, 'Predicted class:', classesNames[classNum][10:-1]
	print 'Probability ', highestProbability
	print 'Entropy ', stats.entropy(probabilityDistr)
	listHighClasses = (idx for idx, value in enumerate(probabilityDistr) if value > 0.5 * highestProbability)
	print 'Other probable classes:'
	for anotherClass in listHighClasses:
		if (anotherClass <> classNum):
    			print(classesNames[anotherClass][10:-1])

def cropImage(im, newW, newH, shiftRight, shiftDown): # shift from center
	h = im.shape[0]
	w = im.shape[1]
	newim = im[shiftDown + (h - newH)/2 : shiftDown + h-(h - newH)/2, (w - newW)/2 + shiftRight : w-(w - newW)/2 + shiftRight, :]
	print newim.shape
	return newim
	
def preprocessImage(im, imageNum):
	if imageNum == 1:
		return cropImage(image, 120, 120, -10, 50) # squirrel
	elif imageNum == 2:
		return cropImage(image, 320, 320, 0, 0) # beans
	elif imageNum == 6:
		return cropImage(image, 250, 250, -30, 30) # mushr
	return image
	
caffe_root = '../'  # this file is expected to be in {caffe_root}/code
sys.path.insert(0, caffe_root + 'python')
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
#IMAGE_FILE = '../examples/images/cat.jpg'
LIST_NAMES = '../data/ilsvrc12/synset_words.txt'
f=open(LIST_NAMES)
classesNames=f.readlines()
f.close()

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

		
for imageNum in range(1, 9):
	print ' ' 
	print '==== Image ', imageNum, '====' 
	IMAGE_FILE = '../test_images/IMG_%d.jpg' % imageNum
	image = caffe.io.load_image(IMAGE_FILE)
	fig = plt.figure()
	sub1 = fig.add_subplot(231)
	sub2 = fig.add_subplot(232)
	sub3 = fig.add_subplot(233)
	sub4 = fig.add_subplot(234)
	sub5 = fig.add_subplot(235)
	sub6 = fig.add_subplot(236)
	
	sub1.imshow(image)

	# do not subtract mean because it will be subtacted inside 'predict' by default
	#image = image - image.mean(1).mean(0)
	prediction = net.predict([image])[0]
	sub2.plot(prediction)#
	sub2.set_title('oversample')
	#printClassificationResults(prediction, classesNames, '  --Oversample. ')

	imagePreProc = preprocessImage(image, imageNum)
	sub4.imshow(imagePreProc)
	prediction = net.predict([imagePreProc])[0]
	sub5.set_title('oversample')
	sub5.plot(prediction)
	printClassificationResults(prediction, classesNames, '  --Oversample. Preprocessed. ')

	prediction = net.predict([image], oversample=False)[0]
	sub3.plot(prediction)
	sub3.set_title('no oversample')
	#printClassificationResults(prediction, classesNames, '  --No oversample. ')
	
	prediction = net.predict([imagePreProc], oversample=False)[0]
	sub6.plot(prediction)
	sub6.set_title('no oversample')
	#printClassificationResults(prediction, classesNames, ' --No oversample. Preproc')
plt.show(block=False)

raw_input("Press Enter to close all...")
plt.close('all')
