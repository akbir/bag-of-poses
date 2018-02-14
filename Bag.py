import cv2
import pickle
import numpy as np
from glob import glob
import argparse
from helpers import *
from matplotlib import pyplot as plt
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')


	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


class BOV:
	def __init__(self, no_clusters):
		self.no_clusters = no_clusters
		self.train_path = None
		self.test_path = None
		self.checkpoint_path = None
		self.checkpoint = {}
		self.im_helper = ImageHelpers()
		self.bov_helper = BOVHelpers(no_clusters)
		self.file_helper = FileHelpers()
		self.images = None
		self.trainImageCount = 0
		self.train_labels = np.array([])
		self.name_dict = {}
		self.descriptor_list = []

	def trainModel(self):
		"""
		This method contains the entire module
		required for training the bag of visual words model

		Use of helper functions will be extensive.

		"""


		# read file. prepare file lists.
		self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
		if bool(self.checkpoint):
			self.trainImageCount = self.checkpoint['ImageCount']
		print(self.trainImageCount)

		# extract SIFT Features from each image
		label_count = 0
		if bool(self.name_dict):
			boo = list(map(int,self.name_dict.keys()))
			label_count = max(boo)
			print("Updated labelcount is", label_count)

		for word, imlist in self.images.items():
			self.name_dict[str(label_count)] = word
			print("Computing Features for ", word)
			im_count = 0
			for im in imlist:
				# cv2.imshow("im", im)
				# cv2.waitKey()
				print(im_count)
				self.train_labels = np.append(self.train_labels, label_count)
				des = self.im_helper.features(im,im_count)
				self.descriptor_list.append(des)
				im_count += 1

			#self.train_labels =np.append(train_labels, label_count)
			#train_labels = []
			im_count = 0

			print("Saving to file:", self.checkpoint_path)
			with open(self.checkpoint_path, 'wb')as fp:
				self.checkpoint['descriptor_list'] = self.descriptor_list
				self.checkpoint['name_dict'] = self.name_dict
				self.checkpoint['ImageCount'] = self.trainImageCount
				self.checkpoint['train_labels']= self.train_labels
				pickle.dump(self.checkpoint, fp)

			label_count += 1

		# perform clustering
		print("Performing Clustering")
		bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
		self.bov_helper.cluster()
		self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)
		print("Clustering done!")
		# show vocabulary trained
		self.bov_helper.plotHist()


		self.bov_helper.standardize()
		self.bov_helper.train(self.train_labels)


	def recognize(self,test_img,imcount, test_image_path=None):

		"""
		This method recognizes a single image
		It can be utilized individually as well.


		"""

		des = self.im_helper.features(test_img,imcount)

		# generate vocab for test image
		vocab = np.array( [ 0 for i in range(self.no_clusters)])
		# locate nearest clusters for each of
		# the visual word (feature) present in the image

		# test_ret =<> return of kmeans nearest clusters for N features
		test_ret = self.bov_helper.mbk_obj.predict(des)

		for each in test_ret:
			vocab[each] += 1

		# Scale the features

		#AkEdit
		vocab = vocab.reshape(1,-1)


		vocab = self.bov_helper.scale.transform(vocab)

		# predict the class of the image
		lb = self.bov_helper.clf.predict(vocab)
		print("Image belongs to class : ", self.name_dict[str(int(lb[0]))])
		return lb



	def testModel(self):

		self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)

		predictions = []
		y_test = []
		y_pred= []

		for word, imlist in self.testImages.items():
			print("processing " ,word)
			imcount=0
			for im in imlist:
				y_test.append(word)
				cl = self.recognize(im,imcount)
				y_pred.append(self.name_dict[str(int(cl[0]))])
				predictions.append({
					'image':im,
					'class':cl,
					'object_name':self.name_dict[str(int(cl[0]))]
					})
				imcount +=1
		return(y_test, y_pred)
		# print predictions



	def print_vars(self):
		pass


if __name__ == '__main__':

	# parse cmd args
	parser = argparse.ArgumentParser(
			description=" Bag of visual words example"
		)
	parser.add_argument('--train_path', action="store", dest="train_path", required=True)
	parser.add_argument('--test_path', action="store", dest="test_path", required=True)
	parser.add_argument('--checkpoint_path', action="store", dest="checkpoint_path", required=False)

	args =  vars(parser.parse_args())
	# print args

	class_names =[]

	bov = BOV(no_clusters=150)

	# set training paths
	bov.train_path = args['train_path']
	# set testing paths
	bov.test_path = args['test_path']
	bov.checkpoint_path=args['checkpoint_path']

	if bov.checkpoint_path is None:
		bov.checkpoint_path = 'checkpoint.pkl'
		print("Created checkpoint")
	else:
		print("Loading checkpoint")
		bov.checkpoint = pickle.load(open(bov.checkpoint_path, 'rb'))
		bov.descriptor_list = bov.checkpoint['descriptor_list']
		bov.name_dict =bov.checkpoint['name_dict']
		bov.train_labels = bov.checkpoint['train_labels']
		#print(bov.checkpoint)

	# train the model
	print("Training the model")
	bov.trainModel()
	# test model
	print("Testing the model")
	y_true, y_pred = bov.testModel()
	class_names = list(set(y_true))

	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y_true, y_pred)
	np.set_printoptions(precision=2)

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
						  title='Normalized confusion matrix')

	plt.savefig('confusion_matrix.png')

	print(accuracy_score(y_true,y_pred))
