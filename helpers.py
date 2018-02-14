import  matplotlib
matplotlib.use('Agg')
import cv2
import torch
import numpy as np
from glob import glob
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from pose import get_pose

class ImageHelpers:
	def __init__(self):
		#self.sift_object = xfeatures2d.SIFT_create()
		self.stupid = 'lol'

	def features(self, image,im_count):
		gpu = im_count%4
		descriptors = get_pose(image,gpu)
		return(descriptors)

class BOVHelpers:
	def __init__(self, n_clusters = 20):
		self.n_clusters = n_clusters
		self.mbk_ret = None
		self.mbk_obj = MiniBatchKMeans(n_clusters=n_clusters)
		self.descriptor_vstack = None
		self.mega_histogram = None
		self.clf  = SVC()

	def cluster(self):
		"""cluster using KMeans algorithm,"""
		self.mbk_ret = self.mbk_obj.fit_predict(self.descriptor_vstack)

	def developVocabulary(self,n_images, descriptor_list, mbk_ret = None):
		"""Each cluster denotes a particular visual word
		Every image can be represeted as a combination of multiple
		visual words. The best method is to generate a sparse histogram
		that contains the frequency of occurence of each visual word
		Thus the vocabulary comprises of a set of histograms of encompassing
		all descriptions for all images
		"""
		self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
		old_count = 0
		for i in range(n_images):
			l = len(descriptor_list[i])
			for j in range(l):
				if mbk_ret is None:
					idx = self.mbk_ret[old_count+j]
				else:
					idx = mbk_ret[old_count+j]
				self.mega_histogram[i][idx] += 1
			old_count += l
		print("Vocabulary Histogram Generated")

	def standardize(self, std=None):
		"""

		standardize is required to normalize the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.
		"""
		if std is None:
			self.scale = StandardScaler().fit(self.mega_histogram)
			self.mega_histogram = self.scale.transform(self.mega_histogram)
		else:
			print("STD not none. External STD supplied")
			self.mega_histogram = std.transform(self.mega_histogram)

	def formatND(self, l):
		"""
		restructures list into vstack array of shape
		M samples x N features for sklearn
		"""
		vStack = np.array(l[0])
		for remaining in l[1:]:
			if remaining != []:
				vStack = np.vstack((vStack, remaining))
		where_are_NaNs = np.isnan(vStack)
		vStack[where_are_NaNs]=np.float(0)
		self.descriptor_vstack = vStack.copy()
		return vStack

	def train(self, train_labels):
		"""
		uses sklearn.svm.SVC classifier (SVM)
		"""
		print("Training SVM")
		print(self.clf)
		print("Train labels", train_labels)
		self.clf.fit(self.mega_histogram, train_labels)
		print("Training completed")

	def predict(self, iplist):
		predictions = self.clf.predict(iplist)
		return predictions

	def plotHist(self, vocabulary = None):
		print("Plotting histogram")
		if vocabulary is None:
			vocabulary = self.mega_histogram

		x_scalar = np.arange(self.n_clusters)
		y_scalar = np.array([abs(np.sum(vocabulary[:,h], dtype=np.int32)) for h in range(self.n_clusters)])

		print(y_scalar)

		plt.bar(x_scalar, y_scalar)
		plt.xlabel("Visual Word Index")
		plt.ylabel("Frequency")
		plt.title("Complete Vocabulary Generated")
		plt.xticks(x_scalar + 0.4, x_scalar)
		plt.savefig('histogram.png')

class FileHelpers:

	def __init__(self):
		pass

	def getFiles(self, path):
		"""
		- returns  a dictionary of all files
		having key => value as  objectname => image path
		- returns total number of files.
		"""
		imlist = {}
		count = 0
		for each in glob(path + "*"):
			word = each.split("/")[-1]
			print(" #### Reading image category ", word, " ##### ")
			imlist[word] = []
			for imagefile in glob(path+word+"/*"):
				print("Reading file ", imagefile)
				im = cv2.imread(imagefile)
				x,y = int(im.shape[0]/1000), int(im.shape[1]/1000)
				if x > 0 and  y>0:
					n = min(x,y)
					fxx, fyy = np.round(0.25*1/n,2),np.round(0.25*1/n,2)
					im2= cv2.resize(im,None, fx= fxx,fy= fyy,interpolation = cv2.INTER_AREA)
					imlist[word].append(im2)
				elif x > 0 or y >0:
					im2= cv2.resize(im,None, fx= 0.5,fy= 0.5,interpolation = cv2.INTER_AREA)
					imlist[word].append(im2)
				else:
					imlist[word].append(im)
				count +=1

		return [imlist, count]
