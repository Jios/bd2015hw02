#!/usr/bin/env python


import sys
import os
import os.path
import zipfile


################################################################


# environment set up for pyspark
# 	Path for spark source folder
os.environ['SPARK_HOME'] = "/usr/local/Cellar/apache-spark/1.3.1"
# 	Append pyspark  to Python Path
sys.path.append("/usr/local/Cellar/apache-spark/1.3.1/libexec/python")
sys.path.append("/usr/local/Cellar/apache-spark/1.3.1/libexec/python/lib/py4j-0.8.2.1-src.zip")


################################################################


# import libraries from pyspark
try:
    from pyspark import SparkContext, SparkConf
    from pyspark.mllib.clustering import KMeans
    from pyspark.mllib.feature import StandardScaler
    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

from collections import OrderedDict
import numpy as np
from numpy import array
from math import sqrt

from pandas import DataFrame

from sklearn.preprocessing import * #LabelBinarizer
from sklearn.metrics import * 

# from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans as skKMeans

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


clusters = 0

################################################################

def load_data(filepath = "datasets/kddcup.data_10_percent"):
	""" load KDDCup99 datasets """

	if not os.path.exists(filepath):
		with zipfile.ZipFile(filepath + '.zip', "r") as z:
			z.extractall("datasets/")

	conf = SparkConf().setAppName("KDDCup99")
	# .set("spark.executor.memory", "4g")
	sc = SparkContext(conf=conf)
	scData = sc.textFile(filepath)
	return scData
	
def parse_scData(line):
    """ parse KDDCup99 datasets """
    line_split = line.split(",")
    clean_line_split = [line_split[0]]+line_split[4:-1]
    return (line_split[-1], array([float(x) for x in clean_line_split]))

# //////////////////////////////////////////////////////////// #

def gaussian_naive_bayse():
	""" Naive Bayse """
	print("[INFO] - NB - Gaussian Navie Bayse")
	model = OneVsRestClassifier(GaussianNB())
	return model

def linear_SVC():
	""" SVM """
	print("[INFO] - SVM - Linear SVC")
	model = OneVsRestClassifier(LinearSVC(random_state=0))
	return model

def k_nearest_neighbors():
	""" K-Nearest Neighbors """
	print("[INFO] - kNN - k-Neighbors Classifier")
	model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
	return model

def random_forest():
	""" random forest """
	print("[INFO] - RF - Random Forest Classifier")
	# model = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
	# return model
	return RandomForestClassifier(n_estimators=10)

def k_means():
	""" KMeans """
	print("[INFO] - KMeans - KMeans Classifier")
	model = skKMeans(n_clusters=clusters)
	return model

# //////////////////////////////////////////////////////////// #

def scores(groundtrues, predictions):
    """ scores: recall, precision, F1 measure and accuracy """
    recall    = recall_score(groundtrues, predictions)
    precision = precision_score(groundtrues, predictions)
    f1        = f1_score(groundtrues, predictions)
    accuracy  = accuracy_score(groundtrues, predictions)
    s = (recall, precision, f1, accuracy)
    print(" recall = %f\n precision = %f\n F1 measure = %f\n accuracy = %f" % s)
    return s

# //////////////////////////////////////////////////////////// #

def get_scLabels(scData):
	labels = scData.map(lambda line: line.strip().split(",")[-1])
	return labels

def print_scLabel_counts(scData):
	""" """
	labels = get_scLabels(scData)
	label_counts = labels.countByValue()
	sorted_labels = OrderedDict(sorted(label_counts.items(), key=lambda t: t[1], reverse=True))
	for label, count in sorted_labels.items():
	    print label, count

def cluster_model(data, k=23):
    model = KMeans.train(data.sample(False, 0.5), 23, maxIterations=10, runs=5, initializationMode="random")
    return model

# //////////////////////////////////////////////////////////// #


def main():
	
	filepath = "datasets/kddcup.data_10_percent"
	scData = load_data(filepath)
	# sample_scData = scData.sample(False, 0.7)	

	scParsed_data = scData.map(parse_scData)
    # parsed_data_values = scParsed_data.values().cache()
	# sample_parsed_data = scData.take(100)

	# get sample data
	sample_scParsed_data = scParsed_data.sample(False, 0.5)
	sample_data = sample_scParsed_data.collect()
	# convert sc data to pandas.DataFrame
	data = DataFrame(sample_data, columns=['labelTags', 'arrData'])

	# labels
	lb = LabelEncoder() #LabelBinarizer()
	Y  = lb.fit_transform(data.labelTags)

	global clusters
	clusters = 23 #Y.shape[1]

	# data
	Xorg = data.arrData.tolist()

	# SVD
	svd  = TruncatedSVD(3)
	lsa  = make_pipeline(svd, Normalizer(copy=False))
	Xsvd = lsa.fit_transform(Xorg)

	options = { "RF"     : random_forest,
				"NB"     : gaussian_naive_bayse,
				"kNN"    : k_nearest_neighbors,
				"SVM"    : linear_SVC,
				"KM"     : k_means,
				"SVD+KM" : k_means, }

	m = int(len(Xsvd) * 0.5)
	while True:
		print("Classification model options:")
		print("\t\"RF\"  	 for Random Forest")
		print("\t\"NB\"  	 for Navie Bayse")
		# print("\t\"kNN\" for k-Nearest Neighbors")
		print("\t\"SVM\" 	 for Support Vector Machines")
		print("\t\"KM\"  	 for KMeans")
		print("\t\"SVD+KM\"  for SVD and KMeans")

		modelOpt = raw_input("Choose a classification method (enter \"q\" to quit): ")

		if modelOpt == "q":
			print("\nHappy clustering, goodbye!")
			break
		elif len(modelOpt) == 0 or not options.has_key(modelOpt):
			continue

		# switch data source for SVD+KM
		if modelOpt == "SVD+KM":
			X = Xsvd
		else:
			X = Xorg

		# classifier model
		model = options[modelOpt]()

		# train classifier
		clf  = model.fit(X[:m], Y[:m])

		# predict results
		predictions = clf.predict(X[m:])

		if modelOpt == "KM" or modelOpt == "SVD+KM":
			predictions = lb.fit_transform(predictions)
		# scores: recall, precision, F1 measure, and accuracy
		tagScore = scores(Y[m:], predictions)


	"""
	# using pyspark
	#
    model = cluster_model(scParsed_data.values(), 23)
    pred  = scParsed_data.map( lambda datum: str(model.predict(datum[1])) + "," + datum[0] ).sample(False, 0.5)

    # save prediction results to file(s)
    pred.saveAsTextFile("prediction_results")

    c = []
    l = []
    for d in pred.collect():
        dlist = d.split(',')
        c.append(int(dlist[0]))
        l.append(dlist[1])	
	labels_c = lb.fit_transform(c)
	labels_l = lb.fit_transform(l)

	recall, precision, f1, accuracy = scores(labels_l, labels_c)
	"""


################################################################


if __name__ == '__main__':
	main()

