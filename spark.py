import os
import h5py
import numpy as np
import pyspark as sp
import logging
import os
import random
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from datetime import datetime
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler, Normalizer, ChiSqSelector
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import PCA
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import SVMModel, SVMWithSGD

# disable the warnings
logging.getLogger('tensorflow').disabled = True

PCA_K = 80

# data needed has been created and saved in raw_data.h5 and gen_data.h5

# read the raw data from raw_data.h5
# the raw data is from teacher and the data has been processed by process_data.py
# the raw data is of 138 days
raw_h5f = h5py.File('raw_data.h5', 'r')
datas = raw_h5f['datas'][:] # an array which has 138 elements
# print(datas.shape) 
# the shape of datas is (138, 32, 32, 144) 
# 32*32 refers to the 32*32 grids
# 144 = 48*3, 48 means that every day has 48 slots
# 3 means that for every slot, inflow, outflow and inflow-outflow are taken into account
others = raw_h5f['others'][:] # an array which has 138 elements
# print(others.shape) 
# the shape of others is (138, 19)
# 19 refers to other information such as weather and windspeed 
labels = raw_h5f['labels'][:] # an array which has 138 elements
# print(labels.shape) 
# the shape of labels is (138, 2)
# [1, 0] means weekend and [0, 1] weekday
raw_h5f.close()
print('read raw data ok')

# read the generated data from gen_data.h5 
# the data was generated in the way of Data Augmentation
# Data Augmentation helps to avoid overfitting
gen_h5f = h5py.File('gen_data.h5', 'r')
gen_d = gen_h5f['datas'][:]
# print(gen_d.shape)
# the shape of gen_d is (828, 32, 32, 144)
gen_o = gen_h5f['others'][:]
# print(gen_o.shape)
# the shape of gen_d is (828, 19)
gen_l = gen_h5f['labels'][:]
# print(gen_l.shape)
# the shape of gen_l is (828, 2)
gen_h5f.close()
print('read gen data ok')

datas_ = len(datas)*[32*32*48*2*[0.0]]
for i in range(len(datas)):
	count = 0
	for j in range(32):
		for k in range(32):
			for m in range(48):
				datas_[i][count] = datas[i][j][k][3*m]
				count += 1
				datas_[i][count] = datas[i][j][k][3*m+1]
				count += 1
datas = np.array(datas_)

datas_ = len(datas)*[(32*32*48*2+19+1)*[0.0]]
for i in range(len(datas)):
	datas_[i] = np.concatenate((datas[i], others[i], np.array([labels[i][0]])))
datas = np.array(datas_)
print('reshape raw data ok')

gen_d_ = len(gen_d)*[32*32*48*2*[0.0]]
for i in range(len(gen_d)):
	count = 0
	for j in range(32):
		for k in range(32):
			for m in range(48):
				gen_d_[i][count] = gen_d[i][j][k][3*m]
				count += 1
				gen_d_[i][count] = gen_d[i][j][k][3*m+1]
				count += 1
gen_d = np.array(gen_d_)

gen_d_ = len(gen_d)*[(32*32*48*2+19+1)*[0.0]]
for i in range(len(gen_d)):
	gen_d_[i] = np.concatenate((gen_d[i], gen_o[i], np.array([gen_l[i][0]])))
gen_d = np.array(gen_d_)
print('reshape gen data ok')


conf = SparkConf()
conf.set('spark.executor.memory', '8g')
conf.set("spark.driver.memory", '8g')
sc = SparkContext(conf = conf)
rdd = sc.parallelize(datas)
rdd_ = sc.parallelize(gen_d)
# rdd_ = sc.parallelize(raw_data_)
# print('共有' + str(rdd.count()) + ' 项数据')

feature_rdd = rdd.map(lambda line: line[:-1])
scaler = StandardScaler(withStd = True, withMean = True).fit(feature_rdd) # 进行数值的标准化
feature_rdd = scaler.transform(feature_rdd) # 特征 rdd
label_rdd = rdd.map(lambda line: line[-1:]) # 标签 rdd

feature_rdd_ = rdd_.map(lambda line: line[:-1])
scaler_ = StandardScaler(withStd = True, withMean = True).fit(feature_rdd_) # 进行数值的标准化
feature_rdd_ = scaler.transform(feature_rdd_) # 特征 rdd
label_rdd_ = rdd_.map(lambda line: line[-1:]) # 标签 rdd

pca_model = PCA(PCA_K).fit(feature_rdd)

pca_feature_rdd = pca_model.transform(feature_rdd)
train_data = pca_feature_rdd.zip(label_rdd.map(lambda line: line[0])).map(lambda line: LabeledPoint(line[1], line[0]))

pca_feature_rdd_ = pca_model.transform(feature_rdd_)
test_data = pca_feature_rdd_.zip(label_rdd_.map(lambda line: line[0])).map(lambda line: LabeledPoint(line[1], line[0]))

def getAUC(train_data, test_data, model_type, model_iterations):
	model = None
	predict = None
	if model_type == 1:
		model = LinearRegressionWithSGD.train(train_data, iterations = model_iterations)
		predict = model.predict(test_data.map(lambda p: p.features)).map(lambda p: float(p)).map(lambda p: 1.0 if p > 0 else 0.0)
	elif model_type == 2:
		model = LogisticRegressionWithSGD.train(train_data, iterations = model_iterations)
		predict = model.predict(test_data.map(lambda p: p.features)).map(lambda p: float(p))
	elif model_type == 3:
		model = LogisticRegressionWithLBFGS.train(train_data, iterations = model_iterations)
		predict = model.predict(test_data.map(lambda p: p.features)).map(lambda p: float(p))
	elif model_type == 4:
		model = SVMWithSGD.train(train_data, iterations = model_iterations)
		predict = model.predict(test_data.map(lambda p: p.features)).map(lambda p: float(p))
	else:
		model = RandomForest.trainClassifier(train_data, numClasses = 2, numTrees = model_iterations, categoricalFeaturesInfo = {}, maxDepth=5)
		predict = model.predict(test_data.map(lambda p: p.features)).map(lambda p: float(p))

	predict_real = predict.zip(test_data.map(lambda p: p.label)) 
	metrics_bi = BinaryClassificationMetrics(predict_real)
	metrics_mul = MulticlassMetrics(predict_real)

	AUC = metrics_bi.areaUnderROC
	precision = metrics_mul.precision()
	recall = metrics_mul.recall()
	f1Score = metrics_mul.fMeasure()

	print('dataset size: ', dataset.count())
	print('test data size: ', predict_real.count())
	print('confusion matrix:\n', metrics_mul.confusionMatrix()) # 打印混淆矩阵

	return AUC, recall, precision, f1Score

AUC, recall, precision, f1Score = getAUC(train_data, test_data, 2, 1000)
print('AUC: ', AUC)
print('recall: ', recall)
print('precision: ', precision)
print('f1Score: ', f1Score)
