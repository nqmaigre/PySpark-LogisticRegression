import os
import h5py
import numpy as np
import pyspark as sp
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
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import SVMModel, SVMWithSGD


def isWeekend(date):
	day = datetime.strptime(date, '%Y%m%d').weekday()
	# Monday - Sunday
	# 0      - 6
	if day <= 4: 
		return False
	else:
		return True

f1 = h5py.File("BJ16_M32x32_T30_InOut.h5", "r")
f2 = h5py.File("BJ_Meteorology.h5", "r")

# Transform a datasets in a .h5 file into an numpy array.

timeslots = f1["date"][:]
# A array of shape 7220x1 where date[i] is the time of i-th timeslot.

data = f1["data"][:]
# An array of shape 7220x2x32x32 where the first dimension represents the index of timeslots. data[i][0] is a (32, 32) inflow matrix and data[i][1] is a (32, 32) outflow matrix.

temperature = f2["Temperature"][:]
# An array of shape 7220x1 where temperature[i] is the temperature at i-th timeslot.

weather = f2["Weather"][:]
# An array of shape 7220x17 where weather[i] is a one-hot vector which means:
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Sunny
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Cloudy
# [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Overcast
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Rainy
# [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] Sprinkle
# [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ModerateRain
# [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] HeavyRain
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] Rainstorm
# [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] Thunderstorm
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] FreezingRain
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] Snowy
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] LightSnow
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] ModerateSnow
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] HeavySnow
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] Foggy
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] Sandstorm
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] Dusty

windspeed = f2["WindSpeed"][:]
# An array of shape 7220x1 where windspeed[i] is the wind speed at i-th timeslot.

# print('\n')
# print(date[7219])
# print(temperature[7219])
# print(weather[7219])
# print(data[0][0])
# print(windspeed[7219])

raw_data = []
# raw_data_ = []

for i in range(7220):
	timeslot = timeslots[i].decode('utf8')
	date = timeslot[:8]
	is_weekend = isWeekend(date)
	slot = int(timeslot[8:])
	item = []

	# inflows of all grids is equal to outflows of all grids
	# inflows = 0
	# outflows = 0
	# flows = 0
	for j in range(32):
		for k in range(32):
			item += [data[i][0][j][k].item(), data[i][1][j][k].item()]
			# inflows += data[i][0][j][k]
			# outflows += data[i][1][j][k]
			# flows += data[i][0][j][k]

	
	# raw_data.append([slot, inflows, outflows, temperature[i], windspeed[i], weather[i], is_weekend])
	# raw_data.append([slot, flows.item(), temperature[i].item(), windspeed[i].item(), weather[i].tolist(), 1 if is_weekend else 0])
	# item = [flows.item()/10000, temperature[i].item(), windspeed[i].item()]
	item += [temperature[i].item(), windspeed[i].item()]
	item += map(lambda x: int(x), weather[i].tolist())
	slot_one_hot = [0] * 48
	slot_one_hot[slot-1] = 1
	item += slot_one_hot
	item += [1 if is_weekend else 0]
	raw_data.append(item)

conf = SparkConf()
conf.set('spark.executor.memory', '8g')
conf.set("spark.driver.memory", '8g')
sc = SparkContext(conf = conf)
rdd = sc.parallelize(raw_data)
# print('共有' + str(rdd.count()) + ' 项数据')

feature_rdd = rdd.map(lambda line: line[:-1])
label_rdd = rdd.map(lambda line: line[-1:])

'''
sqlContext = SQLContext(sc)

scaler = StandardScaler(withStd = True, withMean = True).fit(feature_rdd)
scaler_feature_dataset = scaler.transform(feature_rdd) #.zip(rdd_).map(lambda line: line[0].tolist() + line[1]).map(lambda x: (Vectors.dense([x[i] for i in range(0, 68)])))

pca_model = PCA(20).fit(scaler_feature_dataset)
pca_scaler_feature_dataset = pca_model.transform(scaler_feature_dataset)

dataset = pca_scaler_feature_dataset.zip(label_rdd.map(lambda line: line[0])).map(lambda line: LabeledPoint(line[1], line[0]))

# dataset = sqlContext.createDataFrame(dataset)
# dataset.show(5, False)
# os._exit(0)

(train_data, test_data) = dataset.randomSplit([0.8, 0.2])

# model = LogisticRegressionWithSGD.train(train_data, iterations = 200)
model = SVMWithSGD.train(train_data, iterations = 200)
# model = RandomForest.trainClassifier(train_data, numClasses = 2, numTrees = 200, categoricalFeaturesInfo = {}, maxDepth=5) #, featureSubsetStrategy="auto",impurity='gini', maxDepth=4, maxBins=32)
predict = model.predict(test_data.map(lambda p: p.features)).map(lambda p: float(p)) #.map(lambda p: 1.0 if p > 0 else 0.0)
predict_real = predict.zip(test_data.map(lambda p: p.label))
metrics = BinaryClassificationMetrics(predict_real)
AUC = metrics.areaUnderROC

dataset = sqlContext.createDataFrame(predict_real)
dataset.show(30, False)

## 打印AUC
print('AUC = ' + str(AUC))
'''

def getAUC(feature_rdd, label_rdd, do_pca, pca_k, model_type, model_iterations):
	scaler = StandardScaler(withStd = True, withMean = True).fit(feature_rdd)
	scaler_feature_dataset = scaler.transform(feature_rdd) 

	scaler_feature_dataset_ = scaler_feature_dataset
	if do_pca:
		pca_model = PCA(pca_k).fit(scaler_feature_dataset)
		scaler_feature_dataset_ = pca_model.transform(scaler_feature_dataset)

	dataset = scaler_feature_dataset_.zip(label_rdd.map(lambda line: line[0])).map(lambda line: LabeledPoint(line[1], line[0]))
	(train_data, test_data) = dataset.randomSplit([0.8, 0.2])

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
	metrics = BinaryClassificationMetrics(predict_real)
	AUC = metrics.areaUnderROC

	return AUC

# AUC = getAUC(feature_rdd, label_rdd, True, 500, 1, 100)
# print(AUC)
# os._exit(0)

f3 = open('result.txt', 'w')
result = []
model_type = [2, 4, 5] #[1, 2, 3, 4, 5]
iterations = [100, 200, 500, 1000] #[50, 100, 200, 300, 500, 1000]
pca_k = [5, 10, 20, 50, 100, 500] #[5, 10, 20, 30, 50, 80, 100, 200, 500]
best = [None, None, None, 0]
for type in model_type:
	for model_iter in iterations:
		for k in pca_k:
			AUC = getAUC(feature_rdd, label_rdd, True, k, type, model_iter)
			temp = [type, model_iter, k, AUC]

			print(temp)
			print(temp, file = f3)
			result.append(result)

			if(temp[3] > best[3]):
				best = temp.copy()

print('\n')
print('best: ', best)
print('\n\n', file = f3)
print(result, file = f3)
print('\n', file = f3)
print(best, file = f3)

f3.close()

f1.close()
f2.close()

