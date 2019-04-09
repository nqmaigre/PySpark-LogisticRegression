import h5py
import numpy as np
from datetime import datetime
import tensorflow as tf
import logging
import os
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
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import SVMModel, SVMWithSGD

logging.getLogger('tensorflow').disabled = True

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

# inflows = []
# outflows = []
datas = []
labels = []
others = []
for i in range(7220):
	timeslot = timeslots[i].decode('utf8')
	date = timeslot[:8]
	is_weekend = isWeekend(date)
	slot = int(timeslot[8:])

	inflow = data[i][0] # 32*32
	outflow = data[i][1] # 32*32
	flow = 32*[32*[2*[0]]]
	for j in range(32):
		for k in range(32):
			flow[j][k][0] = inflow[j][k]/100.0
			flow[j][k][1] = outflow[j][k]/100.0
			# print(flow[j][k][0])
			# print(flow[j][k][1])
			# flow.append([inflow[j][k], outflow[j][k]])
	# inflows.append(np.array(inflow))
	# outflows.append(np.array(outflow))
	# datas.append([inflow, outflow])
	datas.append(flow)
	labels.append([1, 0] if is_weekend else [0, 1])

	item = []
	item += [temperature[i].item(), windspeed[i].item()]
	item += map(lambda x: int(x), weather[i].tolist())
	slot_one_hot = [0] * 48
	slot_one_hot[slot-1] = 1
	item += slot_one_hot
	# item += [1 if is_weekend else 0]
	others.append(item)

# inflows = np.array(inflows)
# outflows = np.array(outflows)
datas = np.array(datas)
labels = np.array(labels)
# others = np.array(others)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                    	strides=[1, 2, 2, 1], padding='SAME')

# 打乱顺序
# num_example = datas.shape[0]
# arr = np.arange(num_example)
# np.random.shuffle(arr)
# datas = datas[arr]
# labels = labels[arr]

# 将所有数据分为训练集和验证集
# ratio = 0.8
# train_size = np.int(num_example*ratio)
# x_train = datas[:train_size]
# y_train = labels[:train_size]
# x_test = datas[train_size:]
# y_test = labels[train_size:]

datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 2])
labels_placeholder = tf.placeholder(tf.float32, [None, 2])
dropout_placeholdr = tf.placeholder(tf.float32)

num_conv1 = 16
num_conv2 = 32
num_fc = 128

W_conv1 = weight_variable([5, 5, 2, num_conv1])
b_conv1 = bias_variable([num_conv1])

h_conv1 = tf.nn.relu(conv2d(datas_placeholder, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, num_conv1, num_conv2])
b_conv2 = bias_variable([num_conv2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8*8*num_conv2, num_fc])
b_fc1 = bias_variable([num_fc])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*num_conv2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, dropout_placeholdr)

W_fc2 = weight_variable([num_fc, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(labels_placeholder*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(labels_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(10000):
	# batch = mnist.train.next_batch(50)
	rand_index = np.random.choice(datas.shape[0], size = 64)
	rand_x = datas[rand_index]
	rand_y = labels[rand_index]
	if i % 50 == 0:
		train_accuracy = accuracy.eval(feed_dict={
			datas_placeholder: rand_x, 
			labels_placeholder: rand_y, 
			dropout_placeholdr: 1.0
		})
		print("step %d, training accuracy %g"%(i, train_accuracy))
		# if train_accuracy >= 0.85:
		# 	print('!!')
		# 	os._exit(0)
	train_step.run(feed_dict={
		datas_placeholder: rand_x, 
		labels_placeholder: rand_y, 
		dropout_placeholdr: 0.5
	})

# rand_index = np.random.choice(datas.shape[0], size = 64)
# rand_x = datas[rand_index]
# rand_y = labels[rand_index]	
datas_temp = []
for g in datas:
	datas_temp.append(sess.run(h_fc1, feed_dict={datas_placeholder: np.array(g).reshape((1, 32, 32, 2))})[0])

print(len(datas_temp))
raw_data = []
for i in range(7220):
	raw_data.append(datas_temp[i].tolist() + others[i])

conf = SparkConf()
conf.set('spark.executor.memory', '8g')
conf.set("spark.driver.memory", '8g')
sc = SparkContext(conf = conf)
rdd = sc.parallelize(raw_data)


feature_rdd = rdd.map(lambda line: line[:-1])
scaler = StandardScaler(withStd = True, withMean = True).fit(feature_rdd) # 进行数值的标准化
feature_rdd = scaler.transform(feature_rdd) # 特征 rdd
label_rdd = rdd.map(lambda line: line[-1:]) # 标签 rdd

def getAUC(feature_rdd, label_rdd, do_pca, pca_k, model_type, model_iterations):
	# feature_rdd 特征值
	# label_rdd 标签
	# do_pca 是否进行主成分分析 True/False
	# pca_k 主成分分析后保留的特征数
	# model_type 使用的模型种类 1、线性回归 2、Logistic回归 3、Logistic回归 4、SVM 5、随即森林 （2、3都是逻辑回归）
	# model_iterations 迭代次数，对于随机森林而言是决策树的数量
	pca_feature_rdd = feature_rdd
	if do_pca:
		pca_model = PCA(pca_k).fit(feature_rdd)
		pca_feature_rdd = pca_model.transform(feature_rdd)

	dataset = pca_feature_rdd.zip(label_rdd.map(lambda line: line[0])).map(lambda line: LabeledPoint(line[1], line[0]))
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

	metrics_bi = BinaryClassificationMetrics(predict_real)
	metrics_mul = MulticlassMetrics(predict_real)

	AUC = metrics_bi.areaUnderROC
	# 一下三项好像不准确
	# precision = metrics_mul.precision()
	# recall = metrics_mul.recall()
	# f1Score = metrics_mul.fMeasure()

	print(predict_real.count()) # 打印 predict_real 的数据数量
	print(metrics_mul.confusionMatrix()) # 打印混淆矩阵

	return AUC #, recall, precision, f1Score

AUC = getAUC(feature_rdd, label_rdd, True, 100, 2, 1000)
print(AUC)

f1.close()
f2.close()