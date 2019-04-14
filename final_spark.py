import h5py
import numpy as np
from datetime import datetime
import tensorflow as tf
import logging
import os
import random
import pyspark as sp
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
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

# data needed has been created and saved in raw_data.h5 and gen_data.h5

# read the raw data from raw_data.h5
# the raw data is from teacher and the data has been processed by process_data.py
# the raw data is of 138 days
raw_h5f = h5py.File('raw_data.h5', 'r')
datas = raw_h5f['datas'][:] # an array which has 138 elements
# print(datas.shape) 
# the shape of datas is (138, 32, 32, 96) 
# 32*32 refers to the 32*32 grids
# 96 = 48*2, 48 means that every day has 48 slots
# 2 means that for every slot, inflow, outflow are taken into account
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
# the shape of gen_d is (828, 32, 32, 96)
gen_o = gen_h5f['others'][:]
# print(gen_o.shape)
# the shape of gen_d is (828, 19)
gen_l = gen_h5f['labels'][:]
# print(gen_l.shape)
# the shape of gen_l is (828, 2)
gen_h5f.close()
print('read gen data ok')

# use raw data as test data
x_test = datas
y_test = labels
o_test = others

# use generated data as training data
x_train = gen_d 
y_train = gen_l 
o_train = gen_o 

# following are some functions used in making the neural network
# use CNN to extract the information from datas/gen_d (namely the information from inflows and outflows)
# use normal NN to extract the information from others/gen_o
# then combine the information extracted by CNN and the information extracted from others/gen_o, and use following neural network to do classification work
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

def max_pool_4x4(x):
	return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                    	strides=[1, 4, 4, 1], padding='SAME')

datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 48*2]) # use datas/gen_d as input
other_placeholder = tf.placeholder(tf.float32, [None, 19]) # use others/gen_o as input
labels_placeholder = tf.placeholder(tf.float32, [None, 2]) # use labels/gen_l as input
dropout_placeholder = tf.placeholder(tf.float32)

# num_conv1 = 16 # 第二层卷积核数量
# num_conv2 = 32 # 第二层卷积核数量
# num_fc = 128 # inflow outflow 特征提取数量
# num_other = 19 # 其他信息的特征数量
# num_o_fc1 = 128 # 其他信息中提取出的特振数量
# num_fc2 = 128 # 所有特征提取出的特征数量

num_conv1 = 16 # the amount of convolution kernels of layer1
num_conv2 = 32 # the amount of convolution kernels of layer2
num_fc = 128 # the amount of features extraced by CNN
num_other = 19 # the dimensions of others/gen_o
num_o_fc1 = 128 # the amount of features extraced by normal NN
num_fc2 = 64 # the amount of features extraced by the total network

# convulutional layer1
W_conv1 = weight_variable([5, 5, 48*2, num_conv1])
b_conv1 = bias_variable([num_conv1])
h_conv1 = tf.nn.relu(conv2d(datas_placeholder, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# convulutional layer2
W_conv2 = weight_variable([5, 5, num_conv1, num_conv2])
b_conv2 = bias_variable([num_conv2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

b_fc1 = bias_variable([num_fc])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*num_conv2])
W_fc1 = weight_variable([8*8*num_conv2, num_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, dropout_placeholder) # use dropout to avoid overfitting

W_o_fc1 = weight_variable([num_other, num_o_fc1])
b_o_fc1 = bias_variable([num_o_fc1])
h_o_fc1 = tf.nn.relu(tf.matmul(other_placeholder, W_o_fc1) + b_o_fc1)
h_o_fc1_drop = tf.nn.dropout(h_o_fc1, dropout_placeholder) # use dropout to avoid overfitting

# concatenate the features extracted from CNN and normal NN
h_concat = tf.concat([h_fc1_drop, h_o_fc1_drop], axis=1) #tf.reshape(tf.concat(1, [h_fc1, other_placeholder]), [-1, num_fc+19])

W_fc2 = weight_variable([num_fc+num_o_fc1, num_fc2])
b_fc2 = bias_variable([num_fc2])
h_fc2 = tf.nn.relu(tf.matmul(h_concat, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, dropout_placeholder) # use dropout to avoid overfitting

W_fc3 = weight_variable([num_fc2, 2])
b_fc3 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# add L2 regularization to avoid overfitting
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_conv1)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_conv2)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_fc1)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_o_fc1)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_fc2)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_fc3)
regularizer = tf.contrib.layers.l2_regularizer(scale=10.0/x_test.shape[0])
reg_term = tf.contrib.layers.apply_regularization(regularizer)

cross_entropy = -tf.reduce_sum(labels_placeholder*tf.log(y_conv))
loss = cross_entropy + reg_term # loss function

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(labels_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
# the model has been trained and saved, so now can be restored
ckpt = tf.train.get_checkpoint_state('./checkpoint/')
saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
print(ckpt)

# because the model has been trained and resotred, so it doesn't have to be trained now

'''
# train the model
for i in range(10000):
	rand_index = np.random.choice(x_train.shape[0], size = 32)
	rand_x = x_train[rand_index]
	rand_y = y_train[rand_index]
	rand_o = o_train[rand_index]
	if i % 100 == 0:
		train_feed_dict = {
			datas_placeholder: rand_x, 
			labels_placeholder: rand_y, 
			other_placeholder: rand_o,
			dropout_placeholder: 1.0
		}
		test_feed_dict = {
			datas_placeholder: x_test, 
			labels_placeholder: y_test, 
			other_placeholder: o_test,
			dropout_placeholder: 1.0
		}
		test_accuracy = accuracy.eval(test_feed_dict)
		train_accuracy = accuracy.eval(train_feed_dict)
		train_loss = cross_entropy.eval(train_feed_dict)
		print("step %d, accuracy %g, test_accuracy %g, loss %g"%(i, train_accuracy, test_accuracy, train_loss))

	if (i+1) % 1000 == 0:
		saver.save(sess, './checkpoint/MyModel', global_step=(i+1))

	train_step.run(feed_dict={
		datas_placeholder: rand_x, 
		labels_placeholder: rand_y, 
		other_placeholder: rand_o,
		dropout_placeholder: 0.5
	})

print('train ends')


# use raw data to test the model
test_feed_dict = {
	datas_placeholder: x_test, 
	labels_placeholder: y_test, 
	other_placeholder: o_test,
	dropout_placeholder: 1.0
}
test_accuracy = accuracy.eval(test_feed_dict)
test_loss = cross_entropy.eval(test_feed_dict)

print("accuracy on test data %g, loss on test data %g"%(test_accuracy, test_loss))
# now the classification work finished, and got 100% accuracy
os._exit(0)
'''

# also, the features extracted by neural network can be saved
# then use the extracted features to feed the spark model

# following is using spark to do the classification work
# the input is provided by neural network

# extract the features of raw data
raw_data = []
for i in range(len(x_test)):
	d = x_test[i]
	o = o_test[i]
	l = y_test[i]
	
	temp = sess.run(h_fc2, feed_dict={
		datas_placeholder: np.array(d).reshape((1, 32, 32, 48*2)),
		other_placeholder: np.array(o).reshape((1, 19)),
		dropout_placeholder: 1.0})[0]
	raw_data.append(temp.tolist() + [int(l[0])])

conf = SparkConf()
conf.set('spark.executor.memory', '8g')
conf.set("spark.driver.memory", '8g')
sc = SparkContext(conf = conf)
rdd = sc.parallelize(raw_data)

feature_rdd = rdd.map(lambda line: line[:-1])
scaler = StandardScaler(withStd = True, withMean = True).fit(feature_rdd) 
feature_rdd = scaler.transform(feature_rdd) # feature rdd
label_rdd = rdd.map(lambda line: line[-1:]) # label rdd
# now the 

def getAUC(feature_rdd, label_rdd, do_pca, pca_k, model_type, model_iterations):
	# feature_rdd 
	# label_rdd 
	# do_pca: whether do PCA to select features of not (True/False)
	# pca_k: if do PCA, then how many features to select
	# model_type: type of model (1. LinearRegressionWithSGD 
	# 							 2. LogisticRegressionWithSGD 
	#  							 3. LogisticRegressionWithLBFGS 
	#					  		 4. SVM  5. RandomForest)
	# model_iterations: the train iterations (for RandomForest, this parameter means the amount of trees)
	pca_feature_rdd = feature_rdd
	if do_pca:
		pca_model = PCA(pca_k).fit(feature_rdd)
		pca_feature_rdd = pca_model.transform(feature_rdd)

	dataset = pca_feature_rdd.zip(label_rdd.map(lambda line: line[0])).map(lambda line: LabeledPoint(line[1], line[0]))
	(train_data, test_data) = dataset.randomSplit([0.8, 0.2]) # devide the dataset into train data and test data

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

AUC, recall, precision, f1Score = getAUC(feature_rdd, label_rdd, False, -1, 2, 1000)
print('AUC: ', AUC)
print('recall: ', recall)
print('precision: ', precision)
print('f1Score: ', f1Score)

