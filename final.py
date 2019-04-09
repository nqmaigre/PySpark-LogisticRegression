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

# 读取文件
raw_h5f = h5py.File('raw_data.h5', 'r')
datas = raw_h5f['datas'][:]
others = raw_h5f['others'][:]
labels = raw_h5f['labels'][:]
raw_h5f.close()
print('read raw data ok')

# 读取文件
gen_h5f = h5py.File('gen_data.h5', 'r')
gen_d = gen_h5f['datas'][:]
gen_o = gen_h5f['others'][:]
gen_l = gen_h5f['labels'][:]
print('read gen data ok')

# 打乱顺序
num_example = datas.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
datas = datas[arr]
labels = labels[arr]
others = others[arr]

# 将所有数据分为训练集和验证集
x_test = datas
y_test = labels
o_test = others

x_train = gen_d 
o_train = gen_o 
y_train = gen_l 

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

datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 48*3])
labels_placeholder = tf.placeholder(tf.float32, [None, 2])
dropout_placeholder = tf.placeholder(tf.float32)
other_placeholder = tf.placeholder(tf.float32, [None, 67])

# best
# num_conv1 = 16 # 第二层卷积核数量
# num_conv2 = 32 # 第二层卷积核数量
# num_fc = 128 # inflow outflow 特征提取数量
# num_other = 67 # 其他信息的特征数量
# num_o_fc1 = 128 # 其他信息中提取出的特振数量
# num_fc2 = 128 # 所有特征提取出的特征数量

num_conv1 = 16 # 第二层卷积核数量
num_conv2 = 32 # 第二层卷积核数量
num_fc = 128 # inflow outflow 特征提取数量
num_other = 67 # 其他信息的特征数量
num_o_fc1 = 128 # 其他信息中提取出的特振数量
num_fc2 = 64 # 所有特征提取出的特征数量

W_conv1 = weight_variable([5, 5, 48*3, num_conv1])
b_conv1 = bias_variable([num_conv1])
h_conv1 = tf.nn.relu(conv2d(datas_placeholder, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, num_conv1, num_conv2])
b_conv2 = bias_variable([num_conv2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

b_fc1 = bias_variable([num_fc])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*num_conv2])
W_fc1 = weight_variable([8*8*num_conv2, num_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, dropout_placeholder)

W_o_fc1 = weight_variable([num_other, num_o_fc1])
b_o_fc1 = bias_variable([num_o_fc1])
h_o_fc1 = tf.nn.relu(tf.matmul(other_placeholder, W_o_fc1) + b_o_fc1)
h_o_fc1_drop = tf.nn.dropout(h_o_fc1, dropout_placeholder)

h_concat = tf.concat([h_fc1_drop, h_o_fc1_drop], axis=1) #tf.reshape(tf.concat(1, [h_fc1, other_placeholder]), [-1, num_fc+67])

W_fc2 = weight_variable([num_fc+num_o_fc1, num_fc2])
b_fc2 = bias_variable([num_fc2])
h_fc2 = tf.nn.relu(tf.matmul(h_concat, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, dropout_placeholder)

W_fc3 = weight_variable([num_fc2, 2])
b_fc3 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_conv1)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_conv2)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_fc1)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_o_fc1)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_fc2)
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_fc3)
regularizer = tf.contrib.layers.l2_regularizer(scale=10.0/x_test.shape[0])
reg_term = tf.contrib.layers.apply_regularization(regularizer)

cross_entropy = -tf.reduce_sum(labels_placeholder*tf.log(y_conv))

loss = cross_entropy + reg_term

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(labels_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./checkpoint/MyModel.meta')
saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))

for i in range(500):
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

	train_step.run(feed_dict={
		datas_placeholder: rand_x, 
		labels_placeholder: rand_y, 
		other_placeholder: rand_o,
		dropout_placeholder: 0.5
	})

print('train ends')
test_feed_dict = {
	datas_placeholder: x_test, 
	labels_placeholder: y_test, 
	other_placeholder: o_test,
	dropout_placeholder: 1.0
}
test_accuracy = accuracy.eval(test_feed_dict)
test_loss = cross_entropy.eval(test_feed_dict)

# saver.save(sess, './checkpoint_dir/MyModel')

print("accuracy on test data %g, loss on test data %g"%(test_accuracy, test_loss))


raw_data = []
for i in range(len(x_test)):
	d = x_test[i]
	o = o_test[i]
	l = y_test[i]
	
	temp = sess.run(h_fc2, feed_dict={
		datas_placeholder: np.array(d).reshape((1, 32, 32, 48*3)),
		other_placeholder: np.array(o).reshape((1, 67)),
		dropout_placeholder: 1.0})[0]
	raw_data.append(temp.tolist() + [int(l[0])])

conf = SparkConf()
conf.set('spark.executor.memory', '8g')
conf.set("spark.driver.memory", '8g')
sc = SparkContext(conf = conf)
rdd = sc.parallelize(raw_data)

feature_rdd = rdd.map(lambda line: line[:-1])
scaler = StandardScaler(withStd = True, withMean = True).fit(feature_rdd) # 进行数值的标准化
feature_rdd = scaler.transform(feature_rdd) # 特征 rdd
label_rdd = rdd.map(lambda line: line[-1:]) # 标签 rdd

# sqlContext = SQLContext(sc)
# df = sqlContext.createDataFrame(feature_rdd)
# df.show(5, False)
# os._exit(0)

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

	# sqlContext = SQLContext(sc)
	# df = sqlContext.createDataFrame(dataset)
	# df.show(5, False)
	# os._exit(0)

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
	precision = metrics_mul.precision()
	recall = metrics_mul.recall()
	f1Score = metrics_mul.fMeasure()

	print(dataset.count())
	print(predict_real.count()) # 打印 predict_real 的数据数量
	print(metrics_mul.confusionMatrix()) # 打印混淆矩阵

	return AUC, recall, precision, f1Score

AUC, recall, precision, f1Score = getAUC(feature_rdd, label_rdd, False, -1, 2, 1000)
print(AUC, recall, precision, f1Score)


