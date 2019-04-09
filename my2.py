import h5py
import numpy as np
from datetime import datetime
import tensorflow as tf
import logging
import os
import random
# import pyspark as sp
# from pyspark.conf import SparkConf
# from pyspark.context import SparkContext
# from datetime import datetime
# from pyspark.sql import SQLContext
# from pyspark.mllib.linalg import Vectors
# from pyspark.mllib.feature import StandardScaler, Normalizer, ChiSqSelector
# from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
# from pyspark.mllib.regression import LinearRegressionWithSGD
# from pyspark.mllib.regression import LabeledPoint
# from pyspark.mllib.feature import PCA
# from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
# from pyspark.mllib.tree import RandomForest
# from pyspark.mllib.classification import SVMModel, SVMWithSGD

logging.getLogger('tensorflow').disabled = True

'''
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
f1.close()
f2.close()

d = dict()
for i in range(7220):
	timeslot = timeslots[i].decode('utf8')
	date = timeslot[:8]
	slot = int(timeslot[8:])

	if date in d:
		d[date][slot-1] = 1
	else:
		d[date] = 48*[0]
		d[date][slot-1] = 1

s = set()
for key in d:
	temp = d[key]
	valid = True
	for i in range(48):
		if temp[i] == 0:
			valid = False
			break

	if valid:
		s.add(key)

print(len(s))

datas = []
labels = []
others = []
i = 0
while True:
	if(i >= 7220):
		break

	timeslot = timeslots[i].decode('utf8')
	date = timeslot[:8]
	if not date in s:
		i += 1
		continue
	is_weekend = isWeekend(date)
	slot = int(timeslot[8:])

	flow = 32*[32*[48*3*[0]]]
	count = 0
	for m in range(48):
		# i+m 
		inflow = data[i+m][0] # 32*32
		outflow = data[i+m][1] # 32*32	
		for j in range(32):
			for k in range(32):
				flow[j][k][3*m] = inflow[j][k]/100.0
				flow[j][k][3*m+1] = outflow[j][k]/100.0
				flow[j][k][3*m+1] = (inflow[j][k]-outflow[j][k])/100.0

	# flow = 32*[32*[48*3*[0]]]
	# count = 0
	# for m in range(48):
	# 	# i+m 
	# 	inflow = data[i+m][0] # 32*32
	# 	outflow = data[i+m][1] # 32*32	
	# 	for j in range(32):
	# 		for k in range(32):
	# 			flow[j][k][3*m] = inflow[j][k]/100.0
	# 			flow[j][k][3*m+1] = outflow[j][k]/100.0
	# 			flow[j][k][3*m+2] = (inflow[j][k]-outflow[j][k])/100.0

	datas.append(flow)
	labels.append([1, 0] if is_weekend else [0, 1])

	item = []
	item += [temperature[i].item()/10.0, windspeed[i].item()/10.0]
	# print(temperature[i], ' ', windspeed[i])
	item += map(lambda x: int(x), weather[i].tolist())
	slot_one_hot = [0] * 48
	slot_one_hot[slot-1] = 1
	item += slot_one_hot
	# item += [1 if is_weekend else 0]
	others.append(item)

	i += 48

datas = np.array(datas)
labels = np.array(labels)
others = np.array(others)
print(len(datas))
print(len(labels))
print(len(others))

# 保存到文件
raw_h5f = h5py.File('raw_data.h5', 'w')
raw_h5f.create_dataset('datas', data=datas)
raw_h5f.create_dataset('labels', data=labels)
raw_h5f.create_dataset('others', data=others)
raw_h5f.close()
print('save raw data ok')
'''

# 读取文件
raw_h5f = h5py.File('raw_data.h5', 'r')
datas = raw_h5f['datas'][:]
others = raw_h5f['others'][:]
labels = raw_h5f['labels'][:]
raw_h5f.close()
print('read raw data ok')

'''
def gen_new_data(raw_d, raw_o, raw_l, limit=10.0):
	length = len(raw_l)
	new_d = []
	new_o = []
	new_l = []
	for i in range(length):
		d = raw_d[i] # 32*32*(48*3)
		o = raw_o[i] # 67
		l = raw_l[i]
		d_ = 32*[32*[48*3*[0]]]
		o_ = 67*[0]
		l_ = 2*[0]

		for j in range(32):
			for k in range(32):
				for m in range(48*3):
					d_[j][k][m] = d[j][k][m].copy()
					d_[j][k][m] *= 1 + (random.uniform(-limit, limit)/100)
					# print(d[j][k][m], ' ', d_[j][k][m])

		for j in range(67):
			o_[j] = o[j].copy()
			o_[j] *= 1 + (random.uniform(-limit, limit)/100)
			# print(o[j], ' ', o_[j])

		l_ = l.copy()

		new_d.append(d_)
		new_o.append(o_)
		new_l.append(l_)

	return np.array(new_d), np.array(new_o), np.array(new_l)

# 打乱顺序
num_example = datas.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
datas = datas[arr]
labels = labels[arr]
others = others[arr]

# 将所有数据分为训练集和验证集
ratio = 0.5
train_size = np.int(num_example*ratio)
x_train = datas[:train_size]
o_train = others[:train_size]
y_train = labels[:train_size]
x_test = datas[train_size:]
y_test = labels[train_size:]
o_test = others[train_size:]

gen_d, gen_o, gen_l = gen_new_data(datas, others, labels, 30.0)
for i in range(5):
	print('gen batch %d'%(i+1))
	new_d, new_o, new_l = gen_new_data(datas, others, labels, 15.0)
	gen_d = np.concatenate((gen_d, new_d))
	gen_o = np.concatenate((gen_o, new_o))
	gen_l = np.concatenate((gen_l, new_l))

# 保存到文件
gen_h5f = h5py.File('gen_data.h5', 'w')
gen_h5f.create_dataset('datas', data=gen_d)
gen_h5f.create_dataset('labels', data=gen_l)
gen_h5f.create_dataset('others', data=gen_o)
gen_h5f.close()
print('save gen data ok')
'''

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
ratio = 0.2
train_size = np.int(num_example*ratio)
x_train = datas[:train_size]
o_train = others[:train_size]
y_train = labels[:train_size]
x_test = datas[train_size:]
y_test = labels[train_size:]
o_test = others[train_size:]

x_train = np.concatenate((x_train, gen_d))
o_train = np.concatenate((o_train, gen_o))
y_train = np.concatenate((y_train, gen_l))

print(len(x_train))
print(len(o_train))
print(len(y_train))

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

num_conv1 = 16 # 第二层卷积核数量
num_conv2 = 64 # 第二层卷积核数量
num_fc = 256 # inflow outflow 特征提取数量
num_other = 67 # 其他信息的特征数量
num_o_fc1 = 256 # 其他信息中提取出的特振数量
num_fc2 = 256 # 所有特征提取出的特征数量

W_conv1 = weight_variable([5, 5, 48*3, num_conv1])
b_conv1 = bias_variable([num_conv1])
h_conv1 = tf.nn.relu(conv2d(datas_placeholder, W_conv1) + b_conv1)
h_pool1 = max_pool_4x4(h_conv1)

# W_conv2 = weight_variable([3, 3, num_conv1, num_conv2])
# b_conv2 = bias_variable([num_conv2])
# h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

b_fc1 = bias_variable([num_fc])
h_pool1_flat = tf.reshape(h_pool1, [-1, 8*8*num_conv1])
W_fc1 = weight_variable([8*8*num_conv1, num_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
# h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*num_conv2])
# W_fc1 = weight_variable([16*16*num_conv2, num_fc])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
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
# tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_conv2)
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

for i in range(20000):
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
print("accuracy on test data %g, loss on test data %g"%(test_accuracy, test_loss))

# rand_index = np.random.choice(datas.shape[0], size = 64)
# rand_x = datas[rand_index]
# rand_y = labels[rand_index]	