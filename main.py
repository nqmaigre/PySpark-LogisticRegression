import h5py
import numpy as np
import pyspark as sp
from datetime import datetime
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
# from pyspark.ml.feature import PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint


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

for i in range(7220):
	timeslot = timeslots[i].decode('utf8')
	date = timeslot[:8]
	is_weekend = isWeekend(date)
	slot = int(timeslot[8:])

	# inflows of all grids is equal to outflows of all grids
	# inflows = 0
	# outflows = 0
	flows = 0
	for j in range(32):
		for k in range(32):
			# inflows += data[i][0][j][k]
			# outflows += data[i][1][j][k]
			flows += data[i][0][j][k]

	# raw_data.append([slot, inflows, outflows, temperature[i], windspeed[i], weather[i], is_weekend])
	# raw_data.append([slot, flows.item(), temperature[i].item(), windspeed[i].item(), weather[i].tolist(), 1 if is_weekend else 0])
	item = [slot, flows.item(), temperature[i].item(), windspeed[i].item()]
	item += map(lambda x: int(x), weather[i].tolist())
	map(lambda x: float(x), item)
	item += [1 if is_weekend else 0]
	raw_data.append(item)

# print(raw_data)
	
sc = sp.SparkContext()
rdd = sc.parallelize(raw_data)
# print('共有' + str(rdd.count()) + ' 项数据')

feature_rdd = rdd.map(lambda line: line[:-1])
label_rdd = rdd.map(lambda line: line[-1:])

sqlContext = SQLContext(sc)
# dataset = rdd.map(lambda x: (Vectors.dense([x[i] for i in range(0, 21)]), x[21]))
# dataset = sqlContext.createDataFrame(dataset, ['features', 'label'])
# feature_data.show(5, False)
# print(feature_data.printSchema)

# scaler = StandardScaler(inputCol = 'features', outputCol = 'scaled_features', withStd = True, withMean = False).fit(dataset)
# scaled_dataset = scaler.transform(dataset)

# feature_rdd = sqlContext.createDataFrame(feature_rdd)
# feature_rdd.first()
scaler = StandardScaler(withStd = True, withMean = True).fit(feature_rdd)
scaler_feature_dataset = scaler.transform(feature_rdd)

# dataset = scaler_feature_dataset.zip(label_rdd) #.map(lambda x: (Vectors.dense([x[i] for i in range(0, 21)]), x[21]))
# print(dataset.collect())
dataset = scaler_feature_dataset.zip(label_rdd.map(lambda line: line[0])).map(lambda line: LabeledPoint(line[1], line[0]))

# dataset = sqlContext.createDataFrame(dataset, ['features', 'label'])
# db.show(5)
(train_data, test_data) = dataset.randomSplit([0.8, 0.2])

# train_data = sqlContext.createDataFrame(train_data, ['features', 'label'])
# train_data.show(5)

# pca = PCA(k = 10, inputCol = "scaled_features", outputCol = "pca_features")
# lr = LogisticRegression(maxIter = 10, featuresCol = 'features', labelCol = 'label')

# pipeline = Pipeline(stages = [lr])

# model = pipeline.fit(train_data)
# results = model.transform(test_data)

# results.select('probability', 'prediction', 'prediction').show(truncate = False)
model = LinearRegressionWithSGD.train(train_data, iterations=100)

from pyspark.mllib.evaluation import BinaryClassificationMetrics

## 定义模型评估函数
def evaluateModel(model, validationData):
    ## 使用模型进行预测（作用于验证集上）
    ## 计算AUC
    predict = model.predict(validationData.map(lambda p:p.features))
    predict = predict.map(lambda p: float(p))
    ## 拼接预测值和实际值
    predict_real = predict.zip(validationData.map(lambda p: p.label))
#     predict_real.take(5)
    metrics = BinaryClassificationMetrics(predict_real)
    metrics.areaUnderROC
    return metrics.areaUnderROC

AUC = evaluateModel(model, test_data)
## 打印AUC
print("AUC="+str(AUC))

f1.close()
f2.close()
