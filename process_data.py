import h5py
import numpy as np
from datetime import datetime
import tensorflow as tf
import logging
import os
import random

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

f1.close()
f2.close()

# select the days which have 48 slots
# some days may just have 46 or 47 slots (these days should be deleted from data)
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

# use selected days to generate raw data
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

	flow = 32*[32*[48*2*[0]]]
	count = 0
	for m in range(48):
		# i+m 
		inflow = data[i+m][0] # 32*32
		outflow = data[i+m][1] # 32*32	
		for j in range(32):
			for k in range(32):
				flow[j][k][2*m] = inflow[j][k]/100.0
				flow[j][k][2*m+1] = outflow[j][k]/100.0

	datas.append(flow)
	labels.append([1, 0] if is_weekend else [0, 1])

	item = []
	item += [temperature[i].item()/10.0, windspeed[i].item()/10.0]
	item += map(lambda x: int(x), weather[i].tolist())
	others.append(item)

	i += 48

datas = np.array(datas)
labels = np.array(labels)
others = np.array(others)
print(len(datas))
print(len(labels))
print(len(others))

# raw data is saved in raw_data.h5
raw_h5f = h5py.File('raw_data.h5', 'w')
raw_h5f.create_dataset('datas', data=datas)
raw_h5f.create_dataset('labels', data=labels)
raw_h5f.create_dataset('others', data=others)
raw_h5f.close()
print('save raw data ok')


# the function is to use raw data to generate more data
def gen_new_data(raw_d, raw_o, raw_l, limit=10.0):
	length = len(raw_l)
	new_d = []
	new_o = []
	new_l = []
	for i in range(length):
		d = raw_d[i] # 32*32*(48*3)
		o = raw_o[i] # 67
		l = raw_l[i]
		d_ = 32*[32*[48*2*[0]]]
		o_ = 19*[0]
		l_ = 2*[0]

		for j in range(32):
			for k in range(32):
				for m in range(48*2):
					d_[j][k][m] = d[j][k][m].copy()
					d_[j][k][m] *= 1 + (random.uniform(-limit, limit)/100)
					# print(d[j][k][m], ' ', d_[j][k][m])

		for j in range(19):
			o_[j] = o[j].copy()
			o_[j] *= 1 + (random.uniform(-limit, limit)/100)
			# print(o[j], ' ', o_[j])

		l_ = l.copy()

		new_d.append(d_)
		new_o.append(o_)
		new_l.append(l_)

	return np.array(new_d), np.array(new_o), np.array(new_l)

# use gen_new_data to generate more datas
gen_d, gen_o, gen_l = gen_new_data(datas, others, labels, 20.0)
for i in range(4):
	print('gen batch %d'%(i+1))
	new_d, new_o, new_l = gen_new_data(datas, others, labels, 10.0)
	gen_d = np.concatenate((gen_d, new_d))
	gen_o = np.concatenate((gen_o, new_o))
	gen_l = np.concatenate((gen_l, new_l))

# the generated data is saved in gen_data.h5
gen_h5f = h5py.File('gen_data.h5', 'w')
gen_h5f.create_dataset('datas', data=gen_d)
gen_h5f.create_dataset('labels', data=gen_l)
gen_h5f.create_dataset('others', data=gen_o)
gen_h5f.close()
print('save gen data ok')
	