import h5py
import numpy as np
from datetime import datetime
import tensorflow as tf
import logging
import os
import random

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

# 读取文件
raw_h5f = h5py.File('raw_data.h5', 'r')
datas = raw_h5f['datas'][:]
others = raw_h5f['others'][:]
labels = raw_h5f['labels'][:]
raw_h5f.close()
print('read raw data ok')

gen_d, gen_o, gen_l = gen_new_data(datas, others, labels, 30.0)
for i in range(20):
        print('gen batch %d'%(i+1))
        new_d, new_o, new_l = gen_new_data(datas, others, labels)
        gen_d = np.concatenate((gen_d, new_d))
        gen_o = np.concatenate((gen_o, new_o))
        gen_l = np.concatenate((gen_l, new_l))

# 保存到文件
gen_h5f = h5py.File('gen_data2.h5', 'w')
gen_h5f.create_dataset('datas', data=gen_d)
gen_h5f.create_dataset('labels', data=gen_l)
gen_h5f.create_dataset('others', data=gen_o)
gen_h5f.close()
print('save gen data ok')