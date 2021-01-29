import gensim
from gensim.utils import tokenize
from gensim.models import KeyedVectors
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Conv1D, MaxPooling1D, concatenate, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
import csv
import os
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import json
import pickle

# 要確認
json_file = open('config.json','r')
config = json.load(json_file)
units =  config['units'] #クラス数
batch_size = config['batch_size']

model = load_model('textcnn.h5')

# x_test = np.load('../dataset/test/x_test.npy')
# y_test = np.load('../dataset/test/y_test.npy')

fout = open('../dataset/test/x_test.npy','rb')
x_test = np.array(pickle.load(fout))
fout.close()
fout = open('../dataset/test/y_test.npy','rb')
y_test = np.array(pickle.load(fout))
fout.close()

print(x_test.shape)
print(y_test.shape)

y_pred = model.predict(x_test)
print(y_pred[0])

score = model.evaluate(x=[x_test], y=[y_test],batch_size=batch_size)
print("Loss:",score[0])
print("acc:",score[1])

y_pred_list = []
for y_p in y_pred:
    a = [0]*units
    a[np.argmax(y_p)] = 1
    y_pred_list.append(a)
y_pred = np.array(y_pred_list)

rep = metrics.classification_report(y_test,y_pred,digits=3)
print(rep)
with open('result.txt','w',encoding='utf-8') as f:
    f.write(rep)
