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

# 要確認
units = 5 #クラス数
batch_size = 32

model = load_model('testcnn.h5')

x_test = np.load('../dataset/test/x_test.npy')
y_test = np.load('../dataset/test/y_test.npy')

print(x_test.shape)
print(y_test.shape)

y_pred = model.predict(x_test)

score = model.evaluate(x=[x_test], y=[y_test],batch_size=batch_size)
print("Loss:",score[0])
print("acc:",score[1])

y_pred_list = []
for y_p in y_pred:
    a = [0]*units
    a[np.argmax(y_p)] = 1
    y_pred_list.append(a)
y_pred = np.array(y_pred_list)

rep = metrics.classification_report(y_test,y_pred,digits=4)
print(rep)
with open('result.txt','w') as f:
    f.write(rep)
