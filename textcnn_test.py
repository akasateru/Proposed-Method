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
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import json
import pickle
import matplotlib.pyplot as plt
import itertools

# 混同行列を出力するための関数
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm_normalize = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalize, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 size=5,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# パラメータの読み込み
json_file = open('config.json','r')
config = json.load(json_file)
units =  config['units']
batch_size = config['batch_size']

# 学習したTextCNNモデルの読み込み
model = load_model('textcnn.h5')

# テストデータの読み込み
fout = open('../dataset/test/x_test.npy','rb')
x_test = np.array(pickle.load(fout))
fout.close()
fout = open('../dataset/test/y_test.npy','rb')
y_test = np.array(pickle.load(fout))
fout.close()
print(x_test.shape)
print(y_test.shape)

# テストデータを学習したTextCNNを用いて分類
y_pred = model.predict(x_test)

# 予測結果をone-hotベクトルに変換
y_pred_list = []
for y_p in y_pred:
    pred_onehot = [0]*units
    pred_onehot[np.argmax(y_p)] = 1
    y_pred_list.append(pred_onehot)
y_pred_onehot = np.array(y_pred_list)

# 混同行列を書き出し
labels = ['Com.','Edu.','Art.','Ath.','Off.','Mea.','Bui.', 'Nat.','Vil.','Ani.','Pla.','Alb.','Fil.','Wri.']
y_test_int = np.argmax(y_test,axis=1)
y_pred_int = np.argmax(y_pred,axis=1)
mat = metrics.confusion_matrix(y_test_int,y_pred_int)
plt.figure()
plot_confusion_matrix(mat, classes=labels,title='confusion_matrix')
plt.savefig('confusion_matrix.jpg',dpi=300)

# 再現率、適合率、F値を書き出し
rep = metrics.classification_report(y_test,y_pred_onehot,digits=3)
print(rep)
with open('result.txt','w',encoding='utf-8') as f:
    f.write(rep)