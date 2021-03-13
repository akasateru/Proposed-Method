import gensim
from gensim.utils import tokenize
from gensim.models import KeyedVectors
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, concatenate, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
import csv
import os
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import keras
import json
import pickle
import matplotlib.pyplot as plt
from keras.layers.embeddings import Embedding

# パラメータの設定
json_file = open('config.json','r')
config = json.load(json_file)
max_len = config['max_len']
units = config['units'] # 対象領域のクラス数
epochs = config['epochs']
batch_size = config['batch_size']
filters = 100
filter_sizes = [3,4,5]

# データの読み込み
fout = open('../dataset/train/x_train.npy','rb')
x_train = np.array(pickle.load(fout))
fout.close()
fout = open('../dataset/train/y_train.npy','rb')
y_train = np.array(pickle.load(fout))
fout.close()
fout = open('../dataset/test/x_test.npy','rb')
x_test = np.array(pickle.load(fout))
fout.close()
fout = open('../dataset/test/y_test.npy','rb')
y_test = np.array(pickle.load(fout))
fout.close()

# embedding_matrixの読み込み
fout = open('../dataset/embedding_matrix.npy','rb')
embedding_matrix = np.array(pickle.load(fout))
fout.close()
print(embedding_matrix)

# モデル構造
inputs = Input(shape=(max_len),dtype='float32')
embedding = Embedding(input_dim=embedding_matrix.shape[0],output_dim=embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False, mask_zero=True,name='embedding')(inputs)
conv1 = Conv1D(filters=filters, kernel_size=filter_sizes[0], kernel_initializer='normal', activation='relu')(embedding)
conv2 = Conv1D(filters=filters, kernel_size=filter_sizes[1], kernel_initializer='normal', activation='relu')(embedding)
conv3 = Conv1D(filters=filters, kernel_size=filter_sizes[2], kernel_initializer='normal', activation='relu')(embedding)
pool1 = MaxPooling1D(pool_size=int(conv1.shape[1]),strides=1)(conv1)
pool2 = MaxPooling1D(pool_size=int(conv2.shape[1]),strides=1)(conv2)
pool3 = MaxPooling1D(pool_size=int(conv3.shape[1]),strides=1)(conv3)
x = concatenate([pool1, pool2, pool3])
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(100,activation='relu')(x)
output = Dense(units=units, activation='softmax')(x)
model = Model(inputs, output)
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=True)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# モデルの学習
result = model.fit(x=x_train,
                   y=y_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(x_test,y_test))

# 結果のプロット
print(result.history.keys())
plt.plot(range(1,epochs+1), result.history['accuracy'], label='acc')
plt.plot(range(1,epochs+1), result.history['loss'], label='loss')
plt.plot(range(1,epochs+1), result.history['val_accuracy'], label='val_acc')
plt.plot(range(1,epochs+1), result.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.savefig('plt.jpg')

# モデルの保存
model.save('textcnn.h5')

