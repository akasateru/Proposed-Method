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

# 要確認
max_len = 128
units = 5 # 対象領域のクラス数
epochs = 20
batch_size = 8
filter_sizes = [3,4,5]

inputs = Input(shape=(max_len,300),dtype='float32')
conv1 = Conv1D(filters=512, kernel_size=filter_sizes[0], kernel_initializer='normal', activation='relu')(inputs)
conv2 = Conv1D(filters=512, kernel_size=filter_sizes[1], kernel_initializer='normal', activation='relu')(inputs)
conv3 = Conv1D(filters=512, kernel_size=filter_sizes[2], kernel_initializer='normal', activation='relu')(inputs)
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

x_train = np.load('../dataset/train/x_train.npy')
y_train = np.load('../dataset/train/y_train.npy')
x_test = np.load('../dataset/test/x_test.npy')
y_test = np.load('../dataset/test/y_test.npy')

result = model.fit(x=x_train,
                   y=y_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(x_test,y_test))

print(result.history.keys())
import matplotlib.pyplot as plt
plt.plot(range(1,epochs+1), result.history['accuracy'], label='acc')
plt.plot(range(1,epochs+1), result.history['loss'], label='loss')
plt.plot(range(1,epochs+1), result.history['val_accuracy'], label='val_acc')
plt.plot(range(1,epochs+1), result.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.savefig('plt.jpg')

model.save('testcnn.h5')

