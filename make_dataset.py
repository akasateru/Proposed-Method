# パディングとベクトル化
# input: choiced_train_data.csv,
#        GoogleNews-vectors-negative300.bin
# output: x_train.npy,
#         y_train.npy,
#         x_test.npy,
#         y_test.npy

import csv
import os
import glob
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
import gensim
from gensim.utils import tokenize
from keras.utils import np_utils
from tqdm import tqdm
import json

wv_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

json_file = open('config.json','r')
config = json.load(json_file)

max_len = config['max_len']
traindata = config['traindata']
testdata = config['testdata']

# パディングとベクトル化
def pad_vec(x_list, max_len):
    pad = [0]*300
    x = []
    for doc in tqdm(x_list):
        vec_list = []
        for word in doc:
            if len(vec_list) < max_len:
                try:
                    vec = wv_model[word]
                except KeyError:
                    pass
                vec_list.append(vec)
        if len(vec_list)<max_len:
            for _ in range(max_len-len(vec_list)):
                vec_list.append(pad)
        x.append(vec_list)
    return x

#不要語削除
def chenge_text(text):
    text = text.translate(str.maketrans( '', '',string.punctuation))
    text = re.sub(r'[.]{2,}','.',text)
    text = re.sub(r'[	]',' ',text)
    text = re.sub(r'[ ]{2,}',' ',text)
    return text

# 選択した学習データの読み込み
x_train = []
y_train = []
with open(traindata, 'r', encoding='utf-8') as f:
    read = csv.reader(f)
    for row in read:
        x_train.append(chenge_text(row[1]))
        y_train.append(int(row[0]))

# テストデータの作成
# test.txtをv0とv1に分割。
x_test = []
y_test = []
with open(testdata,'r',encoding='utf-8') as f:
    texts = csv.reader(f)
    for row in texts:
        text_stock = []
        text = row[2][1:].replace('(',')').split(')')
        for i,t in enumerate(text):
            if i % 2 == 0:
                text_stock.append(t)
        text = ''.join(text_stock)
        text = chenge_text(text)
        text = ' '.join([x for x in text.split(' ') if x not in row[1].split(' ')])
        text = text.replace('  ',' ')
        y_test.append(int(row[0])-1)
        x_test.append(text)

x_train = pad_vec(x_train,max_len)
y_train = np_utils.to_categorical(y_train)
x_test = pad_vec(x_test,max_len)
y_test = np_utils.to_categorical(y_test)

np.save('../dataset/train/x_train.npy',x_train)
np.save('../dataset/train/y_train.npy',y_train)
np.save('../dataset/test/x_test.npy',x_test)
np.save('../dataset/test/y_test.npy',y_test)