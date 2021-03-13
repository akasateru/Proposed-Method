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
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# word2vecモデルの読み込み
word2vec = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

# パラメータの読み込み
json_file = open('config.json','r')
config = json.load(json_file)
max_len = config['max_len']
traindata = config['traindata']
testdata = config['testdata']

#不要語削除
def chenge_text(text):
    text = text.translate(str.maketrans( '', '',string.punctuation))
    text = re.sub(r'[.]{2,}','.',text)
    text = re.sub(r'[	]',' ',text)
    text = re.sub(r'[ ]{2,}',' ',text)
    return text

# dbpedia
# 学習データ
x_train = []
y_train = []
with open(traindata, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        x_train.append(chenge_text(row[1]))
        y_train.append(int(row[0]))
y_train = np_utils.to_categorical(y_train)
    
# テストデータ
x_test = []
y_test = []
with open(testdata,'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        text_stock = []
        text = row[2][1:].replace('(',')').split(')')
        for i,t in enumerate(text):
            if i % 2 == 0:
                text_stock.append(t)
        text = ''.join(text_stock)
        text = chenge_text(text)
        text = ' '.join([x for x in text.split(' ') if x not in row[1].split(' ')])
        text = text.replace('  ',' ')
        x_test.append(text)
        y_test.append(int(row[0])-1)
y_test = np_utils.to_categorical(y_test)

# embedding層に渡すためのベクトルのリストを作成
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train+x_test)
sequences = tokenizer.texts_to_sequences(x_train+x_test)

X = pad_sequences(sequences, maxlen=max_len)
x_train = X[:len(x_train)]
x_test = X[len(x_train):]
print(np.array(x_train).shape)
print(np.array(x_test).shape)

embedding_matrix = np.zeros((max(tokenizer.word_index.values())+1, word2vec.vector_size))
for word, i in tokenizer.word_index.items():
    if word in word2vec.wv:
        embedding_matrix[i] = word2vec.wv[word]

fout = open('../dataset/train/x_train.npy','wb')
pickle.dump(x_train, fout, protocol=4)
fout.close()
fout = open('../dataset/train/y_train.npy','wb')
pickle.dump(y_train, fout, protocol=4)
fout.close()
x_train=""
y_train=""  
fout = open('../dataset/test/x_test.npy','wb')
pickle.dump(x_test, fout, protocol=4)
fout.close()
fout = open('../dataset/test/y_test.npy','wb')
pickle.dump(y_test, fout, protocol=4)
fout.close()
x_test=''
y_test=''

fout = open('../dataset/embedding_matrix.npy','wb')
pickle.dump(embedding_matrix, fout, protocol=4)
fout.close()