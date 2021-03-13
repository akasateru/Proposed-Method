# 情報源領域データを作成
# input: data/内の任意のデータ
# output: sourcevec.npy, source.txt

import os
import csv
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
import re
import string
from tqdm import tqdm
import json

# 不要語削除
def chenge_text(text):
    text = text.translate(str.maketrans( '', '',string.punctuation))
    text = re.sub(r'[.]{2,}','.',text)
    text = re.sub(r'[	]',' ',text)
    text = re.sub(r'[ ]{2,}',' ',text)
    return text

model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

json_file = open('config.json', 'r')
config = json.load(json_file)
max_len = config['max_len']

# 情報源の単語を1回だけ使う
def docvec(row):
    feature_vec = np.zeros((300,),dtype='float32')
    word_stack = [] 
    row = row.replace("\n"," ")
    words = row.split(" ")
    count = 0
    for i,word in enumerate(words):
        if i < max_len:
            if word not in word_stack:
                try: 
                    vec = model.wv[word]
                    feature_vec += vec
                    word_stack.append(word)
                    count += 1
                except:
                    pass
    if count == 0:
        return feature_vec
    else:
        return feature_vec/count

def news20():
    dirpath_list = []
    filenames_list = []
    for dirpath,_,filenames in os.walk('../data'+os.sep+'20news'+os.sep+'20news'):
        dirpath_list.append(dirpath)
        filenames_list.append(filenames)

    j = 0
    vec_list_20news = []
    text_list_20news = []
    for i,dirpath in tqdm(enumerate(dirpath_list),total=len(dirpath_list)):
        for filename in filenames_list[i]:
            j += 1
            with open(dirpath+os.sep+filename,'r',encoding='utf-8',errors='ignore') as f:
                text = f.read()
                text_seiri = []
                text = chenge_text(text)
                for s in text.split('\n'):
                    if re.search(r'\n|From:*|Subject:*|Archive-name:*|Alt-atheism-archive-name:*|Last-modified:*|Version:*|In article*|writes:$',s) == None:
                        text_seiri.append(s)
                text = ''.join(text_seiri)
                vec_list_20news.append(docvec(text))
                text_list_20news.append(text)
    print(len(text_list_20news))
    return vec_list_20news, text_list_20news

def dbpedia():
    # #train.csv
    with open('../data'+os.sep+'dbpedia'+os.sep+'dbpedia_csv'+os.sep+'train.csv','r',encoding='utf-8') as f:
        reader = csv.reader(f)
        vec_list_dbpedia = []
        text_list_dbpedia = []
        l = len(list(reader))
        f.seek(0)
        for row in tqdm(reader,total=l):
            text_stock = []
            text = row[2][1:].replace('(',')').split(')')
            for i,t in enumerate(text):
                if i % 2 == 0:
                    text_stock.append(t)
            text = ''.join(text_stock)
            text = chenge_text(text)
            text = ' '.join([x for x in text.split(' ') if x not in row[1].split(' ')])
            text = text.replace('  ',' ')
            vec_list_dbpedia.append(docvec(text))
            text_list_dbpedia.append(text)

    return vec_list_dbpedia, text_list_dbpedia
    
def reuter():
    with open('../data'+os.sep+'reuter'+os.sep+'sourceall.txt','r',encoding='utf-8') as f:
        source_list = []
        text_list = []
        sourceall = f.read().split('\n')[:-1]
        for source in tqdm(sourceall):
            source = source.split('\t')
            source_list.append(docvec(chenge_text(source[1])))
            text_list.append(chenge_text(source[1]))
    vec_list_reuter = source_list
    text_list_reuter = text_list
    return vec_list_reuter, text_list_reuter

def yahootopic():
    vec_list_yahootopic = []
    text_list_yahootopic = []

   #train1
    with open('../data'+os.sep+'yahootopic'+os.sep+'train_pu_half_v1.txt','r',encoding='utf-8') as f:
        train1 = f.read()
        train1 = "".join([s for s in train1.splitlines(True) if s.strip("\n")])
        train1 = train1.split('\n')[:-1]
        for t in tqdm(train1):
            t = t.split('\t')
            t = [t[0],chenge_text(t[1])]
            vec_list_yahootopic.append(docvec(chenge_text(t[1])))
            text_list_yahootopic.append(chenge_text(t[1]))

   #train0
    with open('../data'+os.sep+'yahootopic'+os.sep+'train_pu_half_v0.txt','r',encoding='utf-8') as f:
        train1 = f.read()
        train1 = "".join([s for s in train1.splitlines(True) if s.strip("\n")])
        train1 = train1.split('\n')[:-1]
        for t in tqdm(train1):
            t = t.split('\t')
            t = [t[0],chenge_text(t[1])]
            vec_list_yahootopic.append(docvec(chenge_text(t[1])))
            text_list_yahootopic.append(chenge_text(t[1]))

    return vec_list_yahootopic, text_list_yahootopic

vec_list_20news, text_list_20news = news20()
vec_list_dbpedia, text_list_dbpedia = dbpedia()
vec_list_reuter, text_list_reuter = reuter()
vec_list_yahootopic, text_list_yahootopic = yahootopic()

vec_list = vec_list_20news + vec_list_dbpedia + vec_list_reuter + vec_list_yahootopic
text_list = text_list_20news + text_list_dbpedia + text_list_reuter + text_list_yahootopic

np.save('../dataset/sourcevec.npy',vec_list)
with open('../dataset/source.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(text_list))
