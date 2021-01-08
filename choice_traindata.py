# 情報源領域から対象領域クラスの学習データを選択
# input: sourcevec.npy, 
#        source.txt,
#        yahootopic/classes.txt,
#        GoogleNews-vectors-negative300.bin
# output: choiced_train_data.csv

import csv
import os
import gensim
import numpy as np
from tqdm import tqdm
import json

# 文章のベクトルの平均
model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

json_file = open('config.json','r')
config = json.load(json_file)

max_len = config['max_len']
diff_threshold = config['diff_threshold']
min_count = config['min_count']

def docvec(row,max_len):
    feature_vec = np.zeros((300,),dtype='float32') 
    row = row.replace("\n"," ")
    words = row.split(" ") 
    count = 0
    for i,word in enumerate(words):
        if i < max_len:
            try: 
                vec = model.wv[word]
                feature_vec += vec
                count += 1
            except:
                pass
    if count == 0:
        return feature_vec
    else:
        return feature_vec/count

# 情報源領域データの読み込み
sourcevec = np.load('../dataset/sourcevec.npy')
with open('../dataset/source.txt','r') as f:
    source_text = f.read().splitlines()

# 対象領域のクラス情報の読み込み
with open('../data/dbpedia/dbpedia_csv/classes.csv','r',encoding='utf-8',errors='ignore')as f:
    x = csv.reader(f)
    target_class = []
    for row in x:
        target_class.append(docvec(row[0],max_len))

# cos類似度を基に情報源領域文書を対象領域クラスへ分類
# 情報源領域文書の対象領域クラスへの振り分けとcos類似度の差を計算
choice_list = []
diff_list = []
stack_i_list = []
stack_diff_list = []
for source_v in tqdm(sourcevec):
    cos_list = []
    for class_v in target_class:
        vec1 = source_v
        vec2 = class_v
        cos = np.sum((vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        cos_list.append(cos)
    cos_s = np.sort(np.array(cos_list))[::-1]
    cos_si = np.argsort(cos_list)[::-1]
    rank1 = cos_list[cos_si[0]]
    rank2 = cos_list[cos_si[1]]
    diff = rank1 - rank2
    choice_list.append(cos_si[0])
    diff_list.append(diff)

    # 差の大きいものから順に選択
for num in tqdm(range(len(target_class))):
    stack_i = []
    stack_diff = []
    for i,(choice_l,diff_l) in enumerate(zip(choice_list,diff_list)):
        if num == choice_l and diff_l > diff_threshold:
            stack_i.append(i)
            stack_diff.append(diff_l)
    if min_count > len(stack_i):
        min_count = len(stack_i)

    stack_i_list.append(stack_i)
    stack_diff_list.append(stack_diff)

choicedatanum = []
choicedatanum.append('各クラスの選択された文書数')
for j,i in enumerate(stack_i_list):
    print('class'+str(j)+':'+str(len(i)))
    choicedatanum.append('class'+str(j)+':'+str(len(i)))
choicedatanum.append('各クラスの最大文書数:'+str(min_count))

with open('choicedatanum.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(choicedatanum))

    # 各クラスでmin_count数だけ取ってくる
class_training_data = []
for j,(stack_i,stack_diff) in tqdm(enumerate(zip(stack_i_list,stack_diff_list))):
    sorted_idx = np.argsort(stack_diff)[::-1]
    count = 0
    for s in sorted_idx:
        if count < min_count:
            class_training_data.append([j,source_text[stack_i[s]]])
            count+=1
        else :
            break
    
# 情報源領域から選択した学習データを出力
with open('../dataset'+os.sep+'choiced_train_data.csv','w',encoding='utf-8') as f:
    writer = csv.writer(f)
    for c in class_training_data:
        writer.writerow(c)