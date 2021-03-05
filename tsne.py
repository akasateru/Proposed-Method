import gensim
import json
import numpy as np
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

json_file = open('config.json','r')
config = json.load(json_file)
max_len = config['max_len']

# Word2vecを用いてベクトルに変換
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

with open('../data/dbpedia/dbpedia_csv/classes.csv','r',encoding='utf-8',newline='') as f:
    reader = csv.reader(f)
    classes_vec_list = []
    classes_name_list = []
    for classes_name in reader:
        classes_vec = docvec(classes_name[0],max_len).tolist()
        classes_vec_list.append(classes_vec)
        classes_name_list.append(classes_name[0])

with open('../dataset/choiced_train_data.csv','r',encoding='utf-8',newline='') as f:
    reader = csv.reader(f)
    choiced_vec_list = []
    choiced_class_list = []
    for choiced_train_data in reader:
        choiced_vec = docvec(choiced_train_data[1],max_len).tolist()
        choiced_vec_list.append(choiced_vec)
        choiced_class_list.append(int(choiced_train_data[0]))

vec_list = np.array(classes_vec_list + choiced_vec_list)
embedding = TSNE(n_components=2,random_state=1).fit_transform(vec_list)
class_emb = embedding[:14]
choiced_emb = embedding[14:]

colors = ['b','g','r','c','m','y','gray','coral','aqua','aquamarine','blueviolet','chocolate','darkcyan','fuchsia']

for cls_name,color in zip(classes_name_list,colors):
    plt.scatter([],[],c=color,label=cls_name,s=40)

for c,emb in tqdm(zip(choiced_class_list,choiced_emb),total=len(choiced_class_list)):
    plt.scatter(emb[0],
                emb[1],
                c=colors[c],
                s=0.05)

x = []
y = []
for emb in class_emb:
    x.append(emb[0])
    y.append(emb[1])
plt.scatter(x,y,c='k',s=30)
for i, label in enumerate(classes_name_list):
    plt.annotate(label,(x[i],y[i]))

plt.legend(bbox_to_anchor=(1.05,1),loc='upper left',borderaxespad=0, ncol=2)
plt.savefig('TSNE.jpg',bbox_inches='tight',dpi=1000)