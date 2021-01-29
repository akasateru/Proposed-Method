from nltk.corpus import wordnet as wn
import nltk
import csv

nltk.download("wordnet")

with open('../data/dbpedia/dbpedia_csv/classes.txt','r',encoding='utf-8') as f:
    texts = f.read().splitlines()

wordnet = []
for text in texts:
    dif_all = []
    text = text.split(' ')
    for word in text:
        word = wn.synsets(word)
        if word != []:
            dif = word[0].definition()
            dif = dif.replace(';','')
            dif_all.append(dif)
    wordnet.append(' '.join(dif_all))

with open('../data/dbpedia/dbpedia_csv/classes.csv','w',encoding='utf-8',newline='') as f:
    writer = csv.writer(f)
    for t,w in zip(texts,wordnet):
        writer.writerow([t,w])
