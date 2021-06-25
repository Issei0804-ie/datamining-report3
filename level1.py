import sys

import nltk
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import sklearn.metrics.pairwise as pairwise
import numpy as np

def print_weight(doc, codebook):
    ''' codebook と 重み を視覚的にわかりやすく表示します(変数名は doc とかじゃなくて weight_vector とかがよいかも)．．
    '''
    l_words = {}

    for i in range(len(doc)):
        if doc[i] > 20:
            l_words.setdefault(codebook[i], doc[i])

    tmp = sorted(l_words.items(), key=lambda x:x[1], reverse=True)
    sortdict = dict(tmp)
    print(sortdict)


def collect_words_eng(docs):
    codebook = []
    wnl = nltk.stem.wordnet.WordNetLemmatizer()
    for doc in docs:
        for sent in sent_tokenize(doc):
            for word in wordpunct_tokenize(sent):
                this_word = wnl.lemmatize(word.lower())
                if this_word not in codebook and this_word not in stopwords:
                    codebook.append(this_word)
    return codebook


def make_vectors_eng(docs, codebook):
    vectors = []
    wnl = nltk.stem.wordnet.WordNetLemmatizer()
    for doc in docs:
        this_vector = []
        fdist = nltk.FreqDist()
        for sent in sent_tokenize(doc):
            for word in wordpunct_tokenize(sent):
                this_word = wnl.lemmatize(word.lower())
                fdist[this_word] += 1
        for word in codebook:
            this_vector.append(fdist[word])
        vectors.append(this_vector)
    return vectors


stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('.')  # ピリオドを追加。
stopwords.append(',')  # カンマを追加。
stopwords.append('')  # 空文字を追加。

# html の読み取り
docs = []
for i in range(13):
    number = str(i).zfill(2)
    filename = str("./html/ch" + number + ".html")
    file = open(filename, mode='r')
    all_of_it = file.read()
    docs.append(all_of_it)
    file.close()

codebook = collect_words_eng(docs)
vectors = make_vectors_eng(docs, codebook)


#print_weight(vectors[0], codebook)
#print("------------------------\n\n")
#print_weight(vectors[8], codebook)
#print("------------------------\n\n")
#print_weight(vectors[9], codebook)



distances = pairwise.cosine_similarity(vectors)

# markdown の表専用に出力
for index in range(len(distances)):
    print("|", end="")
    distance = list(distances[index])
    for i in range(len(distances[index])):
        print(round(distance[i],3), end="")
        print("|", end="")
    print("\n")
