import nltk
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import sklearn.metrics.pairwise as pairwise
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def print_weight(doc, codebook):
    ''' codebook と 重み を視覚的にわかりやすく表示します(変数名は doc とかじゃなくて weight_vector とかがよいかも)．．
    '''
    l = 0 < doc
    l_words = {}
    for i in range(len(l)):
        if l[i]:
            l_words.setdefault(codebook[i], doc[i])


    tmp = sorted(l_words.items(), key=lambda x:x[1], reverse=True)
    sortdict = dict(tmp)
    print(sortdict)


# html の読み取り
docs = []
for i in range(13):
    number = str(i).zfill(2)
    filename = str("./html/ch" + number + ".html")
    file = open(filename, mode='r')
    all_of_it = file.read()
    docs.append(all_of_it)
    file.close()

import sklearn.feature_extraction.text as fe_text

def bow_tfidf(docs):
    '''Bag-of-WordsにTF-IDFで重み調整したベクトルを生成。

    :param docs(list): 1文書1文字列で保存。複数文書をリストとして並べたもの。
    :return: 重み調整したベクトル。
    '''
    vectorizer = fe_text.TfidfVectorizer(norm=None, stop_words='english')
    vectors = vectorizer.fit_transform(docs)
    return vectors.toarray(), vectorizer

vectors, vectorizer = bow_tfidf(docs)
#print('# BoW + tfidf')
#print(vectorizer.get_feature_names())
#print(type(vectors))

#print_weight(vectors[0],vectorizer.get_feature_names())
#print("------------------------\n\n")
print_weight(vectors[8],vectorizer.get_feature_names())
print("------------------------\n\n")
print_weight(vectors[9],vectorizer.get_feature_names())
print("------------------------\n\n")
'''
sim = cosine_similarity(vectors) # 類似度行列の作成
for from_id in range(len(docs)):

    print("|", end="")
    for to_id in range(len(docs)):
        print(round(sim[from_id][to_id], 3), end="")
        print("|", end="")
    print("\n")


'''
