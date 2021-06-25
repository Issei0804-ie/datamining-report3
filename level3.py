import numpy as np

def preprocess(text):
    """テキストに対する前処理。
    「ゼロから作るDeepLearning2 自然言語処理辺」p.66より。

    :param text:
    :return:
      courpus(list): id_to_wordのidに基づいたone-hot vector。
      word_to_id(dict): 単語をkeyとして、idを参照する辞書。
      id_to_word(dict): idをkeyとして、単語を参照する辞書。
    """
    text = text.lower()
    text = text.replace('.', ' .')
    text = text.replace('"', '')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    """共起行列を作成。
    「ゼロから作るDeepLearning2 自然言語処理辺」p.72より。

    :param corpus(str): テキスト文。
    :param vocab_size: 語彙数。
    :param window_size: 共起判定の範囲。
    :return:
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x/(np.sqrt(np.sum(x**2))+eps)
    ny = y/(np.sqrt(np.sum(y**2))+eps)
    return np.dot(nx, ny)


def most_similar(query, searches, word_to_id, id_to_word, word_matrix, top=5):
    """コサイン類似度Top5を出力。

    :param query(str): クエリ。
    :param searches(list): 検索させたい単語
    :param word_to_id(dict): 単語をkeyとして、idを参照する辞書。
    :param id_to_word(dict): idをkeyとして、単語を参照する辞書。
    :param word_matrix: 共起行列。
    :param top(int): 上位何件まで表示させるか。
    :return: なし。
    """
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    for search in searches:
        try:
            search_id = word_to_id[search]
            print(round(similarity[search_id],3))
        except:
            print(0.000)




text = ""
for i in range(13):
    number = str(i).zfill(2)
    filename = str("./html/ch" + number + ".html")
    file = open(filename, mode='r')
    all_of_it = file.read()
    text = text + " " + all_of_it
    file.close()



corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
co_matrix = create_co_matrix(corpus, vocab_size, window_size=2)


searches = ["natural", "language", "text", "count", "python"]
for query in searches:
    most_similar(query, searches, word_to_id, id_to_word, co_matrix)
# search_similar(query_list, word_to_id, id_to_word, word_matrix):