from datasets import create_dicts
import numpy as np


def build_sequences(corpus_dict, key_x='form', key_y='ner', tolower=True):
    """
    Creates sequences from a list of dictionaries
    :param corpus_dict:
    :param key_x:
    :param key_y:
    :return:
    """
    X = []
    Y = []
    for sentence in corpus_dict:
        x = []
        y = []
        for word in sentence:
            x += [word[key_x]]
            y += [word[key_y]]
        if tolower:
            x = list(map(str.lower, x))
        X += [x]
        Y += [y]
    return X, Y

def to_index(X, idx, num_words=None):
    """
    Convert the word lists (or POS lists) to indexes
    :param X: List of word (or POS) lists
    :param idx: word to number dictionary
    :param num_words: total number of words. Used for the unknown word
    :return:
    """
    X_idx = []
    for x in X:
        if num_words:
            # We map the unknown words to the last index of the matrix
            x_idx = list(map(lambda x: idx.get(x, num_words + 1), x))
        else:
            x_idx = list(map(idx.get, x))
        X_idx += [x_idx]
    return X_idx

def comp_cosines(dict1, dict2):
    words1 = list(dict1.values())
    words2 = list(dict2.values())
    mul_sum = 0
    sqrt1 = 0
    sqrt2 = 0
    for i in range(len(words1)):
        mul_sum += words1[i]*words2[i]
        sqrt1 += words1[i]*words1[i]
        sqrt2 += words2[i]*words2[i]

    return mul_sum/(math.sqrt(sqrt1) * math.sqrt(sqrt2))

def load(file):
    """
    Return the embeddings in the from of a dictionary
    :param file:
    :return:
    """
    file = file
    embeddings = {}
    glove = open(file)
    for line in glove:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embeddings[word] = vector
    glove.close()
    embeddings_dict = embeddings
    embedded_words = sorted(list(embeddings_dict.keys()))
    return embeddings_dict

if __name__ ==  '__main__':
    train_dict, dev_dict, test_dict = create_dicts()
    X, Y = build_sequences(train_dict, 'form', 'ner')

    vocabulary_words = sorted(list(set([word for sentence in X for word in sentence])))
    ner = sorted(list(set([pos for sentence in Y for pos in sentence])))
    print(ner)
    NB_CLASSES = len(ner)

    # We start at one to make provision for the padding symbol 0 in RNN and LSTMs
    rev_idx_words = dict(enumerate(vocabulary_words, start=1))
    rev_idx_ner = dict(enumerate(ner, start=1))
    idx_words = {v: k for k, v in rev_idx_words.items()}
    idx_ner = {v: k for k, v in rev_idx_ner.items()}
    print('word index:', list(idx_words.items())[:10])
    print('POS index:', list(idx_ner.items())[:10])

    # We create the parallel sequences of indexes
    X_idx = to_index(X, idx_words)
    Y_idx = to_index(Y, idx_ner)
    print('First sentences, word indices', X_idx[:3])
    print('First sentences, POS indices', Y_idx[:3])

    embedding_file = 'Datasets/glove.6B.100d.txt'
    embeddings_dict = load(embedding_file)
    embeddings_dict['table']

    rdstate = np.random.RandomState(1234567)
    embedding_matrix = rdstate.uniform(-0.05, 0.05, (len(vocabulary_words) + 2, EMBEDDING_DIM))