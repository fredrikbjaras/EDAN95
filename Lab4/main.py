from datasets import create_dicts
import sys
import os
from sklearn.feature_extraction import DictVectorizer
import time
from keras import models, layers
from keras.models import load_model
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
import math
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Bidirectional, SimpleRNN, Dense


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


def comp_cosines(input, dict):
    list = []
    search_word = dict[input]
    for word in dict:
        word_list = dict[word]
        mul_sum = 0.0
        sqrt1 = 0.0
        sqrt2 = 0.0
        if word != input:
            for i in range(len(word_list)):
                mul_sum += word_list[i] * search_word[i]
                sqrt1 += word_list[i] * word_list[i]
                sqrt2 += search_word[i] * search_word[i]

            list.append((word, mul_sum/(math.sqrt(sqrt1) * math.sqrt(sqrt2))))
    list.sort(key=lambda tup: tup[1], reverse=True)
    return list[:5]

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

def print_result(Y):
    test_file = 'Datasets/eng.test'
    test_sentences = open(test_file).read().strip()
    text_file = open("result.txt", "w")
    rows = test_sentences.splitlines()
    i = 0
    j = 0
    for row in rows:
        if len(row) != 0:
            out = row + ' '
            pred = Y[i][j]
            if pred is None:
                pred = 'O'
            out += pred
            out += '\n'
            text_file.write(out)
            j += 1
        else:
            i += 1
            j = 0
            text_file.write('\n')
    text_file.close()


if __name__ ==  '__main__':

    OPTIMIZER = 'rmsprop'
    BATCH_SIZE = 128
    EPOCHS = 4
    EMBEDDING_DIM = 100
    MAX_SEQUENCE_LENGTH = 150
    LSTM_UNITS = 512


    embedding_file = 'Datasets/glove.6B.100d.txt'
    embeddings_dict = load(embedding_file)
    embeddings_dict['table']

    train_dict, dev_dict, test_dict = create_dicts()

    X, Y = build_sequences(train_dict, 'form', 'ner')

    vocabulary_words = sorted(list(set([word for sentence in X for word in sentence])))
    ner = sorted(list(set([pos for sentence in Y for pos in sentence])))
    print(ner)
    NB_CLASSES = len(ner)

    embeddings_words = embeddings_dict.keys()
    print('Words in GloVe:', len(embeddings_dict.keys()))
    vocabulary_words = sorted(list(set(vocabulary_words +
                                       list(embeddings_words))))
    cnt_uniq = len(vocabulary_words) + 2
    print('# unique words in the vocabulary: embeddings and corpus:',
          cnt_uniq)

    # We start at one to make provision for the padding symbol 0 in RNN and LSTMs
    rev_idx_words = dict(enumerate(vocabulary_words, start=1))
    rev_idx_ner = dict(enumerate(ner, start=1))
    idx_words = {v: k for k, v in rev_idx_words.items()}
    idx_ner = {v: k for k, v in rev_idx_ner.items()}
    print('word index:', list(idx_words.items())[:10])
    print('NER index:', list(idx_ner.items())[:10])

    # We create the parallel sequences of indexes
    X_idx = to_index(X, idx_words)
    Y_idx = to_index(Y, idx_ner)
    print('First sentences, word indices', X_idx[:3])
    print('First sentences, NER indices', Y_idx[:3])

    print(comp_cosines('table', embeddings_dict))
    print(comp_cosines('france', embeddings_dict))
    print(comp_cosines('sweden', embeddings_dict))

    X = pad_sequences(X_idx)
    Y = pad_sequences(Y_idx)

    # The number of POS classes and 0 (padding symbol)
    Y_train = to_categorical(Y, num_classes=len(ner) + 2)
    rdstate = np.random.RandomState(1234567)
    embedding_matrix = rdstate.uniform(-0.05, 0.05, (len(vocabulary_words) + 2, EMBEDDING_DIM))

    for word in vocabulary_words:
        if word in embeddings_dict:
            # If the words are in the embeddings, we fill them with a value
            embedding_matrix[idx_words[word]] = embeddings_dict[word]

    X_dev, Y_dev = build_sequences(dev_dict)

    X_idx_dev = to_index(X_dev, idx_words, len(X_dev))
    Y_idx_dev = to_index(Y_dev, idx_ner, len(Y_dev))

    X_dev = pad_sequences(X_idx_dev)
    Y_dev = pad_sequences(Y_idx_dev)
    Y_train_dev = to_categorical(Y_dev, num_classes=len(ner) + 2)

    NoModel = True
    model_name = 'test.h5'
    if NoModel:

        model = models.Sequential()
        model.add(layers.Embedding(len(vocabulary_words) + 2,
                                   EMBEDDING_DIM,
                                   mask_zero=True,
                                   input_length=None))
        model.layers[0].set_weights([embedding_matrix])
        # The default is True
        model.layers[0].trainable = True
        #model.add(SimpleRNN(100, return_sequences=True))
        model.add(layers.Dropout(0.5))
        #model.add(Bidirectional(SimpleRNN(100, return_sequences=True)))
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Dense(NB_CLASSES + 2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        model.summary()

        model.fit(X, Y_train, validation_data=(X_dev,Y_train_dev), epochs=EPOCHS, batch_size=BATCH_SIZE)
        model.save(model_name)
    else:
        model = load_model(model_name)
        model.summary()

    # In X_dict, we replace the words with their index
    X_test_cat, Y_test_cat = build_sequences(test_dict)
    # We create the parallel sequences of indexes
    X_test_idx = to_index(X_test_cat, idx_words, len(X_test_cat))
    Y_test_idx = to_index(Y_test_cat, idx_ner, len(Y_test_cat))
    print('X[0] test idx', X_test_idx[0])
    print('Y[0] test idx', Y_test_idx[0])

    X_test_padded = pad_sequences(X_test_idx)
    Y_test_padded = pad_sequences(Y_test_idx)
    print('X[0] test idx passed', X_test_padded[0])
    print('Y[0] test idx padded', Y_test_padded[0])
    # One extra symbol for 0 (padding)
    Y_test_padded_vectorized = to_categorical(Y_test_padded,
                                              num_classes=len(ner) + 2)
    print('Y[0] test idx padded vectorized', Y_test_padded_vectorized[0])
    print(X_test_padded.shape)
    print(Y_test_padded_vectorized.shape)

    test_loss, test_acc = model.evaluate(X_test_padded, Y_test_padded_vectorized)
    print('Loss:', test_loss)
    print('Accuracy:', test_acc)

    pred = model.predict(X_test_padded)

    pos_pred_num = []
    for sent_nbr, sent_pos_predictions in enumerate(pred):
        pos_pred_num += [sent_pos_predictions[-len(X_test_cat[sent_nbr]):]]

    pos_pred = []
    for sentence in pos_pred_num:
        pos_idx = list(map(np.argmax, sentence))
        pos_cat = list(map(rev_idx_ner.get, pos_idx))
        pos_pred += [pos_cat]

    #print(pos_pred)
    print_result(pos_pred)