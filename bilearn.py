#!/usr/bin/env python3

import gzip
from pickle import load
from glob import glob
from csv import reader

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.initializers import Constant

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Flatten, Bidirectional, Dropout

TRAINING_CORPUS_X = ["html_training_corpus.bin.gz_x.?", ]
TRAINING_CORPUS_Y = ["html_training_corpus.bin.gz_y.?", ]
VOCABULARY = "html_vocabulary.cvs.gz"

def read_vocabulary_file(fname):
    vocabulary = {}
    with gzip.open(VOCABULARY, 'rt') as f:
        csv = reader(f)
        for word, idx in csv:
            vocabulary[word] = int(idx)

    return vocabulary


def get_matrix(data, sequence_len, vocabulary_size):
    print("Creating data matrix...")
    result = np.full([len(data), sequence_len, vocabulary_size], -1)
    for no, example in enumerate(data):
        for word_idx, word in enumerate(example):
            result[no][word_idx][word] = 1
    print("Completed computation of data matrix...")
    return result

def test_get_matrix():
    data = [[1, 0],
            [0, 2],
            [1, 2]]
    matrix = get_matrix(data, 2, 3)
    reference = np.asarray([[[-1, 1, -1], [1, -1, -1]],
                            [[1, -1, -1], [-1, -1, 1]],
                            [[-1, 1, -1], [-1, -1, 1]]])

    assert np.array_equal(matrix, reference)


def l(corpus_pattern):
    ''' loads all corpora matching the given corpus_pattern '''
    result = []
    for fname in sorted(glob(corpus_pattern)):
        with gzip.open(fname) as f:
            result.extend(load(f))
    return result


if __name__ == '__main__':
    vocabulary = read_vocabulary_file(VOCABULARY)
    print("Vocabulary size:", len(vocabulary))

    #
    # preparing the network
    #
    vocabulary_size = len(vocabulary)  # number of features in the vocabulary
    sequence_len = 15                 # max len of the input sequence

    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(sequence_len, vocabulary_size)))
    model.add(Dropout(0.5))
    model.add(Dense(sequence_len*vocabulary_size, activation='tanh'))
    model.build()
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
    # 
    # preparing the training data
    #
    for training_corpus_x, training_corpus_y in zip(TRAINING_CORPUS_X, TRAINING_CORPUS_Y):
        print("Training bi-LSTM with corpus:", training_corpus_x)
        data_x = get_matrix(l(training_corpus_x), sequence_len, vocabulary_size)
        corpus_y = l(training_corpus_y)
        data_y = get_matrix(corpus_y, sequence_len, vocabulary_size).reshape(len(corpus_y), sequence_len*vocabulary_size)

        num_validation_samples = int(len(data_x)*0.1)
        x_train = data_x[:-num_validation_samples]
        y_train = data_y[:-num_validation_samples]
        x_val = data_x[-num_validation_samples:]
        y_val = data_y[-num_validation_samples:]

        model.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_val, y_val))

    # save the model
    with open('model.json', 'w') as f:
        f.write(model.to_json())

    # save the model weights
    model.save_weights('model.h5')
