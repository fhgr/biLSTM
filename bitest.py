#!/usr/bin/env python3

import gzip
from pickle import load
from glob import glob
from csv import reader

import numpy as np
from encode import int_to_vector, vector_to_int

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model, model_from_json
from keras.initializers import Constant

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Flatten, Bidirectional, Dropout

VOCABULARY = "html_vocabulary.cvs.gz"
MODEL = "model1"

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

def index_to_matrix(index_sequence, vocabulary_size):
    vocabulary_vector_size = vocabulary_size.bit_length()
    result = np.full([1, len(index_sequence), vocabulary_vector_size], -1)
    result[0] = [int_to_vector(v, vocabulary_vector_size) for v in index_sequence]
    return result


def get_vocabulary(html_sequence, vocabulary):
    '''
    Translates the given html_sequence into the corresponding
    word indices within the vocabulary.
    '''
    return index_to_matrix([vocabulary.get(term, vocabulary['[UNKOWN]']) for term in html_sequence], len(vocabulary))

def estimate_sequence(model, html_sequence, vocabulary, rev_vocabulary, sequence_len):
    '''
    Uses the classifier to estimate the given sequence.
    Replaces [MASK] tags with the most likely HTML tag.
    '''
    assert len(html_sequence) == sequence_len
    x = get_vocabulary(html_sequence, vocabulary)
    print(x)
    y = model.predict(x)
    print(y)
    print(binary_matrix_to_word_index(y, rev_vocabulary, sequence_len))

def binary_matrix_to_word_index(result, rev_vocabulary, sequence_len):
    html = []
    for element in result.reshape(sequence_len, len(rev_vocabulary).bit_length()):
        print(element)
        word = rev_vocabulary[vector_to_int(element)]
        html.append(word)

    return html




if __name__ == '__main__':
    vocabulary = read_vocabulary_file(VOCABULARY)
    rev_vocabulary = {v:k for k,v in vocabulary.items()}
    print("Vocabulary size:", len(vocabulary))

    #
    # preparing the network
    #
    vocabulary_size = len(vocabulary)  # number of features in the vocabulary
    sequence_len = 15                 # max len of the input sequence

    with open(MODEL + '.json') as f:
        model = model_from_json(f.read())

    model.load_weights(MODEL + '.h5')
    print('Loaded model from disk...')

    # run estimation

    html_sequence = ['html', 'body', 'ul', '[SEP]', 'html', 'body', 'ul','[MASK]', '[MASK]', '[SEP]', 'html', 'body', 'ul', 'li', '[SEP]']
    estimate_sequence(model, html_sequence, vocabulary, rev_vocabulary, sequence_len)

