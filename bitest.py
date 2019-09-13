#!/usr/bin/env python3

import gzip
from pickle import load
from glob import glob
from csv import reader

import numpy as np
from encode import TermTranslator

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

def estimate_sequence(model, html_sequence, term_translator, sequence_len):
    '''
    Uses the classifier to estimate the given sequence.
    Replaces [MASK] tags with the most likely HTML tag.
    '''
    assert len(html_sequence[0]) == sequence_len
    x = tt.term_sequence_to_matrix(html_sequence)
    print(x.reshape(1, sequence_len*tt.vector_len))
    y = model.predict(x)
    print(y)
    print("INPUT:::", tt.matrix_to_term_sequeence(x.reshape(1, sequence_len*tt.vector_len)))
    print("OUTPUT::", tt.matrix_to_term_sequeence(y))




if __name__ == '__main__':
    vocabulary = read_vocabulary_file(VOCABULARY)
    tt = TermTranslator(vocabulary)
    print("Vocabulary size:", len(vocabulary))

    #
    # preparing the network
    #
    sequence_len = 15                 # max len of the input sequence

    with open(MODEL + '.json') as f:
        model = model_from_json(f.read())

    model.load_weights(MODEL + '.h5')
    print('Loaded model from disk...')

    # run estimation

    html_sequence = ['html', 'body', 'ul', '[SEP]', 'html', 'body', 'ul', '[MASK]', '[MASK]', '[SEP]', 'html', 'body', 'ul', 'li', '[SEP]']
    estimate_sequence(model, [html_sequence], tt, sequence_len)

