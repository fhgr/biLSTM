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
MODEL = "model-g1"

def read_vocabulary_file(fname):
    vocabulary = {}
    with gzip.open(VOCABULARY, 'rt') as f:
        csv = reader(f)
        for word, idx in csv:
            vocabulary[word] = int(idx)

    return vocabulary


def get_vocabulary(html_sequence, vocabulary):
    '''
    Translates the given html_sequence into the corresponding
    word indices within the vocabulary.
    '''
    return [vocabulary.get(term, vocabulary['[UNKNOWN]']) for term in html_sequence]

def translate_prediction_result(sequences, rev_vocabulary):
    result = []
    for sequence in sequences:
        result.append([rev_vocabulary[np.argmax(t)] for t in sequence])
    return result


def estimate_sequence(model, html_sequences, vocabulary, rev_vocabulary):
    '''
    Uses the classifier to estimate the given sequence.
    Replaces [MASK] tags with the most likely HTML tag.
    '''
    x = np.asarray([get_vocabulary(html_sequence, vocabulary) for html_sequence in html_sequences])
    y = model.predict(x)
    print(y)
    print("INPUT:::", html_sequence)
    print("OUTPUT::", translate_prediction_result(y, rev_vocabulary))



if __name__ == '__main__':
    vocabulary = read_vocabulary_file(VOCABULARY)
    rev_vocabulary = {v:k for k, v in vocabulary.items()}
    tt = TermTranslator(vocabulary)
    print("Vocabulary size:", len(vocabulary))

    #
    # preparing the network
    #

    with open(MODEL + '.json') as f:
        model = model_from_json(f.read())

    model.load_weights(MODEL + '.h5')
    print('Loaded model from disk...')

    # run estimation

    prefix = "div div div [SEP] html"
    suffix = "div div div ul li"

    prefix = "div div div div ul"
    suffix = "u b html body div"

    html_sequence = (prefix + ' [MASK] ' + suffix).split(" ")
    estimate_sequence(model, [html_sequence], vocabulary, rev_vocabulary)

