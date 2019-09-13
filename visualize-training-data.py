#!/usr/bin/env python3

import gzip
import numpy as np

from encode import TermTranslator

from glob import glob
from pickle import load
from csv import reader

TRAINING_CORPUS_X = "./train/html_training_corpus.bin.gz_x.?" 
TRAINING_CORPUS_Y = "./train/html_training_corpus.bin.gz_y.?" 
VOCABULARY = "html_vocabulary.cvs.gz"

def l(corpus_pattern):
    ''' loads all corpora matching the given corpus_pattern '''
    result = []
    for fname in sorted(glob(corpus_pattern)):
        with gzip.open(fname) as f:
            result.extend(load(f))
    print("Read:", len(result), "records.")
    return result


def get_matrix(data, sequence_len, term_translator, size, skip):
    print("Creating data matrix...")
    result = np.full([size, sequence_len, term_translator.vector_len], 0)
    for no, example in enumerate(data):
        if no < skip:
            continue
        result[no-skip] = [term_translator.int_to_vector(v) for v in example]
        if no == (skip+size-1):
            break
    print("Completed computation of data matrix with shape", result[0].shape, "...")
    return result

def read_vocabulary_file(fname):
    vocabulary = {}
    with gzip.open(VOCABULARY, 'rt') as f:
        csv = reader(f)
        for word, idx in csv:
            vocabulary[word] = int(idx)

    return vocabulary

if __name__ == '__main__':
    vocabulary = read_vocabulary_file(VOCABULARY)
    tt = TermTranslator(vocabulary)
    print("Vocabulary size:", len(vocabulary))
    print("Vocabulary vector size:", tt.vector_len)

    sequence_len = 15
    
    sample_size = 5
    data_y = get_matrix(l(TRAINING_CORPUS_Y), sequence_len, tt, sample_size, skip=500000).reshape(sample_size, sequence_len*tt.vector_len)
    data_x = get_matrix(l(TRAINING_CORPUS_X), sequence_len, tt, sample_size, skip=500000).reshape(sample_size, sequence_len*tt.vector_len)

    x = tt.matrix_to_term_sequeence(data_x)
    y = tt.matrix_to_term_sequeence(data_y)

    for xx, yy in zip(x, y):
        print("\t".join(xx) + "\n" + "\t".join(yy))
        print("----------")


