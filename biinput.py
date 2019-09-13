#!/usr/bin/env python3

"""
Computes the input sequences for the bi-LSTM autoencoder.

Ideas:
 a) remove duplicates
 b) add reference data

"""

import gzip
import os
import numpy as np
from pickle import dump, load
from csv import reader, writer
from collections import Counter


CORPUS = "html_corpus.text.gz"
BINARY_CORPUS = "html_corpus.bin.gz"
TRAINING_CORPUS = "html_training_corpus.bin.gz"
CORPUS_MAX_CHUNK_SIZE = 10000000
CORPUS_MAX_CHUNK_SIZE = 100000
VOCABULARY = "html_vocabulary.cvs.gz"
VOCABULARY_MIN_COUNT = 3

def read_corpus():
    with gzip.open(CORPUS) as f:
        sentences = f.readlines()

    return sentences


def create_vocabulary_file(sentences):
    words = Counter()
    print(len(sentences))
    for sentence in sentences:
        words.update(sentence.decode("utf8").split())

    # write vocabulary file
    vocabulary = {'[MASK]': 0, '[SEP]': 1, '[UNKOWN]': 2}
    with gzip.open(VOCABULARY, 'wt') as f:
        csv = writer(f)
        for word in sorted(words):
            if words[word] >= VOCABULARY_MIN_COUNT:
                vocabulary[word] = len(vocabulary)

        for word, idx in vocabulary.items():
            csv.writerow([word, idx])

    return vocabulary

def read_vocabulary_file(fname):
    vocabulary = {}
    with gzip.open(VOCABULARY, 'rt') as f:
        csv = reader(f)
        for word, idx in csv:
            vocabulary[word] = int(idx)

    return vocabulary

def create_binary_corpus():
    sentences = read_corpus()
    if not os.path.exists(VOCABULARY):
        print("*** Computing vocabulary")
        vocabulary = create_vocabulary_file(sentences)
    else:
        vocabulary = read_vocabulary_file(VOCABULARY)

    binary_corpus = []
    for sentence in sentences:
        binary_corpus += [vocabulary.get(term, vocabulary['[UNKOWN]']) for term in sentence.decode('utf8').split()]
        binary_corpus.append(0) # end of sequence symbol
    return binary_corpus

def get_training_examples(training_sequence, max_estimation_size, mask_value=0, seen_examples=set()):
    '''
    Returns: Training examples for the given training_sequence with up to
             max_estimation_size entries masked.
    '''
    examples = []
    for mask_size in range(1, max_estimation_size+1):
        mask = [mask_value] * mask_size
        for mask_pos in range(len(training_sequence)-mask_size+1):
            example = tuple(training_sequence[:mask_pos] + mask + training_sequence[mask_pos+mask_size:])
            h = hash(example)
            if not h in seen_examples:
                seen_examples.add(h)
                examples.append(example)
    return examples


def create_trainings_corpus(binary_corpus, sliding_window_size, max_estimation_size):
    corpus_sequence = 0

    seen_examples = set()
    x_training_data = []
    y_training_data = []
    last_i = 0
    for i in range(len(binary_corpus)-sliding_window_size):
        # sequence to shuffle
        reference_sequence = binary_corpus[i:i+sliding_window_size]
        training_sequence = get_training_examples(reference_sequence, max_estimation_size, mask_value=0, seen_examples=seen_examples)
        x_training_data.extend(training_sequence)
        y_training_data.extend([reference_sequence] * len(training_sequence))

        # serialize chunk, once CORPUS_MAX_CHUNK_SIZE is reached
        if len(x_training_data) > CORPUS_MAX_CHUNK_SIZE:
            print("Dumping corpus of", len(x_training_data), "examples based on the sliding window position ", i, "with on average ", len(x_training_data)/float(i-last_i), "lines per example.")
            last_len_seen = len(seen_examples)
            #f = open(TRAINING_CORPUS + "." + str(corpus_sequence), 'wb')
            #np.savez(f, np.asarray(training_data))
            with gzip.open(TRAINING_CORPUS + "_x." + str(corpus_sequence), 'w') as f:
                dump(x_training_data, f)
            with gzip.open(TRAINING_CORPUS + "_y." + str(corpus_sequence), 'w') as f:
                dump(y_training_data, f)
            corpus_sequence += 1
            x_training_data = []
            y_training_data = []

    print("Computed", len(training_data), "examples...")
    return training_data

#
# Unit Tests
#
def test_get_training_examples():
    examples = get_training_examples(training_sequence=[1,2,3,4,5,6,7,8,9], max_estimation_size=3, mask_value=0)
    reference = [[0,2,3,4,5,6,7,8,9],
                 [1,0,3,4,5,6,7,8,9],
                 [1,2,0,4,5,6,7,8,9],
                 [1,2,3,0,5,6,7,8,9],
                 [1,2,3,4,0,6,7,8,9],
                 [1,2,3,4,5,0,7,8,9],
                 [1,2,3,4,5,6,0,8,9],
                 [1,2,3,4,5,6,7,0,9],
                 [1,2,3,4,5,6,7,8,0],

                 [0,0,3,4,5,6,7,8,9],
                 [1,0,0,4,5,6,7,8,9],
                 [1,2,0,0,5,6,7,8,9],
                 [1,2,3,0,0,6,7,8,9],
                 [1,2,3,4,0,0,7,8,9],
                 [1,2,3,4,5,0,0,8,9],
                 [1,2,3,4,5,6,0,0,9],
                 [1,2,3,4,5,6,7,0,0],

                 [0,0,0,4,5,6,7,8,9],
                 [1,0,0,0,5,6,7,8,9],
                 [1,2,0,0,0,6,7,8,9],
                 [1,2,3,0,0,0,7,8,9],
                 [1,2,3,4,0,0,0,8,9],
                 [1,2,3,4,5,0,0,0,9],
                 [1,2,3,4,5,6,0,0,0]]
    assert examples == reference



if __name__ == '__main__':
    if not os.path.exists(BINARY_CORPUS):
        binary_corpus = create_binary_corpus()
        with gzip.open(BINARY_CORPUS, 'w') as f:
           dump(binary_corpus, f)
    else:
        with gzip.open(BINARY_CORPUS) as f:
            binary_corpus = load(f)

    if not os.path.exists(TRAINING_CORPUS + ".0"):
        sliding_window_size = 15
        max_estimation_size = 5

        training_corpus = create_trainings_corpus(binary_corpus, sliding_window_size, max_estimation_size)

