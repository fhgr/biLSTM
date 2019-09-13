#!/usr/bin/env python3

import numpy as np

UKN = '[UNKNOWN]'

class TermTranslator(object):
    ''' translates terms and sentences into the corresponding
        bit patterns and vice versa '''

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.rev_vocabulary = {v:k for k,v in vocabulary.items()}
        self.vector_len = len(vocabulary).bit_length()
        self.unknown_term_id = vocabulary[UKN]

    def term_sequence_to_matrix(self, s):
        '''
        Transforms a list of term sequences to the corresponding matrix.

        s: the sequence to transform
        '''
        result = np.empty([len(s), len(s[0]), self.vector_len])
        for no, example in enumerate(s):
            result[no] = [self.term_to_vector(e) for e in example]
        return result

    def matrix_to_term_sequeence(self, m):
        result = []
        for mm in m:
            html = []
            sequence_len = int(len(mm)/self.vector_len)
            for v in mm.reshape(sequence_len, self.vector_len):
                html.append(self.vector_to_term(v))
            result.append(html)
        return result

    def term_to_vector(self, t):
        return self.int_to_vector(self.vocabulary.get(t, self.unknown_term_id))

    def vector_to_term(self, v):
        return self.rev_vocabulary[self.vector_to_int(v)]

    def int_to_vector(self, i):
        v = np.full(self.vector_len, fill_value=-1)
        for no, bit in enumerate('{0:08b}'.format(i)):
            v[no] = 1 if bit == '1' else -1
        return v
    
    def vector_to_int(self, v):
        s = ''.join(['0' if vv < 0 else '1' for vv in (v)])
        return int(s, 2)


def test_init_to_vector():
    vocabulary = {k: k for k in range(254)}
    vocabulary[UKN] = 255
    tt = TermTranslator(vocabulary)
    for i in range(255):
        v = tt.int_to_vector(i)
        assert i == tt.vector_to_int(v)
   

def test_int_to_vector():
    vocabulary = {k: k for k in range(254)}
    vocabulary[UKN] = 255
    tt = TermTranslator(vocabulary)
    v = tt.int_to_vector(0)
    assert len(v) == 8
    assert sum(v) == -8
    assert tt.vector_to_int(v) == 0

    v = tt.int_to_vector(255)
    assert sum(v) == 8
    assert tt.vector_to_int(v) == 255

    v = tt.int_to_vector(7)
    assert sum(v) == 3 - 5
    assert tt.vector_to_int(v) == 7

def test_term_to_vector():
    vocabulary = {k: k for k in range(252)}
    vocabulary['hallo'] = 253
    vocabulary['world'] = 254
    vocabulary[UKN] = 255
    tt = TermTranslator(vocabulary)

    v = tt.term_to_vector('hallo')
    assert len(v) == 8
    assert tt.vector_to_term(v) == 'hallo'

    v = tt.term_to_vector('world')
    assert len(v) == 8
    assert tt.vector_to_term(v) == 'world'

    v = tt.term_to_vector('juhu')
    assert len(v) == 8
    assert tt.vector_to_term(v) == UKN 

