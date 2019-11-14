#!/usr/bin/env python3

"""
generate HTML corpurs text (i.e. the xpaths) based on a corpus file
- note: this library reuses tree_util from path-extractor ai
"""

import lzma
import gzip
import tarfile
import logging
import re
import os.path

from pathextractorai.tree_util import get_dom_tree, dom_tree_to_xpaths

from json import load


RE_FILTER_XML_HEADER = re.compile("<\?xml version=\".*? encoding=.*?\?>")
CORPUS_FILE = '../embeddings/corpus/forum.tar.lzma'


class HtmlArchiveReader(object):

    def __init__(self, corpus_file):
        self.corpus_file = corpus_file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

    def __iter__(self):
        ''' iterates over the HTML archive and extracts all
            relevant word embeddings
        '''
        with lzma.open(self.corpus_file) as f:
            with tarfile.open(fileobj=f) as src:
                for fname in src.getmembers():
                    if not fname.isfile or not fname.name.endswith('.json'):
                        continue
                    self.logger.info("Processing file '%s'." % fname.name)
                    fobj = load(src.extractfile(fname))
                    yield from list(self.extract_features(fobj['html']))


    @staticmethod
    def normalize_text(txt):
        '''
        Normalizes text based on its length
        '''
        txt = txt.strip()
        if not txt:
            return ""
        elif len(txt) <= 5:
            return "*"
        elif len(txt) <= 10:
            return "**"
        elif len(txt) <= 20:
            return "***"
        elif len(txt) <= 40:
            return "****"
        return "*****"


    @staticmethod
    def extract_features(html):
        '''
        Extracts all relevant features from the given html file
        '''
        try:
            html = RE_FILTER_XML_HEADER.sub("", html)
            dom = get_dom_tree(html)
        except ValueError as e:
            print(html[:240])
            print(e)
            import sys
            sys.exit(-1)

        for element_list in dom_tree_to_xpaths(dom):
            yield element_list


def generate_textual_corpus():
    text_corpus = '\n'.join([' '.join(xpath) for xpath in HtmlArchiveReader(CORPUS_FILE)])
    with gzip.open('html_corpus.text.gz', 'w') as f:
        f.write(text_corpus.encode('utf-8'))

if __name__ == '__main__':

    if not os.path.exists('html_corpus.text.gz'):
        generate_textual_corpus()

