# Embeddings process

1. cleanup: delete the following files to ensure that they are re computed
   - `html_corpus.txt.gz` ... contains the extracted training xpaths
   - `html_vocabulary.cvs.gz` ... the vocabulary to id mapping
   - `html_corpus.bin.gz` ... the binary version (translated using the vocabulary) of the html xpath corpus
2. generate a file with the Xpath representations using `generate-html-corpus-texts.py`. the corresponding XPaths are stored in `html_corpus.txt.gz`.
3. use `triinput.py` to generate (a) the html vocabulary file and (b) the html corpus file as well as the corresponding training corpora.
4. use `tritrain.py` for training the embeddings.
