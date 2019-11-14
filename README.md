# Embeddings process

1. generate a file with the Xpath representations using `generate-html-corpus-texts.py`. the corresponding XPaths are stored in `html_corpus.txt.gz`.
2. use `biinput.py` to generate the corresponding binary training corpus and `bitrain.py` for training.
3. use `triinput.py` to generate the corresponding ternary training corpus and `tritrain.py` for training.
